# extract_mask_crops.py
# Build crops_manifest.json from a single mask95_smoothed__*.npy
# using dynamic crops (CC bbox + padding). No CLI, no PNGs.

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re, json
import numpy as np
from scipy.ndimage import (
    label, find_objects, binary_opening, binary_closing,
    binary_closing as morph_close, gaussian_filter
)

# ---------------- parsing ----------------
BONA_RX = re.compile(r"(LA_[TDE]_\d{7})")
VOC_BONA_RX = re.compile(r"([A-Za-z0-9\-]+)_(LA_[TDE]_\d{7})$")
MASK_REQUIRED_PREFIX = "mask95_smoothed__"

def parse_bona_vocoder_from_stem(stem: str) -> Dict[str, str]:
    m = VOC_BONA_RX.search(stem)
    if not m:
        m2 = BONA_RX.search(stem)
        if not m2:
            raise ValueError(f"Cannot parse bona/vocoder from '{stem}'")
        bona = m2.group(1)
        prefix = stem[:m2.start()].rstrip("_")
        voc = prefix.split("_")[-1] if prefix else "unknown"
        return {"vocoder": voc, "bona_id": bona}
    vocoder, bona_id = m.group(1), m.group(2)
    return {"vocoder": vocoder, "bona_id": bona_id}

def parse_from_filename(name: str) -> Dict[str, str]:
    meta = parse_bona_vocoder_from_stem(Path(name).stem)
    return {"bona_id": meta["bona_id"], "vocoder": meta["vocoder"],
            "spoof_stem": f'{meta["vocoder"]}_{meta["bona_id"]}'}

# ---------------- TextGrid (optional tags) ----------------
VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
SONORANTS = VOWELS | {"M","N","NG"} | {"L","R","W","Y","J"}
VOICED_OBS = {"B","D","G","V","DH","Z","ZH"}

def _core(lbl: str) -> str:
    return re.sub(r"\d$", "", (lbl or "").upper())

def _load_textgrid_phones(textgrid_root: Optional[Path], bona_id: str) -> List[Dict]:
    if not textgrid_root:
        return []
    tg_path = Path(textgrid_root) / bona_id / f"{bona_id}.TextGrid"
    if not tg_path.exists():
        return []
    txt = tg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i, n = 0, len(txt)
    phones: List[Dict] = []
    while i < n:
        if 'class = "IntervalTier"' in txt[i]:
            name = ""
            for j in range(i, min(i+10, n)):
                if txt[j].strip().startswith('name ='):
                    name = txt[j].split('=',1)[1].strip().strip('"').lower()
                    break
            if name in {"phones","phone","phoneme"}:
                j = i
                while j < n and 'intervals [' not in txt[j]:
                    j += 1
                while j < n and 'item [' not in txt[j]:
                    if 'intervals [' in txt[j]:
                        try:
                            xmin = float(txt[j+1].split('=')[1])
                            xmax = float(txt[j+2].split('=')[1])
                            lab  = txt[j+3].split('=',1)[1].strip().strip('"')
                            phones.append({"start": xmin, "end": xmax, "label": lab})
                            j += 4; continue
                        except Exception:
                            pass
                    j += 1
                break
        i += 1
    return phones

def _crop_phone_tags(phones: List[Dict],
                     t0_s: float, t1_s: float,
                     boundary_tol_ms: float = 25.0) -> Dict:
    if not phones:
        return {"boundary_near_mid": None, "voiced_fraction": None, "vowel_fraction": None}
    dur = max(1e-9, t1_s - t0_s)
    mid = 0.5 * (t0_s + t1_s)
    tol = boundary_tol_ms / 1000.0
    boundary_near_mid = False
    voiced_overlap = 0.0
    vowel_overlap = 0.0
    for ph in phones:
        s, e = float(ph["start"]), float(ph["end"])
        ovl = max(0.0, min(t1_s, e) - max(t0_s, s))
        if ovl > 0.0:
            lab = _core(ph["label"])
            if lab in SONORANTS or lab in VOICED_OBS:
                voiced_overlap += ovl
            if lab in VOWELS:
                vowel_overlap += ovl
        if abs(s - mid) <= tol or abs(e - mid) <= tol:
            boundary_near_mid = True
    return {
        "boundary_near_mid": bool(boundary_near_mid),
        "voiced_fraction": float(voiced_overlap / dur),
        "vowel_fraction": float(vowel_overlap / dur)
    }

# ---------------- utils ----------------
def as_bool_mask(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(bool)

def hz_per_bin(sr: int, n_freq_bins: int) -> float:
    return (sr/2.0) / max(1, n_freq_bins-1)

def frames_to_seconds(frames: int, hop: int, sr: int) -> float:
    return frames * hop / sr

def clamp(lo: int, hi: int, maxv: int) -> Tuple[int,int]:
    lo = max(0, lo); hi = min(maxv, hi)
    if lo >= hi: hi = min(maxv, lo+1)
    return lo, hi

def grow_slice(H: int, W: int,
               slc: Tuple[slice, slice],
               pad_t_frames: int,
               pad_f_bins: int) -> Tuple[slice, slice]:
    fs, ts = slc
    f0 = max(0, (fs.start or 0) - pad_f_bins)
    f1 = min(H, (fs.stop  or 0) + pad_f_bins)
    t0 = max(0, (ts.start or 0) - pad_t_frames)
    t1 = min(W, (ts.stop  or 0) + pad_t_frames)
    t0, t1 = clamp(t0, t1, W)
    f0, f1 = clamp(f0, f1, H)
    return slice(f0, f1), slice(t0, t1)

def slice_to_bbox_idx(fs: slice, ts: slice) -> Tuple[int,int,int,int]:
    return int(ts.start), int(ts.stop), int(fs.start), int(fs.stop)

def iou_bbox_idx(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    at0, at1, af0, af1 = a
    bt0, bt1, bf0, bf1 = b
    it = max(0, min(at1, bt1) - max(at0, bt0))
    ifr = max(0, min(af1, bf1) - max(af0, bf0))
    inter = it * ifr
    area_a = max(0, at1 - at0) * max(0, af1 - af0)
    area_b = max(0, bt1 - bt0) * max(0, bf1 - bf0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

# ---------------- canonical map + hysteresis (mask-only) ----------------
def _canonical_map_from_mask(mask_bool: np.ndarray) -> np.ndarray:
    """Light blur so CCs are more contiguous; used for ranking/hysteresis only."""
    return gaussian_filter(mask_bool.astype(np.float32), sigma=0.8)

def _hysteresis_ccs(canon: np.ndarray,
                    q_hi: float, q_lo: float,
                    min_area: int,
                    close_time: int = 3,
                    close_freq: int = 1) -> List[Tuple[slice, slice, float, float]]:
    finite = canon[np.isfinite(canon)]
    if finite.size == 0:
        return []
    tau_hi = float(np.quantile(finite, q_hi))
    tau_lo = float(np.quantile(finite, q_lo))
    seeds = canon >= tau_hi
    grow  = canon >= tau_lo
    if close_time > 1 or close_freq > 1:
        kernel = np.ones((max(1, close_freq), max(1, close_time)), bool)
        grow = morph_close(grow, structure=kernel)
    lab, nlab = label(grow, structure=np.ones((3,3), bool))
    if nlab == 0:
        return []
    slcs = find_objects(lab)
    out = []
    for lid, slc in enumerate(slcs, start=1):
        if slc is None:
            continue
        region = (lab[slc] == lid)
        if not np.any(seeds[slc] & region):
            continue
        area = float(region.sum())
        if area < float(min_area):
            continue
        s = float((canon[slc] * region).sum())
        fs, ts = slc
        out.append((fs, ts, area, s))
    out.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return out

def _auto_tune_hysteresis(canon: np.ndarray,
                          k_target: int,
                          min_area: int,
                          q_hi_init: float = 0.97,
                          q_lo_init: float = 0.90,
                          max_iter: int = 6) -> Tuple[List[Tuple[slice,slice,float,float]], float, float]:
    q_hi, q_lo = q_hi_init, q_lo_init
    k_min = max(2, int(0.75 * k_target))
    k_max = max(k_min + 1, int(6.0 * k_target))
    step_hi = 0.01
    step_lo = 0.01
    for _ in range(max_iter):
        ccs = _hysteresis_ccs(canon, q_hi, q_lo, min_area)
        n = len(ccs)
        if k_min <= n <= k_max:
            return ccs, q_hi, q_lo
        if n < k_min:
            q_hi = max(0.90, q_hi - step_hi)
            q_lo = max(0.80, q_lo - step_lo)
        else:
            q_hi = min(0.995, q_hi + step_hi)
            q_lo = min(q_hi - 0.02, q_lo + step_lo)
    ccs = _hysteresis_ccs(canon, q_hi, q_lo, min_area)
    return ccs, q_hi, q_lo

# ---------------- components + scoring + tags ----------------
def _components_with_scores_from_ccs(ccs: List[Tuple[slice, slice, float, float]],
                                     canon: np.ndarray,
                                     min_solidity: float) -> List[Dict]:
    out: List[Dict] = []
    for fs, ts, area, sum_canon in ccs:
        f0, f1 = fs.start or 0, fs.stop or 0
        t0, t1 = ts.start or 0, ts.stop or 0
        bbox_area = max(1, (f1 - f0) * (t1 - t0))
        solidity = float(area / bbox_area)
        if solidity < float(min_solidity):
            continue
        width = float((t1 - t0) + 1e-6)
        height = float((f1 - f0) + 1e-6)
        ar = width / height
        orient = "H" if ar > 1.2 else ("V" if ar < (1/1.2) else "N")
        out.append({
            "fs": fs, "ts": ts,
            "area": float(area),
            "sum_canon": float(sum_canon),
            "solidity": float(solidity),
            "orient": orient,
        })
    out.sort(key=lambda c: c["sum_canon"], reverse=True)
    return out

def _freq_band_tag(f0_idx: int, f1_idx: int, hzbin: float) -> str:
    cen = 0.5 * (f0_idx + f1_idx) * hzbin
    if cen < 1000.0: return "low"
    if cen <= 3000.0: return "mid"
    return "high"

# ---------------- selection with bucketed round-robin + NMS ----------------
def iou_bbox_idx(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    at0, at1, af0, af1 = a
    bt0, bt1, bf0, bf1 = b
    it = max(0, min(at1, bt1) - max(at0, bt0))
    ifr = max(0, min(af1, bf1) - max(af0, bf0))
    inter = it * ifr
    area_a = max(0, at1 - at0) * max(0, af1 - af0)
    area_b = max(0, bt1 - bt0) * max(0, bf1 - bf0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def _nms_keep(new_bb: Tuple[int,int,int,int],
              kept: List[Tuple[int,int,int,int]],
              iou_thr: float) -> bool:
    return all(iou_bbox_idx(new_bb, kb) <= iou_thr for kb in kept)

def _round_robin_select(cands: List[Dict],
                        H: int, W: int, hzbin: float,
                        k_per_pair: int,
                        iou_thr: Optional[float]) -> List[Dict]:
    buckets: Dict[Tuple[str,str,str], List[Dict]] = {}
    for c in cands:
        t0, t1 = c["ts"].start or 0, c["ts"].stop or 0
        f0, f1 = c["fs"].start or 0, c["fs"].stop or 0
        fb = _freq_band_tag(f0, f1, hzbin)
        buckets.setdefault((fb, c["orient"], "UNK"), []).append(c)
    for b in buckets.values():
        b.sort(key=lambda z: z["sum_canon"], reverse=True)

    keys = list(buckets.keys())
    sel: List[Dict] = []
    kept_bbs: List[Tuple[int,int,int,int]] = []
    i = 0
    while len(sel) < int(k_per_pair) and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]; i += 1
        if not buckets[k]: continue
        cand = buckets[k][0]
        t0, t1 = cand["ts"].start or 0, cand["ts"].stop or 0
        f0, f1 = cand["fs"].start or 0, cand["fs"].stop or 0
        bb = (t0, t1, f0, f1)
        if (iou_thr is None) or (iou_thr <= 0.0) or (iou_thr >= 1.0) or _nms_keep(bb, kept_bbs, float(iou_thr)):
            sel.append(buckets[k].pop(0)); kept_bbs.append(bb)
        else:
            buckets[k].pop(0)
    return sel

# ---------------- core (mask-only) ----------------
def _load_mask95_only(path: Path) -> np.ndarray:
    stem = path.stem
    if not stem.startswith(MASK_REQUIRED_PREFIX):
        raise ValueError(f"Expected a '{MASK_REQUIRED_PREFIX}*.npy' file, got '{path.name}'")
    A = np.load(path)
    uniq = np.unique(A)
    if not (A.dtype == np.bool_ or (uniq.size <= 3 and np.all(np.isin(uniq, [0, 1])))):
        raise ValueError(f"Input is not a binary mask (0/1); dtype={A.dtype}, unique={uniq}")
    return as_bool_mask(A)

def ensure_out_dirs(out_root: Path, bona_id: str, vocoder: str) -> Path:
    d = out_root / bona_id / vocoder
    d.mkdir(parents=True, exist_ok=True)
    return d

def process_file(
    npy_path: Path,
    out_root: Path,
    k_per_pair: int = 8,
    min_area: int = 30,
    min_solidity: float = 0.60,
    open_sz: int = 0,
    close_sz: int = 0,
    sr: int = 16000,
    hop: int = 256,
    pad_t_ms: float = 32.0,
    pad_f_hz: float = 300.0,
    nms_iou: Optional[float] = 0.70,
    q_hi_init: float = 0.97,
    q_lo_init: float = 0.90,
    close_time: int = 3,
    close_freq: int = 1,
    auto_tune: bool = True,
    textgrid_root: Optional[Path] = None,
    boundary_tol_ms: float = 25.0,
) -> Optional[Path]:
    """
    Mask-only entry point:
      - Require mask95_smoothed__*.npy
      - Build canonical soft map from the mask (gaussian blur)
      - Hysteresis (Q_hi/Q_lo) with optional auto-tuning → CCs
      - Geometry filter + bucketed round-robin + NMS
      - Optional phone tags
      - Write crops_manifest.json
    """
    meta = parse_from_filename(npy_path.name)
    bona_id, voc = meta["bona_id"], meta["vocoder"]

    # 1) load binary mask (required)
    mask_bool = _load_mask95_only(npy_path)
    H, W = mask_bool.shape[:2]

    # 2) canonical soft map from mask for ranking/hysteresis
    canon = _canonical_map_from_mask(mask_bool)

    # 3) hysteresis (+auto-tune) to get dense CCs
    if auto_tune:
        ccs, q_hi_used, q_lo_used = _auto_tune_hysteresis(
            canon, k_target=k_per_pair, min_area=min_area,
            q_hi_init=q_hi_init, q_lo_init=q_lo_init
        )
    else:
        ccs = _hysteresis_ccs(canon, q_hi_init, q_lo_init, min_area, close_time=close_time, close_freq=close_freq)
        q_hi_used, q_lo_used = q_hi_init, q_lo_init
    if not ccs:
        return None

    # 4) candidates with scores + orientation (no norm_s; rank by sum_canon)
    candidates = _components_with_scores_from_ccs(ccs, canon, min_solidity=min_solidity)
    if not candidates:
        return None

    out_dir = ensure_out_dirs(out_root, bona_id, voc)
    hzbin = hz_per_bin(sr, H)
    pad_t_frames = int(round((pad_t_ms/1000.0) * (sr / hop)))
    pad_f_bins   = int(round(pad_f_hz / max(hzbin, 1e-9)))

    # 5) selection + NMS
    selected_core = _round_robin_select(candidates, H, W, hzbin,
                                        k_per_pair=k_per_pair, iou_thr=nms_iou)
    if not selected_core:
        return None

    # 6) apply padding and serialize
    final = []
    for c in selected_core:
        fs_p, ts_p = grow_slice(H, W, (c["fs"], c["ts"]),
                                pad_t_frames=pad_t_frames, pad_f_bins=pad_f_bins)
        t0_idx, t1_idx, f0_idx, f1_idx = slice_to_bbox_idx(fs_p, ts_p)
        t0_s = frames_to_seconds(t0_idx, hop, sr)
        t1_s = frames_to_seconds(t1_idx, hop, sr)
        f0_hz, f1_hz = f0_idx * hzbin, f1_idx * hzbin
        final.append({
            "bbox_idx": [int(t0_idx), int(t1_idx), int(f0_idx), int(f1_idx)],
            "bbox_sec_hz": [float(t0_s), float(t1_s), float(f0_hz), float(f1_hz)],
            "score_sum_canon": float(c["sum_canon"]),
            "score_sum_norm": None,  # mask-only
            "area_bins": int(c["area"]),
            "solidity": float(c["solidity"]),
            "orientation": c["orient"],
            "freq_band": _freq_band_tag(int(f0_idx), int(f1_idx), hzbin),
        })

    # 7) optional phone tags
    phones = _load_textgrid_phones(textgrid_root, bona_id) if textgrid_root else []
    for c in final:
        if phones:
            t0_s, t1_s = c["bbox_sec_hz"][0], c["bbox_sec_hz"][1]
            c.update(_crop_phone_tags(phones, t0_s, t1_s, boundary_tol_ms=boundary_tol_ms))
        else:
            c.update({"boundary_near_mid": None, "voiced_fraction": None, "vowel_fraction": None})

    # 8) manifest
    manifest = {
        "bona_id": bona_id,
        "vocoder": voc,
        "spoof_stem": meta["spoof_stem"],
        "source_path": str(npy_path),
        "source_kind": "mask95_smoothed",
        "score_mode": "soft_sum",
        "mask_shape": [int(H), int(W)],
        "stft": {"sr": sr, "hop": hop, "hz_per_bin": hzbin},
        "padding": {"pad_t_ms": float(pad_t_ms), "pad_f_hz": float(pad_f_hz)},
        "nms": {"enabled": bool(nms_iou is not None and 0.0 < nms_iou < 1.0),
                "iou_thresh": float(nms_iou) if nms_iou is not None else None},
        "mask_from_norm_quantile": None,
        "hysteresis": {"q_hi_used": float(q_hi_used), "q_lo_used": float(q_lo_used),
                       "auto_tuned": bool(auto_tune)},
        "filters": {"min_area_bins": int(min_area), "min_solidity": float(min_solidity)},
        "selection": {"k_per_pair": int(k_per_pair), "strategy": "bucketed_round_robin"},
        "crops": []
    }

    final.sort(key=lambda z: z["score_sum_canon"], reverse=True)
    for rank, c in enumerate(final[:k_per_pair], start=1):
        manifest["crops"].append({
            "crop_id": f"k{rank:02d}",
            "rank": int(rank),
            "score_sum_norm": None,
            "score_sum_canon": c["score_sum_canon"],
            "score_area": float(c["area_bins"]),
            "area_bins": int(c["area_bins"]),
            "solidity": float(c["solidity"]),
            "orientation": c["orientation"],
            "freq_band": c["freq_band"],
            "boundary_near_mid": c["boundary_near_mid"],
            "voiced_fraction": c["voiced_fraction"],
            "vowel_fraction": c["vowel_fraction"],
            "bbox_idx": c["bbox_idx"],
            "bbox_sec_hz": c["bbox_sec_hz"]
        })

    out_dir = ensure_out_dirs(out_root, bona_id, voc)
    out_json = out_dir / "crops_manifest.json"
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_json