# extract_mask_crops.py
# Build crops_manifest.json from a single .npy (mask or norm_s),
# using dynamic crops (CC bbox + padding). No CLI, no PNGs.

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re, json
import numpy as np
from scipy.ndimage import label, find_objects, binary_opening, binary_closing

# ---------------- parsing ----------------
BONA_RX = re.compile(r"(LA_[TDE]_\d{7})")
VOC_BONA_RX = re.compile(r"([A-Za-z0-9\-]+)_(LA_[TDE]_\d{7})$")  # last token before bona = vocoder

def parse_bona_vocoder_from_stem(stem: str) -> Dict[str, str]:
    """
    Robustly extract vocoder and bona_id from names like:
      - mask95_smoothed__hifigan_LA_D_1026868
      - norm_s__hifigan_LA_D_1026868
      - Ms_smooth__hifigan_LA_D_1026868
    """
    m = VOC_BONA_RX.search(stem)
    if not m:
        # Fallback to simpler: find bona_id and assume whatever precedes underscore is vocoder
        m2 = BONA_RX.search(stem)
        if not m2:
            raise ValueError(f"Cannot parse bona/vocoder from '{stem}'")
        bona = m2.group(1)
        # take the substring before bona, grab last token split by '_'
        prefix = stem[:m2.start()].rstrip("_")
        voc = prefix.split("_")[-1] if prefix else "unknown"
        return {"vocoder": voc, "bona_id": bona}
    vocoder, bona_id = m.group(1), m.group(2)
    return {"vocoder": vocoder, "bona_id": bona_id}

def parse_from_filename(name: str) -> Dict[str, str]:
    meta = parse_bona_vocoder_from_stem(Path(name).stem)
    return {
        "bona_id": meta["bona_id"],
        "vocoder": meta["vocoder"],
        "spoof_stem": f'{meta["vocoder"]}_{meta["bona_id"]}'
    }

# ---------------- utils ----------------
def as_bool_mask(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(bool)

def clean_mask(m: np.ndarray, open_sz: int, close_sz: int) -> np.ndarray:
    s_open = np.ones((open_sz, open_sz), bool) if open_sz and open_sz > 1 else None
    s_close = np.ones((close_sz, close_sz), bool) if close_sz and close_sz > 1 else None
    out = m
    if s_open is not None:
        out = binary_opening(out, structure=s_open)
    if s_close is not None:
        out = binary_closing(out, structure=s_close)
    return out

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
    """Return (t0, t1, f0, f1) integer indices (right-open)."""
    return int(ts.start), int(ts.stop), int(fs.start), int(fs.stop)

def iou_bbox_idx(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """
    IoU for [t0,t1,f0,f1] with right-open indices. Works in index space.
    """
    at0, at1, af0, af1 = a
    bt0, bt1, bf0, bf1 = b
    # intersection lengths (clamped at 0)
    it = max(0, min(at1, bt1) - max(at0, bt0))
    ifr = max(0, min(af1, bf1) - max(af0, bf0))
    inter = it * ifr
    area_a = max(0, at1 - at0) * max(0, af1 - af0)
    area_b = max(0, bt1 - bt0) * max(0, bf1 - bf0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

# ---------------- core ----------------
def _threshold_mask_from_norm(norm_s: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    flat = norm_s[np.isfinite(norm_s)].ravel()
    tau = np.quantile(flat, quantile) if flat.size else 0.0
    return norm_s > tau

def _infer_kind_and_arrays(path: Path,
                           bona_id: str,
                           vocoder: str,
                           norm_quantile: float) -> Tuple[np.ndarray, Optional[np.ndarray], str, Optional[float]]:
    """
    Returns (binary_mask, norm_s_or_None, score_mode, used_norm_quantile)
    - If file is float-like: treat as norm_s, derive mask@quantile, score by sum_norm.
    - If file is binary mask: try to find sibling norm_s__{voc}_{bona}.npy (score by sum_norm if found),
      else score by area. (used_norm_quantile=None)
    """
    A = np.load(path)
    # Heuristic for norm vs mask
    is_float = np.issubdtype(A.dtype, np.floating)
    looks_binary = np.unique(A).size <= 3 and np.all(np.isin(np.unique(A), [0,1]))
    if is_float and not looks_binary:
        norm_s = A.astype(np.float32, copy=False)
        mask = _threshold_mask_from_norm(norm_s, norm_quantile)
        return mask.astype(bool), norm_s, "sum_norm", float(norm_quantile)
    else:
        # binary mask input
        mask = as_bool_mask(A)
        # try sibling norm_s in same folder
        sib = path.parent / f"norm_s__{vocoder}_{bona_id}.npy"
        if sib.exists():
            norm_s = np.load(sib).astype(np.float32, copy=False)
            return mask, norm_s, "sum_norm", None
        return mask, None, "area", None

def _components_with_scores(mask: np.ndarray,
                            min_area: int,
                            norm_s: Optional[np.ndarray]) -> List[Tuple[float, float, float, int, Tuple[slice, slice]]]:
    """
    Label CCs and compute scores.
    Returns list of tuples: (score_primary, area, score_mean, label_id, slice)
    - If norm_s is provided: score_primary = sum(norm_s * region), score_mean = score/area
    - Else: score_primary = area, score_mean = 0 (unused)
    """
    lab, nlab = label(mask, structure=np.ones((3,3), bool))
    if nlab == 0:
        return []
    slcs = find_objects(lab)
    out = []
    for lid, slc in enumerate(slcs, start=1):
        if slc is None:
            continue
        region = (lab[slc] == lid)
        area = float(region.sum())
        if area < float(min_area):
            continue
        if norm_s is not None:
            # robust: mask out non-finite
            ns = norm_s[slc]
            ns = np.where(np.isfinite(ns), ns, 0.0)
            s = float((ns * region).sum())
            mean_s = s / max(1.0, area)
            out.append((s, area, mean_s, lid, slc))
        else:
            out.append((area, area, 0.0, lid, slc))
    # sort: by score_primary desc, then by score_mean desc
    out.sort(key=lambda x: (x[0], x[2]), reverse=True)
    return out

def ensure_out_dirs(out_root: Path, bona_id: str, vocoder: str) -> Path:
    d = out_root / bona_id / vocoder
    d.mkdir(parents=True, exist_ok=True)
    return d

def process_file(
    npy_path: Path,
    out_root: Path,
    k_per_pair: int = 8,
    min_area: int = 30,
    open_sz: int = 3,
    close_sz: int = 3,
    sr: int = 16000,
    hop: int = 256,
    pad_t_ms: float = 32.0,
    pad_f_hz: float = 300.0,
    norm_quantile: float = 0.95,
    nms_iou: Optional[float] = 0.70,   # set to None/<=0/>=1 to disable
) -> Optional[Path]:
    """
    Single-entry point:
      - Accept either mask or norm_s .npy
      - If norm_s: derive mask@quantile (default Q95), rank by sum(norm_s)
      - If mask: try sibling norm_s; else rank by area
      - Clean (open/close), light NMS on padded boxes, take top-K, write crops_manifest.json
    """
    meta = parse_from_filename(npy_path.name)
    bona_id, voc = meta["bona_id"], meta["vocoder"]

    mask_raw, norm_s, score_mode, used_q = _infer_kind_and_arrays(
        npy_path, bona_id, voc, norm_quantile
    )
    H, W = mask_raw.shape[:2]

    mask_clean = clean_mask(mask_raw, open_sz=open_sz, close_sz=close_sz) if (open_sz>1 or close_sz>1) else mask_raw
    comps = _components_with_scores(mask_clean, min_area=min_area, norm_s=norm_s)
    if not comps:
        return None

    out_dir = ensure_out_dirs(out_root, bona_id, voc)

    hzbin = hz_per_bin(sr, H)
    pad_t_frames = int(round((pad_t_ms/1000.0) * (sr / hop)))
    pad_f_bins   = int(round(pad_f_hz / max(hzbin, 1e-9)))

    # Prepare candidate boxes with padding applied (used for NMS and output)
    candidates = []
    for (score_primary, area, score_mean, _lid, slc) in comps:
        fs_p, ts_p = grow_slice(H, W, slc, pad_t_frames=pad_t_frames, pad_f_bins=pad_f_bins)
        bbox_idx = slice_to_bbox_idx(fs_p, ts_p)  # (t0, t1, f0, f1)
        candidates.append({
            "score_primary": float(score_primary),
            "score_mean": float(score_mean),
            "score_area": float(area),
            "slc": slc,
            "fs_p": fs_p,
            "ts_p": ts_p,
            "bbox_idx": bbox_idx
        })

    # Sort by score_primary desc, then score_mean desc (already sorted in comps,
    # but re-assert here after padding, just to be safe)
    candidates.sort(key=lambda z: (z["score_primary"], z["score_mean"]), reverse=True)

    # Greedy NMS on padded boxes (time x freq IoU)
    def _nms_enabled(x: Optional[float]) -> bool:
        return (x is not None) and (0.0 < x < 1.0)

    selected = []
    if _nms_enabled(nms_iou):
        kept_bboxes: List[Tuple[int,int,int,int]] = []
        for c in candidates:
            bb = c["bbox_idx"]
            # keep if it doesn't highly overlap with any already kept
            if all(iou_bbox_idx(bb, kb) <= float(nms_iou) for kb in kept_bboxes):
                selected.append(c)
                kept_bboxes.append(bb)
            if len(selected) >= int(k_per_pair):
                break
    else:
        selected = candidates[:k_per_pair]

    # Build manifest
    manifest = {
        "bona_id": bona_id,
        "vocoder": voc,
        "spoof_stem": meta["spoof_stem"],
        "source_path": str(npy_path),
        "source_kind": "norm_s" if norm_s is not None and score_mode == "sum_norm" and np.issubdtype(np.load(npy_path).dtype, np.floating) else "mask",
        "score_mode": score_mode,   # "sum_norm" or "area"
        "mask_shape": [int(H), int(W)],
        "stft": {"sr": sr, "hop": hop, "hz_per_bin": hzbin},
        "padding": {"pad_t_ms": float(pad_t_ms), "pad_f_hz": float(pad_f_hz)},
        "nms": {"enabled": bool(_nms_enabled(nms_iou)), "iou_thresh": float(nms_iou) if nms_iou is not None else None},
        "mask_from_norm_quantile": float(used_q) if used_q is not None else None,
        "crops": []
    }

    # Emit selected (up to K)
    for rank, c in enumerate(selected, start=1):
        t0_idx, t1_idx, f0_idx, f1_idx = c["bbox_idx"]
        t0_s = frames_to_seconds(t0_idx, hop, sr)
        t1_s = frames_to_seconds(t1_idx, hop, sr)
        f0_hz, f1_hz = f0_idx*hzbin, f1_idx*hzbin

        manifest["crops"].append({
            "crop_id": f"k{rank:02d}",
            "rank": int(rank),
            "score_sum_norm": float(c["score_primary"]) if score_mode == "sum_norm" else None,
            "score_mean_norm": float(c["score_mean"]) if score_mode == "sum_norm" else None,
            "score_area": float(c["score_area"]),
            "area_bins": int(c["score_area"]),
            "bbox_idx": [int(t0_idx), int(t1_idx), int(f0_idx), int(f1_idx)],      # [t0,t1,f0,f1]
            "bbox_sec_hz": [float(t0_s), float(t1_s), float(f0_hz), float(f1_hz)]  # [t0,t1,f0,f1]
        })

    out_json = out_dir / "crops_manifest.json"
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_json