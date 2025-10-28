#!/usr/bin/env python3
# viz_concept_exemplars.py
# Visualize top-N concept exemplars produced by reference_exemplars_from_suggested.py
# Extended to color differences by phone class using Praat TextGrids.
#
# New:
#   --textgrid-root (default: /home/opc/aligned_textgrids)
#   --tier (default tries: phones, phoneme, phonemes, phone)
#   --color-mode phone|diff (default phone)
#   Subcommand: grid-from-json (render arbitrary list of {bona_id, vocoder[, stem, title]} panels)

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe

# ---------------------- Concepts (unchanged) ----------------------
CONCEPTS = [
    "fogging_vowel",
    "formant_attenuation",
    "fogging_fricative",
    "pseudo_formant",
    "concatenatedness",
    "coarticulatory_deficit",
    "hyperneatness",
    "hyperflat_prosody",
]

# ---------------------- Display helpers ----------------------
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0, out_lo=0.30, out_hi=0.92, gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

# Add/miss cmaps (you can bump alphas to 0.95 if you want stronger)
RED_MASK_CMAP  = ListedColormap([[0, 0, 0, 0], [0.80, 0.00, 0.00, 1.0]])
BLUE_MASK_CMAP = ListedColormap([[0, 0, 0, 0], [0.00, 0.40, 0.95, 1.0]])

# Phone-class overlay colors (RGBA)
PHONE_COLORS = {
    1: (0.95, 0.15, 0.95, 0.95),  # Vowel = bright magenta
    2: (0.30, 1.00, 0.20, 0.95),  # Approximant = bright lime
    3: (0.10, 0.90, 1.00, 0.95),  # Consonant = bright cyan
}

# ---------------------- IO ----------------------
def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def safe_find_first(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dirpath.glob(pat))
        if hits:
            return hits[0]
    return None

def safe_find_first_with_stem(dirpath: Path, stem: Optional[str], stemmed: List[str], generic: List[str]) -> Optional[Path]:
    pats: List[str] = []
    if stem:
        pats.extend([p.format(stem=stem) for p in stemmed])
    pats.extend(generic)
    return safe_find_first(dirpath, pats)

# ---------------------- Geometry ----------------------
def hz_per_bin(n_freq_bins: int, sr: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def rows_for_fmax(arr_2d: np.ndarray, sr: int, fmax_hz: float) -> int:
    hz_bin = hz_per_bin(arr_2d.shape[0], sr)
    return max(1, int(round(fmax_hz / hz_bin)))

def bbox_idx_to_sechz(bbox_idx: List[int], sr: int, hop: int, H: int) -> Tuple[float,float,float,float]:
    t0,t1,f0,f1 = [int(x) for x in bbox_idx]
    hz_bin = hz_per_bin(H, sr)
    return (t0 * hop / sr, t1 * hop / sr, f0 * hz_bin, f1 * hz_bin)

# ---------------------- STFT meta resolver ----------------------
def load_stft_meta(suggested_root: Path, bona: str, voc: str) -> Tuple[int,int]:
    # Try suggested_crops.json -> k_summary.json -> fallback
    sug = suggested_root / bona / voc / "suggested_crops.json"
    if sug.exists():
        try:
            j = load_json(sug)
            stft = j.get("stft") or {}
            return int(stft.get("sr", 16000)), int(stft.get("hop", 256))
        except Exception:
            pass
    ksum = suggested_root / bona / voc / "k_summary.json"
    if ksum.exists():
        try:
            j = load_json(ksum)
            stft = j.get("stft") or {}
            return int(stft.get("sr", 16000)), int(stft.get("hop", 256))
        except Exception:
            pass
    return 16000, 256

# ---------------------- TextGrid parsing + phone mapping ----------------------
DEFAULT_TIER_CANDIDATES = ["phones", "phoneme", "phonemes", "phone"]

_ARPA_VOWELS = {
    "AA","AE","AH","AO","AW","AY","EH","ER","EY",
    "IH","IY","OW","OY","UH","UW","AX","IX","AXR","UX"
}
_ARPA_APPROX = {"L","R","W","Y","HH"}  # include HH as approximant-ish for our purposes
_SILENCE = {"SIL","SP","SPN","NSN","Pau","PAU","#", ""}

def _norm_phone(ph: str) -> str:
    if ph is None:
        return ""
    p = ph.strip().upper().strip('"').strip("'")
    # drop stress digits: AH0 -> AH
    p = re.sub(r"\d+$", "", p)
    return p

def phone_to_class(ph: str) -> int:
    """
    Map a (likely ARPAbet) phone to {0:unknown/skip, 1:vowel, 2:approximant, 3:consonant}
    """
    p = _norm_phone(ph)
    if p in _SILENCE:
        return 0
    if p in _ARPA_VOWELS:
        return 1
    if p in _ARPA_APPROX:
        return 2
    if p == "":
        return 0
    # everything else treated as consonant
    return 3

@lru_cache(maxsize=512)
def load_textgrid_intervals(textgrid_path: str) -> Dict[str, List[Tuple[float,float,str]]]:
    """
    Very small, dependency-free TextGrid reader.
    Returns {tier_name: [(t0, t1, label), ...], ...}
    Supports Praat long text and (common) short text formats.
    """
    p = Path(textgrid_path)
    tiers: Dict[str, List[Tuple[float,float,str]]] = {}
    if not p.exists():
        return tiers

    txt = p.read_text(encoding="utf-8", errors="ignore")

    # Heuristic: detect long format "item [n]:" blocks
    if "item [" in txt and "class =" in txt:
        # Long format
        for item_block in re.split(r"\n\s*item \[\d+\]:\s*\n", txt):
            if "class =" not in item_block:
                continue
            m_class = re.search(r'class\s*=\s*"([^"]+)"', item_block)
            m_name  = re.search(r'name\s*=\s*"([^"]+)"', item_block)
            cls = (m_class.group(1) if m_class else "").strip()
            name = (m_name.group(1) if m_name else cls).strip()
            # only IntervalTier has intervals
            if cls.lower() != "intervaltier":
                continue
            ints: List[Tuple[float,float,str]] = []
            for ib in re.split(r"\n\s*intervals \[\d+\]:\s*\n", item_block):
                m_xmin = re.search(r"\bxmin\s*=\s*([0-9.]+)", ib)
                m_xmax = re.search(r"\bxmax\s*=\s*([0-9.]+)", ib)
                m_txt  = re.search(r'\btext\s*=\s*"(.*)"', ib)
                if m_xmin and m_xmax and m_txt is not None:
                    t0 = float(m_xmin.group(1)); t1 = float(m_xmax.group(1))
                    lab = m_txt.group(1)
                    ints.append((t0, t1, lab))
            if ints:
                tiers[name] = ints
        return tiers

    # Short text format: tiers with "intervals:" listing
    block_re = re.compile(r'"IntervalTier"\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*("intervals:"|intervals:)\s*size\s*=\s*(\d+)', re.IGNORECASE)
    pos = 0
    while True:
        m = block_re.search(txt, pos)
        if not m:
            break
        name = m.group(1)
        size = int(m.group(5))
        pos = m.end()
        ints: List[Tuple[float,float,str]] = []
        for _ in range(size):
            mi = re.search(r'intervals\s*\[\d+\]\s*:\s*xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*"(.*)"', txt[pos:], re.IGNORECASE)
            if not mi:
                break
            t0 = float(mi.group(1)); t1 = float(mi.group(2)); lab = mi.group(3)
            ints.append((t0, t1, lab))
            pos += mi.end()
        if ints:
            tiers[name] = ints
    return tiers

def choose_tier(tiers: Dict[str, List[Tuple[float,float,str]]], user_tier: Optional[str]) -> Optional[str]:
    if user_tier and user_tier in tiers:
        return user_tier
    # try common names
    for cand in DEFAULT_TIER_CANDIDATES:
        if cand in tiers:
            return cand
    # fallback: first tier
    return next(iter(tiers.keys()), None)

def frame_classes_from_textgrid(bona_id: str, textgrid_root: Path, tier_name: Optional[str],
                                W: int, hop: int, sr: int) -> np.ndarray:
    """
    Return per-frame class ids of shape [W], mapping each time frame to {0,1,2,3}
    """
    tg_path = textgrid_root / bona_id / f"{bona_id}.TextGrid"
    tiers = load_textgrid_intervals(str(tg_path))
    if not tiers:
        return np.zeros((W,), dtype=np.uint8)
    tn = choose_tier(tiers, tier_name)
    if tn is None:
        return np.zeros((W,), dtype=np.uint8)
    intervals = tiers.get(tn, [])
    times = (np.arange(W) * hop) / float(sr)
    classes = np.zeros((W,), dtype=np.uint8)
    if not intervals:
        return classes
    # sweep intervals
    j = 0
    for i, t in enumerate(times):
        while j + 1 < len(intervals) and t >= intervals[j][1]:
            j += 1
        if j < len(intervals):
            t0, t1, lab = intervals[j]
            if t0 <= t < t1 or (i == W-1 and t == t1):
                classes[i] = phone_to_class(lab)
    return classes

# ---------------------- Drawing with phone overlay ----------------------
def _overlay_by_phone(ax, Mb_view: np.ndarray, add_mask: Optional[np.ndarray], miss_mask: Optional[np.ndarray],
                      sr: int, hop: int, fmax: float, bona_id: str,
                      textgrid_root: Path, tier_name: Optional[str], show_legend: bool):
    """
    Draw ONLY the phone-class overlay, assuming the gray spectrogram and (optionally) diff masks
    have already been drawn. This function does not touch the background.
    We recolor only where diffs exist and where a phone label is known.
    """
    H, W = Mb_view.shape
    if H == 0 or W == 0:
        return

    # union of diffs (we recolor only where diffs exist)
    if add_mask is None and miss_mask is None:
        return
    if add_mask is not None and miss_mask is not None:
        union = (add_mask.astype(bool) | miss_mask.astype(bool))
    elif add_mask is not None:
        union = add_mask.astype(bool)
    else:
        union = miss_mask.astype(bool)

    # Per-frame phone class -> broadcast to H
    frame_classes = frame_classes_from_textgrid(bona_id, textgrid_root, tier_name, W, hop, sr)
    class2d = np.broadcast_to(frame_classes[np.newaxis, :], (H, W))

    # Build RGBA overlay only for known classes (1..3). Silence/unknown (0) won't be recolored.
    overlay = np.zeros((H, W, 4), dtype=float)
    for cid, color in PHONE_COLORS.items():
        sel = (class2d == cid) & union
        if np.any(sel):
            r, g, b, a = color
            overlay[..., 0][sel] = r
            overlay[..., 1][sel] = g
            overlay[..., 2][sel] = b
            overlay[..., 3][sel] = a

    # Use the same extent as the main image
    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    ax.imshow(overlay, extent=extent, aspect="auto", origin="lower", interpolation="nearest")

    if show_legend:
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color=PHONE_COLORS[1], label="Vowel"),
            mpatches.Patch(color=PHONE_COLORS[2], label="Approximant"),
            mpatches.Patch(color=PHONE_COLORS[3], label="Consonant"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.6)

def draw_one_crop(out_path: Path, Mb_db: np.ndarray, add_mask: Optional[np.ndarray], miss_mask: Optional[np.ndarray],
                  sr: int, hop: int, fmax: float, crop: Dict[str, Any], title: str, dpi: int,
                  color_mode: str,
                  textgrid_root: Path, tier_name: Optional[str], bona_id: Optional[str],
                  show_legend: bool):
    H, W = Mb_db.shape
    max_row = rows_for_fmax(Mb_db, sr, fmax)
    Mb_view = Mb_db[:max_row, :]
    if add_mask is not None:  add_mask_v  = add_mask[:max_row, :]
    else:                     add_mask_v  = None
    if miss_mask is not None: miss_mask_v = miss_mask[:max_row, :]
    else:                     miss_mask_v = None

    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    bg = paper_gray_from_db(Mb_view)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
              cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    if color_mode == "diff":
        if miss_mask_v is not None:
            ax.imshow(np.ma.masked_where(~miss_mask_v.astype(bool), miss_mask_v.astype(float)),
                      extent=extent, aspect="auto", origin="lower",
                      cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
        if add_mask_v is not None:
            ax.imshow(np.ma.masked_where(~add_mask_v.astype(bool), add_mask_v.astype(float)),
                      extent=extent, aspect="auto", origin="lower",
                      cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    else:
        # Draw diffs EVERYWHERE first so silence/unlabeled frames are visible.
        if miss_mask_v is not None:
            ax.imshow(np.ma.masked_where(~miss_mask_v.astype(bool), miss_mask_v.astype(float)),
                      extent=extent, aspect="auto", origin="lower",
                      cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
        if add_mask_v is not None:
            ax.imshow(np.ma.masked_where(~add_mask_v.astype(bool), add_mask_v.astype(float)),
                      extent=extent, aspect="auto", origin="lower",
                      cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
        # Then recolor ONLY frames with phone labels
        if bona_id is not None:
            _overlay_by_phone(ax, Mb_view, add_mask_v, miss_mask_v, sr, hop, fmax,
                              bona_id, textgrid_root, tier_name, show_legend)

    # Crop rectangle
    if crop.get("bbox_sec_hz"):
        t0, t1, f0, f1 = [float(x) for x in crop["bbox_sec_hz"]]
    else:
        t0, t1, f0, f1 = bbox_idx_to_sechz(crop["bbox_idx"], sr, hop, H)

    # clip to display region
    f0c, f1c = max(0.0, min(f0, fmax)), max(0.0, min(f1, fmax))
    t0c, t1c = max(0.0, min(t0, t_max)), max(0.0, min(t1, t_max))
    if not (f1c <= 0 or f0c >= fmax or t1c <= 0 or t0c >= t_max):
        rect = Rectangle((t0c, f0c), max(1e-6, t1c - t0c), max(1e-6, f1c - f0c),
                 lw=2.2, ec=(1, 1, 1), fc=(0,0,0,0))
        ax.add_patch(rect)

    # label
    cname = crop.get("concept", "concept")
    conf = float(crop.get("confidence", 0.0))
    lbl = f"{cname} ({conf*100:.1f}%)"
    tx = t0c + 0.01 * (t1c - t0c + 1e-6)
    ty = f0c + 0.05 * (f1c - f0c + 1e-6)
    ax.text(tx, ty, lbl, fontsize=10, color="white",
        bbox=dict(facecolor=(0,0,0,0.45), edgecolor="none", pad=2.5),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(out_path)

# ---------------------- concept JSON discovery ----------------------
def find_top_file(concept_root: Path, concept: str) -> Optional[Path]:
    cands = sorted(concept_root.glob(f"{concept}_top*.json"))
    if not cands:
        return None
    def n_of(p: Path):
        m = re.search(r"_top(\d+)\.json$", p.name)
        return int(m.group(1)) if m else -1
    cands.sort(key=n_of, reverse=True)
    return cands[0]

def read_items_from_json(p: Path) -> List[Dict]:
    try:
        data = load_json(p)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for k in ("items", "topN", "topn", "exemplars", "hits", "crops"):
        v = data.get(k)
        if isinstance(v, list):
            return v
    return []

# ---------------------- Shared Mb/mask resolver ----------------------
def resolve_mb_and_masks(pair_dir: Path, stem: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    Mb_path = safe_find_first_with_stem(
        pair_dir, stem,
        stemmed=["Mb_db__{stem}.npy", "Mb_smooth__{stem}.npy"],
        generic=["Mb_db.npy", "Mb_smooth.npy", "Mb.npy"]
    )
    if not Mb_path:
        return None, None, None
    Mb_db = np.load(Mb_path)

    add_path  = safe_find_first_with_stem(
        pair_dir, stem,
        stemmed=["mask95_add_smoothed__{stem}.npy", "mask95_add__{stem}.npy"],
        generic=["mask95_add_smoothed.npy", "mask95_add_smoothed__*.npy", "add__*.npy", "add.npy"]
    )
    miss_path = safe_find_first_with_stem(
        pair_dir, stem,
        stemmed=["mask95_miss_smoothed__{stem}.npy", "mask95_miss__{stem}.npy"],
        generic=["mask95_miss_smoothed.npy", "mask95_miss_smoothed__*.npy", "miss__*.npy", "miss.npy"]
    )
    comb_path = safe_find_first_with_stem(
        pair_dir, stem,
        stemmed=["mask95_smoothed__{stem}.npy"],
        generic=["mask95_smoothed.npy", "mask95_smoothed__*.npy"]
    )

    add_mask  = np.load(add_path) if add_path else None
    miss_mask = np.load(miss_path) if miss_path else None
    if add_mask is None and miss_mask is None and comb_path:
        add_mask = np.load(comb_path)
        miss_mask = np.zeros_like(add_mask)
    return Mb_db, add_mask, miss_mask

# ---------------------- Main exemplar flow ----------------------
def run_exemplars(args):
    concept_root = Path(args.concept_json_root)
    suggested_root = Path(args.suggested_root)
    crops_root = Path(args.crops_root)
    array_root = Path(args.array_root)
    textgrid_root = Path(args.textgrid_root)

    for concept in CONCEPTS:
        top_file = find_top_file(concept_root, concept)
        items: List[Dict] = []
        src_label = None

        if top_file and top_file.exists():
            items = read_items_from_json(top_file)
            src_label = top_file.name

        if not items:
            fallback = concept_root / f"{concept}_95plus.json"
            if fallback.exists():
                items = read_items_from_json(fallback)
                items = sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                src_label = fallback.name

        if not items:
            print(f"[warn] no items for {concept} (looked for {concept}_top*.json / {concept}_95plus.json)")
            continue

        if args.debug:
            print(f"[debug] {concept}: using {src_label} with {len(items)} items")

        items = items[:args.take_first]
        out_dir = concept_root / concept
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, it in enumerate(items, start=1):
            bona = it.get("bona_id"); voc = it.get("vocoder")
            if not bona or not voc:
                print(f"[warn] {concept}: missing bona/vocoder in item (skipping): {it}")
                continue

            sr, hop = load_stft_meta(suggested_root, bona, voc)
            pair_dir = array_root / bona / voc

            # Resolve spoof stem if present in K manifest
            stem: Optional[str] = None
            K = it.get("K")
            if K is not None:
                mani = crops_root / f"k{int(K)}" / bona / voc / "crops" / "crops_manifest.json"
                if mani.exists():
                    try:
                        m = load_json(mani)
                        s = (m.get("spoof_stem") or "").strip()
                        stem = s if s else None
                    except Exception:
                        stem = None

            Mb_db, add_mask, miss_mask = resolve_mb_and_masks(pair_dir, stem)
            if Mb_db is None:
                print(f"[warn] {concept}: No Mb_* found under {pair_dir} (stem={stem}), skipping {bona}/{voc}")
                continue

            crop = {
                "concept": concept,
                "confidence": float(it.get("confidence", 0.0)),
                "bbox_idx": it.get("bbox_idx"),
                "bbox_sec_hz": it.get("bbox_sec_hz"),
            }

            title = f"{concept} — {bona}/{voc} — {crop['confidence']*100:.1f}%"
            conf_str = f"{crop['confidence']:.6f}".rstrip("0").rstrip(".")
            out_name = f"{idx:02d}__{concept}__{bona}__{voc}__conf{conf_str}.png"
            out_path = out_dir / out_name

            draw_one_crop(
                out_path, Mb_db, add_mask, miss_mask, sr, hop, args.fmax, crop, title, dpi=args.dpi,
                color_mode=args.color_mode,
                textgrid_root=textgrid_root, tier_name=args.tier, bona_id=bona,
                show_legend=args.legend
            )

# ---------------------- Grid-from-JSON subcommand ----------------------
def run_grid_from_json(args):
    """
    JSON schema: a list of objects
      [
        {"bona_id": "LA_D_1000265", "vocoder": "hifigan", "stem": "optional_spoof_stem", "title": "optional"},
        ...
      ]
    """
    items = load_json(Path(args.grid_json))
    if not isinstance(items, list) or not items:
        raise SystemExit("grid-from-json: JSON must be a non-empty list of {bona_id, vocoder[, stem, title]}")

    array_root = Path(args.array_root)
    suggested_root = Path(args.suggested_root)
    textgrid_root = Path(args.textgrid_root)

    N = len(items)
    rows = int(args.rows)
    cols = (N + rows - 1) // rows

    # rough sizing: each panel ~ (4.6 x 3.2) inches
    fig_w = 4.6 * cols
    fig_h = 3.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    ax_list = axes.flatten()

    for i, it in enumerate(items):
        ax = ax_list[i]
        bona = it.get("bona_id"); voc = it.get("vocoder")
        if not bona or not voc:
            ax.axis("off"); continue

        sr, hop = load_stft_meta(suggested_root, bona, voc)
        pair_dir = array_root / bona / voc
        stem = it.get("stem")

        Mb_db, add_mask, miss_mask = resolve_mb_and_masks(pair_dir, stem)
        if Mb_db is None:
            ax.text(0.5, 0.5, f"Missing Mb for\n{bona}/{voc}", ha="center", va="center", color="red")
            ax.axis("off"); continue

        # cap frequency
        H, W = Mb_db.shape
        max_row = rows_for_fmax(Mb_db, sr, args.fmax)
        Mb_cut = Mb_db[:max_row, :]
        add_cut  = add_mask[:max_row, :]  if add_mask  is not None else None
        miss_cut = miss_mask[:max_row, :] if miss_mask is not None else None

        t_max = W * hop / sr
        extent = [0.0, t_max, 0.0, args.fmax]
        bg = paper_gray_from_db(Mb_cut)

        ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
                  cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        if args.color_mode == "diff":
            if miss_cut is not None:
                ax.imshow(np.ma.masked_where(~miss_cut.astype(bool), miss_cut.astype(float)),
                          extent=extent, aspect="auto", origin="lower",
                          cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
            if add_cut is not None:
                ax.imshow(np.ma.masked_where(~add_cut.astype(bool), add_cut.astype(float)),
                          extent=extent, aspect="auto", origin="lower",
                          cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
        else:
            # Draw diffs everywhere first
            if miss_cut is not None:
                ax.imshow(np.ma.masked_where(~miss_cut.astype(bool), miss_cut.astype(float)),
                          extent=extent, aspect="auto", origin="lower",
                          cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
            if add_cut is not None:
                ax.imshow(np.ma.masked_where(~add_cut.astype(bool), add_cut.astype(float)),
                          extent=extent, aspect="auto", origin="lower",
                          cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
            # Then recolor labeled frames
            _overlay_by_phone(ax, Mb_cut, add_cut, miss_cut, sr, hop, args.fmax,
                              bona, textgrid_root, args.tier, show_legend=False)

        ttl = it.get("title") or f"{bona} / {voc}"
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, args.fmax)
        # lighter labels to reduce clutter
        if i // cols < rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time [s]")
        if i % cols != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Hz")

    # hide extra axes
    for j in range(N, rows*cols):
        ax_list[j].axis("off")

    if args.legend:
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color=PHONE_COLORS[1], label="Vowel"),
            mpatches.Patch(color=PHONE_COLORS[2], label="Approximant"),
            mpatches.Patch(color=PHONE_COLORS[3], label="Consonant"),
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.6)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(out_path)

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize concept exemplars, with optional phone-class overlays, and produce grids from JSON.")
    sub = ap.add_subparsers(dest="cmd", required=False)

    # Exemplar mode (default)
    ap.add_argument("--concept-json-root", type=str,
                    help="Folder with <concept>_topN.json and/or <concept>_95plus.json")
    ap.add_argument("--suggested-root", type=str,
                    help="Root with per-pair metadata: <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json}")
    ap.add_argument("--crops-root", type=str, default="./crops_sweep",
                    help="Root with k{K}/<BONA>/<VOCODER>/crops/crops_manifest.json (for spoof_stem resolution)")
    ap.add_argument("--array-root", type=str,
                    help="Arrays root: <root>/<BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--take-first", type=int, default=10, help="Render only the first N items per concept.")
    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap for display in Hz.")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--debug", action="store_true", default=False)

    # Phone overlay options
    ap.add_argument("--textgrid-root", type=str, default="/home/opc/aligned_textgrids",
                    help="Root with <BONA>/<BONA>.TextGrid")
    ap.add_argument("--tier", type=str, default=None, help="Tier name (default tries phones/phoneme/phonemes/phone)")
    ap.add_argument("--color-mode", type=str, choices=["phone","diff"], default="phone",
                    help=("phone = draw add/miss everywhere (red/blue), then recolor only frames with a phone label "
                          "(Vowel/Approximant/Consonant); diff = just red/blue add/miss"))
    ap.add_argument("--legend", action="store_true", default=False, help="Show legend for phone classes")

    # Grid subcommand
    g = sub.add_parser("grid-from-json", help="Render a 4-row grid from a JSON list of {bona_id, vocoder[, stem, title]}")
    g.add_argument("--grid-json", type=str, required=True, help="Path to JSON list of items")
    g.add_argument("--array-root", type=str, required=True, help="Arrays root")
    g.add_argument("--suggested-root", type=str, required=True, help="Suggested/meta root for sr/hop")
    g.add_argument("--textgrid-root", type=str, default="/home/opc/aligned_textgrids")
    g.add_argument("--tier", type=str, default=None)
    g.add_argument("--color-mode", type=str, choices=["phone","diff"], default="phone")
    g.add_argument("--fmax", type=float, default=6000.0)
    g.add_argument("--rows", type=int, default=4)
    g.add_argument("--dpi", type=int, default=140)
    g.add_argument("--legend", action="store_true", default=False)
    g.add_argument("--out", type=str, required=True, help="Output PNG path")

    args = ap.parse_args()

    if args.cmd == "grid-from-json":
        run_grid_from_json(args)
    else:
        # exemplar mode requires these roots
        if not (args.concept_json_root and args.suggested_root and args.array_root):
            raise SystemExit("Exemplar mode needs --concept-json-root, --suggested-root, and --array-root")
        run_exemplars(args)

if __name__ == "__main__":
    main()