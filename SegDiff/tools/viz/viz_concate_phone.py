#!/usr/bin/env python3
# viz_bg_phone_sections.py
# Visualize diff maps with a Vowel/Consonant background tint from Praat TextGrids.
# - Diffs: deep-red only (union of add+miss). No phone-colored diffs.
# - Background: two soft tints (Vowel vs Consonant). Silence/unlabeled remains untinted.
#
# CLI example:
#   python viz_bg_phone_sections.py \
#     --grid-json /path/to/pairs.json \
#     --array-root /home/opc/difussionXAI/mask_outputs_crop_phonetier \
#     --suggested-root /home/opc/SegDiff/suggested_crops \
#     --textgrid-root /home/opc/aligned_textgrids \
#     --tier phones \
#     --fmax 6000 \
#     --rows 4 \
#     --legend \
#     --out /home/opc/SegDiff/phone_bg_grid.png

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------- Display helpers ----------------------
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0, out_lo=0.30, out_hi=0.92, gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

# Deep red overlay for diffs (union of add+miss)
DIFF_CMAP = ListedColormap([[0, 0, 0, 0], [0.95, 0.05, 0.05, 0.95]])

# Background tints for phones (low alpha so diffs stand out)
VOWEL_BG      = (1.00, 0.95, 0.20, 0.28)  # warm yellow, subtle
CONSONANT_BG  = (0.20, 0.85, 1.00, 0.28)  # cool cyan, subtle

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
_ARPA_APPROX = {"L","R","W","Y","HH"}  # grouped with consonants for background
_SILENCE = {"SIL","SP","SPN","NSN","Pau","PAU","#", ""}

def _norm_phone(ph: str) -> str:
    if ph is None:
        return ""
    p = ph.strip().upper().strip('"').strip("'")
    p = re.sub(r"\d+$", "", p)  # drop stress digits: AH0 -> AH
    return p

def phone_to_bg_class(ph: str) -> int:
    """
    Map a phone to background classes:
      0 = silence/unknown (no tint)
      1 = vowel
      2 = consonant/approximant
    """
    p = _norm_phone(ph)
    if p in _SILENCE or p == "":
        return 0
    if p in _ARPA_VOWELS:
        return 1
    # everything else (incl. approximants) is consonant for background tinting
    return 2

@lru_cache(maxsize=512)
def load_textgrid_intervals(textgrid_path: str) -> Dict[str, List[Tuple[float,float,str]]]:
    """
    Lightweight TextGrid reader (Praat long text + common short text).
    Returns: {tier_name: [(t0, t1, label), ...], ...}
    """
    p = Path(textgrid_path)
    tiers: Dict[str, List[Tuple[float,float,str]]] = {}
    if not p.exists():
        return tiers

    txt = p.read_text(encoding="utf-8", errors="ignore")

    # Long text format
    if "item [" in txt and "class =" in txt:
        for item_block in re.split(r"\n\s*item \[\d+\]:\s*\n", txt):
            if "class =" not in item_block:
                continue
            m_class = re.search(r'class\s*=\s*"([^"]+)"', item_block)
            m_name  = re.search(r'name\s*=\s*"([^"]+)"', item_block)
            cls = (m_class.group(1) if m_class else "").strip()
            name = (m_name.group(1) if m_name else cls).strip()
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

    # Short text format
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
    for cand in DEFAULT_TIER_CANDIDATES:
        if cand in tiers:
            return cand
    return next(iter(tiers.keys()), None)

def frame_bg_classes(bona_id: str, textgrid_root: Path, tier_name: Optional[str],
                     W: int, hop: int, sr: int) -> np.ndarray:
    """
    Return per-frame BG class ids of shape [W], mapping each time frame to:
      0 = no tint (silence/unknown), 1 = vowel, 2 = consonant
    """
    tg_path = textgrid_root / bona_id / f"{bona_id}.TextGrid"
    tiers = load_textgrid_intervals(str(tg_path))
    classes = np.zeros((W,), dtype=np.uint8)
    if not tiers:
        return classes
    tn = choose_tier(tiers, tier_name)
    if tn is None:
        return classes
    intervals = tiers.get(tn, [])
    if not intervals:
        return classes

    times = (np.arange(W) * hop) / float(sr)
    j = 0
    for i, t in enumerate(times):
        while j + 1 < len(intervals) and t >= intervals[j][1]:
            j += 1
        if j < len(intervals):
            t0, t1, lab = intervals[j]
            if t0 <= t < t1 or (i == W-1 and t == t1):
                classes[i] = phone_to_bg_class(lab)
    return classes

# ---------------------- Mb/mask resolver ----------------------
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

# ---------------------- Core drawing ----------------------
def draw_panel(ax, Mb_db: np.ndarray, add_mask: Optional[np.ndarray], miss_mask: Optional[np.ndarray],
               sr: int, hop: int, fmax: float, bona: str,
               textgrid_root: Path, tier_name: Optional[str], title: str):
    H, W = Mb_db.shape
    max_row = rows_for_fmax(Mb_db, sr, fmax)
    Mb_cut = Mb_db[:max_row, :]
    add_cut  = add_mask[:max_row, :]  if add_mask  is not None else None
    miss_cut = miss_mask[:max_row, :] if miss_mask is not None else None

    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    bg = paper_gray_from_db(Mb_cut)

    # 1) Base spectrogram
    ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
              cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    # 2) Background tint: vowel vs consonant (silence stays neutral)
    classes = frame_bg_classes(bona, textgrid_root, tier_name, W, hop, sr)
    class2d = np.broadcast_to(classes[np.newaxis, :], (max_row, W))
    bg_overlay = np.zeros((max_row, W, 4), dtype=float)

    # Vowels
    v_sel = (class2d == 1)
    if np.any(v_sel):
        bg_overlay[..., 0][v_sel] = VOWEL_BG[0]
        bg_overlay[..., 1][v_sel] = VOWEL_BG[1]
        bg_overlay[..., 2][v_sel] = VOWEL_BG[2]
        bg_overlay[..., 3][v_sel] = VOWEL_BG[3]
    # Consonants (incl. approximants)
    c_sel = (class2d == 2)
    if np.any(c_sel):
        bg_overlay[..., 0][c_sel] = CONSONANT_BG[0]
        bg_overlay[..., 1][c_sel] = CONSONANT_BG[1]
        bg_overlay[..., 2][c_sel] = CONSONANT_BG[2]
        bg_overlay[..., 3][c_sel] = CONSONANT_BG[3]

    ax.imshow(bg_overlay, extent=extent, aspect="auto", origin="lower", interpolation="nearest")

    # 3) Diff overlay (deep red) — union of add+miss
    union = None
    if add_cut is not None and miss_cut is not None:
        union = (add_cut.astype(bool) | miss_cut.astype(bool))
    elif add_cut is not None:
        union = add_cut.astype(bool)
    elif miss_cut is not None:
        union = miss_cut.astype(bool)

    if union is not None:
        ax.imshow(np.ma.masked_where(~union, union.astype(float)),
                  extent=extent, aspect="auto", origin="lower",
                  cmap=DIFF_CMAP, interpolation="nearest", vmin=0, vmax=1)

    # Axes cosmetics
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)

# ---------------------- Grid runner ----------------------
def run_grid(args):
    items = load_json(Path(args.grid_json))
    if not isinstance(items, list) or not items:
        raise SystemExit("grid-json must be a non-empty list of {bona_id, vocoder[, stem, title]}")

    array_root = Path(args.array_root)
    suggested_root = Path(args.suggested_root)
    textgrid_root = Path(args.textgrid_root)

    N = len(items)
    rows = int(args.rows)
    cols = (N + rows - 1) // rows

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

        title = it.get("title") or f"{bona} / {voc}"
        draw_panel(ax, Mb_db, add_mask, miss_mask, sr, hop, args.fmax, bona,
                   textgrid_root, args.tier, title)

        # Label density control
        W = Mb_db.shape[1]
        t_max = W * hop / sr
        if i // cols < rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time [s]")
        if i % cols != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Hz")

    # Hide extras
    for j in range(N, rows*cols):
        ax_list[j].axis("off")

    if args.legend:
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color=VOWEL_BG, label="Vowel region"),
            mpatches.Patch(color=CONSONANT_BG, label="Consonant region"),
            mpatches.Patch(color=(0.95,0.05,0.05,0.95), label="Diff (add/miss union)"),
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
    ap = argparse.ArgumentParser(description="Render diff maps with vowel/consonant background tints from TextGrids.")
    ap.add_argument("--grid-json", type=str, required=True, help="Path to JSON list of items [{bona_id, vocoder[, stem, title]}, ...]")
    ap.add_argument("--array-root", type=str, required=True, help="Arrays root: <root>/<BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--suggested-root", type=str, required=True, help="Root with <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json} for sr/hop")
    ap.add_argument("--textgrid-root", type=str, default="/home/opc/aligned_textgrids", help="Root with <BONA>/<BONA>.TextGrid")
    ap.add_argument("--tier", type=str, default=None, help="Tier name (tries phones/phoneme/phonemes/phone if omitted)")
    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap for display in Hz")
    ap.add_argument("--rows", type=int, default=4, help="Rows in the grid")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--legend", action="store_true", default=False, help="Show legend")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = ap.parse_args()
    run_grid(args)

if __name__ == "__main__":
    main()