#!/usr/bin/env python3
# viz_diffmaps_grid_red_only.py
# Plot difference maps (union of add+miss) with a single deep-red overlay.
# - No TextGrid usage / no phone backgrounds
# - No blue for misses (all diffs are the same red)
# - Show title and x/y axes (titles per panel)
#
# JSON format:
#   [
#     {"bona_id": "LA_D_1000265", "vocoder": "hifigan", "title": "fogging_vowel"},
#     {"bona_id": "...", "vocoder": "...", "stem": "optional_spoof_stem", "title": "optional"}
#   ]

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------- IO helpers ----------------------
def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def safe_find_first(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dirpath.glob(pat))
        if hits:
            return hits[0]
    return None

def safe_find_first_with_stem(dirpath: Path, stem: Optional[str],
                              stemmed: List[str], generic: List[str]) -> Optional[Path]:
    pats: List[str] = []
    if stem:
        pats.extend([p.format(stem=stem) for p in stemmed])
    pats.extend(generic)
    return safe_find_first(dirpath, pats)

# ---------------------- Display helpers ----------------------
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0, out_lo=0.30, out_hi=0.92, gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

DEEP_RED_CMAP = ListedColormap([[0, 0, 0, 0], [0.80, 0.00, 0.00, 1.0]])  # transparent -> deep red

# ---------------------- STFT geometry ----------------------
def hz_per_bin(n_freq_bins: int, sr: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def rows_for_fmax(arr_2d: np.ndarray, sr: int, fmax_hz: float) -> int:
    hz_bin = hz_per_bin(arr_2d.shape[0], sr)
    return max(1, int(round(fmax_hz / hz_bin)))

# ---------------------- sr/hop meta resolver ----------------------
def load_stft_meta(suggested_root: Path, bona: str, voc: str) -> Tuple[int, int]:
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

# ---------------------- Mb/mask resolver ----------------------
def resolve_mb_and_masks(pair_dir: Path, stem: Optional[str]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Returns (Mb_db, add_mask, miss_mask, comb_mask_if_present)
    """
    Mb_path = safe_find_first_with_stem(
        pair_dir, stem,
        stemmed=["Mb_db__{stem}.npy", "Mb_smooth__{stem}.npy"],
        generic=["Mb_db.npy", "Mb_smooth.npy", "Mb.npy"]
    )
    if not Mb_path:
        return None, None, None, None
    Mb_db = np.load(Mb_path)

    add_path = safe_find_first_with_stem(
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

    add_mask  = np.load(add_path)  if add_path  else None
    miss_mask = np.load(miss_path) if miss_path else None
    comb_mask = np.load(comb_path) if comb_path else None
    return Mb_db, add_mask, miss_mask, comb_mask

# ---------------------- Panel drawing ----------------------
def draw_panel(ax, Mb_db: np.ndarray, union_mask: Optional[np.ndarray],
               sr: int, hop: int, fmax: float, title: str):
    H, W = Mb_db.shape
    max_row = rows_for_fmax(Mb_db, sr, fmax)
    Mb_cut = Mb_db[:max_row, :]
    m_cut  = union_mask[:max_row, :] if union_mask is not None else None

    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    bg = paper_gray_from_db(Mb_cut)

    # base spectrogram
    ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
              cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    # deep-red union overlay (no blue)
    if m_cut is not None:
        ax.imshow(np.ma.masked_where(~m_cut.astype(bool), m_cut.astype(float)),
                  extent=extent, aspect="auto", origin="lower",
                  cmap=DEEP_RED_CMAP, interpolation="nearest", vmin=0, vmax=1)

    ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)
    ax.set_title(title, fontsize=10)

# ---------------------- Grid runner ----------------------
def run_grid(args):
    items = load_json(Path(args.grid_json))
    if not isinstance(items, list) or not items:
        raise SystemExit("grid-json must be a non-empty list of {bona_id, vocoder[, stem, title]}")

    array_root = Path(args.array_root)
    suggested_root = Path(args.suggested_root)

    N = len(items)
    rows = int(args.rows)
    cols = (N + rows - 1) // rows

    # ~4.6x3.2 in per panel (similar to your prior grids)
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

        Mb_db, add_mask, miss_mask, comb_mask = resolve_mb_and_masks(pair_dir, stem)
        if Mb_db is None:
            ax.text(0.5, 0.5, f"Missing Mb for\n{bona}/{voc}", ha="center", va="center", color="red")
            ax.axis("off"); continue

        # union of diffs (all red)
        if comb_mask is not None:
            union = comb_mask.astype(bool)
        else:
            if add_mask is not None and miss_mask is not None:
                union = (add_mask.astype(bool) | miss_mask.astype(bool))
            elif add_mask is not None:
                union = add_mask.astype(bool)
            elif miss_mask is not None:
                union = miss_mask.astype(bool)
            else:
                union = None

        title = it.get("title") or f"{bona} / {voc}"
        draw_panel(ax, Mb_db, union, sr, hop, args.fmax, title)

        # declutter labels: x on last row, y on first col
        W = Mb_db.shape[1]; t_max = W * hop / sr
        if i // cols < rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time [s]")
        if i % cols != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Frequency [Hz]")

    # hide any extra axes
    for j in range(N, rows * cols):
        ax_list[j].axis("off")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(out_path)

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="Grid of red-only diff maps with titles and axes (no TextGrid).")
    ap.add_argument("--grid-json", type=str, required=True,
                    help="JSON list: [{bona_id, vocoder[, stem, title]}, ...]")
    ap.add_argument("--array-root", type=str, required=True,
                    help="Arrays root: <root>/<BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--suggested-root", type=str, required=True,
                    help="Meta root for sr/hop: <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json}")
    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap (Hz)")
    ap.add_argument("--rows", type=int, default=3, help="Rows in the grid (columns auto)")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--out", type=str, required=True, help="Output PNG")
    args = ap.parse_args()
    run_grid(args)

if __name__ == "__main__":
    main()