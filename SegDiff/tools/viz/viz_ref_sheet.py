#!/usr/bin/env python3
# viz_concepts_side_by_side.py
# Render a grid of concept panels, one per (bona_id, vocoder, concept) item
# from a JSON list. Each panel shows the gray spectrogram with a single deep-red
# overlay for the union of add+miss differences. X/Y axes are shown; the title
# is the concept string only. The grid wraps automatically by --cols.

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
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0,
                       out_lo=0.30, out_hi=0.92, gamma=1.0):
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

# ---------------------- Meta resolver ----------------------
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

# ---------------------- Array/mask resolver ----------------------
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

# ---------------------- Grid drawing ----------------------
def draw_grid(items: List[Dict[str, Any]],
              array_root: Path, suggested_root: Path,
              out_path: Path, fmax: float, dpi: int,
              cols: int, panel_w: float, panel_h: float):
    """
    items: list of dicts with keys: bona_id, vocoder, concept, (optional) stem
    """
    N = len(items)
    if N == 0:
        raise SystemExit("No items to draw.")

    rows = (N + cols - 1) // cols
    fig_w = panel_w * cols
    fig_h = panel_h * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    ax_list = axes.flatten()

    for i, it in enumerate(items):
        ax = ax_list[i]
        bona = it.get("bona_id"); voc = it.get("vocoder")
        concept = it.get("concept", "concept")
        stem = it.get("stem")

        if not bona or not voc:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing bona_id/vocoder", ha="center", va="center", color="red")
            continue

        sr, hop = load_stft_meta(suggested_root, bona, voc)
        pair_dir = array_root / bona / voc

        Mb_db, add_mask, miss_mask, comb_mask = resolve_mb_and_masks(pair_dir, stem)
        if Mb_db is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No Mb for\n{bona}/{voc}", ha="center", va="center", color="red")
            continue

        # cap frequency and prep background
        H, W = Mb_db.shape
        max_row = rows_for_fmax(Mb_db, sr, fmax)
        Mb_cut = Mb_db[:max_row, :]
        t_max = W * hop / sr
        extent = [0.0, t_max, 0.0, fmax]
        bg = paper_gray_from_db(Mb_cut)

        ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
                  cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        # union mask (single deep red)
        union = None
        if comb_mask is not None:
            union = comb_mask[:max_row, :].astype(bool)
        else:
            if add_mask is not None and miss_mask is not None:
                union = (add_mask[:max_row, :].astype(bool) | miss_mask[:max_row, :].astype(bool))
            elif add_mask is not None:
                union = add_mask[:max_row, :].astype(bool)
            elif miss_mask is not None:
                union = miss_mask[:max_row, :].astype(bool)

        if union is not None:
            ax.imshow(np.ma.masked_where(~union, union.astype(float)),
                      extent=extent, aspect="auto", origin="lower",
                      cmap=DEEP_RED_CMAP, interpolation="nearest", vmin=0, vmax=1)

        # axes + title
        ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(str(concept), fontsize=11)

    # hide extra axes if any
    for j in range(N, rows * cols):
        ax_list[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(out_path)

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot concepts side-by-side from JSON [{bona_id, vocoder, concept[, stem]}]."
    )
    ap.add_argument("--grid-json", type=str, required=True,
                    help="Path to JSON list of items.")
    ap.add_argument("--array-root", type=str, required=True,
                    help="Root with <BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--suggested-root", type=str, required=True,
                    help="Root with per-pair sr/hop metadata: <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json}")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")

    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap in Hz")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--cols", type=int, default=3, help="Number of columns before wrapping")

    # Per-panel sizing (roughly like your example figure)
    ap.add_argument("--panel-w", type=float, default=4.3, help="Panel width (inches)")
    ap.add_argument("--panel-h", type=float, default=3.6, help="Panel height (inches)")
    args = ap.parse_args()

    items = load_json(Path(args.grid_json))
    if not isinstance(items, list) or not items:
        raise SystemExit("grid-json must be a non-empty list of objects.")

    draw_grid(items, Path(args.array_root), Path(args.suggested_root),
              Path(args.out), args.fmax, args.dpi,
              cols=max(1, int(args.cols)),
              panel_w=float(args.panel_w), panel_h=float(args.panel_h))

if __name__ == "__main__":
    main()