#!/usr/bin/env python3
# viz_diffmaps_individual_clean.py
# Render per-pair difference maps (union of add+miss) in a single deep-red overlay.
# No titles, no axes. Nearly-square figure (width slightly larger than height).

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- IO ----------
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

# ---------- display helpers ----------
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0, out_lo=0.30, out_hi=0.92, gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

DEEP_RED_CMAP = ListedColormap([[0,0,0,0], [0.80, 0.00, 0.00, 1.0]])  # transparent -> deep red

# ---------- STFT geometry ----------
def hz_per_bin(n_freq_bins: int, sr: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def rows_for_fmax(arr_2d: np.ndarray, sr: int, fmax_hz: float) -> int:
    hz_bin = hz_per_bin(arr_2d.shape[0], sr)
    return max(1, int(round(fmax_hz / hz_bin)))

# ---------- meta resolver ----------
def load_stft_meta(suggested_root: Path, bona: str, voc: str) -> Tuple[int,int]:
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

# ---------- Mb/mask resolver ----------
def resolve_mb_and_masks(pair_dir: Path, stem: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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

    add_mask  = np.load(add_path)  if add_path  else None
    miss_mask = np.load(miss_path) if miss_path else None
    comb_mask = np.load(comb_path) if comb_path else None
    return Mb_db, add_mask, miss_mask, comb_mask

# ---------- drawing ----------
def draw_single(out_path: Path, Mb_db: np.ndarray, union_mask: Optional[np.ndarray],
                sr: int, hop: int, fmax: float, dpi: int,
                fig_w: float, fig_h: float):
    H, W = Mb_db.shape
    max_row = rows_for_fmax(Mb_db, sr, fmax)
    Mb_view = Mb_db[:max_row, :]
    m_view  = union_mask[:max_row, :] if union_mask is not None else None

    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    bg = paper_gray_from_db(Mb_view)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # background spectrogram
    ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
              cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    # union mask in deep red
    if m_view is not None:
        ax.imshow(np.ma.masked_where(~m_view.astype(bool), m_view.astype(float)),
                  extent=extent, aspect="auto", origin="lower",
                  cmap=DEEP_RED_CMAP, interpolation="nearest", vmin=0, vmax=1)

    # clean look: no axes/ticks/labels/spines
    ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # borderless save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(out_path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Render red-only union diff maps (no axes) from a JSON list.")
    ap.add_argument("--list-json", type=str, required=True,
                    help="Path to JSON list: [{bona_id, vocoder[, stem]}, ...]")
    ap.add_argument("--array-root", type=str, required=True,
                    help="Root with <BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--suggested-root", type=str, required=True,
                    help="Root with per-pair metadata for sr/hop: <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json}")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Directory to write per-pair PNG files")
    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap for display in Hz")
    ap.add_argument("--dpi", type=int, default=150)
    # Nearly-square defaults: width slightly larger than height
    ap.add_argument("--figw", type=float, default=5.4, help="Figure width in inches (default 5.4)")
    ap.add_argument("--figh", type=float, default=5.0, help="Figure height in inches (default 5.0)")
    ap.add_argument("--suffix", type=str, default="", help="Optional filename suffix before .png")
    args = ap.parse_args()

    items = load_json(Path(args.list_json))
    if not isinstance(items, list) or not items:
        raise SystemExit("JSON must be a non-empty list of {bona_id, vocoder[, stem]}")

    array_root = Path(args.array_root)
    suggested_root = Path(args.suggested_root)
    out_root = Path(args.out_dir)

    for idx, it in enumerate(items, start=1):
        bona = it.get("bona_id"); voc = it.get("vocoder")
        if not bona or not voc:
            print(f"[skip] item missing bona_id/vocoder: {it}")
            continue

        sr, hop = load_stft_meta(suggested_root, bona, voc)
        pair_dir = array_root / bona / voc
        stem = it.get("stem")

        Mb_db, add_mask, miss_mask, comb_mask = resolve_mb_and_masks(pair_dir, stem)
        if Mb_db is None:
            print(f"[warn] no Mb_* under {pair_dir} (stem={stem}), skipping {bona}/{voc}")
            continue

        # Build a single union mask in boolean
        union: Optional[np.ndarray] = None
        if comb_mask is not None:
            union = comb_mask.astype(bool)
        else:
            if add_mask is not None and miss_mask is not None:
                union = (add_mask.astype(bool) | miss_mask.astype(bool))
            elif add_mask is not None:
                union = add_mask.astype(bool)
            elif miss_mask is not None:
                union = miss_mask.astype(bool)

        suffix = (args.suffix.strip() and f"__{args.suffix.strip()}") or ""
        out_name = f"{idx:03d}__{bona}__{voc}{suffix}.png"
        out_path = out_root / out_name

        draw_single(out_path, Mb_db, union, sr, hop, args.fmax, dpi=args.dpi,
                    fig_w=args.figw, fig_h=args.figh)

if __name__ == "__main__":
    main()