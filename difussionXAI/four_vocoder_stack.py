#!/usr/bin/env python3
# four_vocoder_stack.py
# Make 4×5 sheets: columns are audios, rows are 4 vocoders (stacked).
# Order columns by audio length.

import argparse, re, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# constants from your generator
TARGET_SR = 16000
HOP = 256

BONA_RX = re.compile(r"^(LA_[TDE]_\d{7})$")

def bona_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and BONA_RX.match(p.name)])

def find_png_for(bona_dir: Path, vocoder: str) -> Optional[Path]:
    """Look under <bona>/<vocoder>/ for smoothed_mask95__*_{BONA}.png."""
    vdir = bona_dir / vocoder
    if not vdir.exists(): return None
    # safest: match suffix with bona id
    cand = list(vdir.glob(f"smoothed_mask95__*_{bona_dir.name}.png"))
    if cand: return sorted(cand)[0]
    # fallback: any smoothed png
    cand = list(vdir.glob("smoothed_mask95__*.png"))
    return sorted(cand)[0] if cand else None

def find_frames_for_duration(bona_dir: Path) -> Optional[int]:
    """
    Estimate audio frames for sorting by taking the MAX across vocoders.
    Prefers Mb_smooth.npy (real), else Ms_smooth__*.npy, else PNG width.
    """
    best = 0

    for vdir in bona_dir.iterdir():
        if not vdir.is_dir():
            continue

        # Prefer Mb_smooth.npy (may be truncated per vocoder; we take the max)
        mbs = vdir / "Mb_smooth.npy"
        if mbs.exists():
            try:
                arr = np.load(mbs, mmap_mode="r")
                best = max(best, int(arr.shape[1]))
            except Exception:
                pass

        # Then any Ms_smooth__*.npy
        for ms in vdir.glob("Ms_smooth__*.npy"):
            try:
                arr = np.load(ms, mmap_mode="r")
                best = max(best, int(arr.shape[1]))
            except Exception:
                pass

        # Finally, PNG width as a rough proxy
        for png in vdir.glob("smoothed_mask95__*.png"):
            try:
                img = mpimg.imread(str(png))
                best = max(best, int(img.shape[1]))
            except Exception:
                pass

    return best or None

def frames_to_seconds(frames: int) -> float:
    return frames * HOP / TARGET_SR

def make_sheet(audios: List[Dict], vocoders: List[str], out_path: Path,
               cols_per_sheet: int = 5, fig_dpi: int = 130):
    """
    audios: list of dicts {bona_id, duration_frames, pngs: {voc: Path or None}}
    Layout: rows=len(vocoders)=4, cols=cols_per_sheet (5)
    """
    rows, cols = len(vocoders), cols_per_sheet
    fig_w, fig_h = 2.1 * cols, 1.9 * rows  # tweak as you like
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    # Titles on top row: audio id + seconds
    for c, a in enumerate(audios):
        sec = frames_to_seconds(a["duration_frames"]) if a["duration_frames"] else None
        title = a["bona_id"] + (f"  ({sec:.2f}s)" if sec else "")
        axes[0, c].set_title(title, fontsize=8)

    # Y labels on first column: vocoder names
    for r, voc in enumerate(vocoders):
        axes[r, 0].set_ylabel(voc, rotation=90, fontsize=8)

    # Fill images
    for r, voc in enumerate(vocoders):
        for c, a in enumerate(audios):
            ax = axes[r, c]
            ax.axis("off")
            png = a["pngs"].get(voc)
            if png and png.exists():
                try:
                    img = mpimg.imread(str(png))
                    ax.imshow(img, aspect="auto")
                except Exception:
                    ax.text(0.5, 0.5, f"read err\n{png.name}", ha="center", va="center", fontsize=7)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7, color="red")

    plt.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=fig_dpi)
    plt.close(fig)

def paginate(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i+k], (i // k) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Root with <BONA>/<VOCODER>/smoothed_mask95__*.png and Mb_smooth.npy")
    ap.add_argument("--vocoders", required=True,
                    help="Comma-separated list of exactly 4 vocoders, top-to-bottom order.")
    ap.add_argument("--out-dir", default="/home/opc/difussionXAI/viz_compare_fourstack",
                    help="Output directory for figures.")
    ap.add_argument("--cols", type=int, default=5, help="Audios per sheet (columns).")
    ap.add_argument("--sort", choices=["asc","desc"], default="desc",
                    help="Order audios by duration (asc/desc).")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of audios.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    vocoders = [v.strip() for v in args.vocoders.split(",") if v.strip()]
    if len(vocoders) != 4:
        raise SystemExit("Please provide exactly 4 vocoders via --vocoders v1,v2,v3,v4")

    # Gather audios that have all 4 PNGs
    rows = []
    for bdir in bona_dirs(root):
        pngs = {}
        ok = True
        for voc in vocoders:
            p = find_png_for(bdir, voc)
            pngs[voc] = p
            if p is None:
                ok = False
        if not ok:
            continue
        frames = find_frames_for_duration(bdir)
        rows.append({"bona_id": bdir.name, "duration_frames": frames, "pngs": pngs})

    if not rows:
        raise SystemExit("No audios found that have all 4 vocoders present.")

    # Sort by duration
    rows = [r for r in rows if r["duration_frames"] is not None] or rows
    rows.sort(key=lambda r: (r["duration_frames"] or 0), reverse=(args.sort=="desc"))

    if args.limit > 0:
        rows = rows[:args.limit]

    # Build sheets
    for chunk, idx in paginate(rows, args.cols):
        out_path = out_dir / f"{'+'.join(vocoders)}_stack4_part{idx:02d}.png"
        make_sheet(chunk, vocoders, out_path, cols_per_sheet=args.cols)

    print(f"[ok] wrote sheets to {out_dir}")

if __name__ == "__main__":
    main()