#!/usr/bin/env python3
# compare_vocoder_maps.py
# Build 10x2 grids of difference-map PNGs for one vocoder.

import argparse, math
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_maps(root: Path, vocoder: str) -> List[Path]:
    # pattern: .../<BONA>/<VOCODER>/smoothed_mask95__{vocoder}_<BONA>.png
    pattern = f"**/{vocoder}/smoothed_mask95__{vocoder}_*.png"
    return sorted(root.glob(pattern))

def label_from_path(p: Path) -> str:
    # expects .../<BONA>/<VOCODER>/<file.png>
    try:
        bona = p.parent.parent.name  # LA_T_..., LA_D_..., LA_E_...
        return bona
    except Exception:
        return p.stem

def draw_grid(img_paths: List[Path], out_path: Path, cols: int = 10, rows: int = 2, dpi: int = 130):
    n = len(img_paths)
    if n == 0:
        return
    fig_w = 2.0 * cols
    fig_h = 2.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, p in enumerate(img_paths[:rows*cols]):
        ax = axes[i]
        try:
            img = mpimg.imread(str(p))
            ax.imshow(img, aspect="auto")
            ax.set_title(label_from_path(p), fontsize=7)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed:\n{p.name}", ha="center", va="center", fontsize=7)
            ax.axis("off")

    plt.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)

def paginate(lst: List[Path], page_size: int):
    for i in range(0, len(lst), page_size):
        yield lst[i:i+page_size], (i // page_size) + 1, math.ceil(len(lst) / page_size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Root dir with <BONA>/<VOCODER>/smoothed_mask95__*.png")
    ap.add_argument("--vocoder", required=True, help="e.g., hifigan, waveglow, etc.")
    ap.add_argument("--out", default="./vocoder_compare.png",
                    help="Output path (suffix _partXX.png added if paginated)")
    ap.add_argument("--per-grid", type=int, default=20, help="Tiles per image (default 20)")
    ap.add_argument("--cols", type=int, default=10, help="Columns per image (default 10)")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    root = Path(args.root)
    vocoder = args.vocoder.strip()
    out = Path(args.out)

    paths = find_maps(root, vocoder)
    if not paths:
        print(f"[warn] No maps found for vocoder '{vocoder}' under {root}")
        return

    rows = max(1, args.per_grid // args.cols)
    page_size = rows * args.cols

    if len(paths) <= page_size:
        draw_grid(paths, out, cols=args.cols, rows=rows, dpi=args.dpi)
        print(f"[ok] Wrote {out} with {min(len(paths), page_size)} tiles for {vocoder}")
    else:
        for chunk, page_idx, total_pages in paginate(paths, page_size):
            out_i = out.with_name(f"{out.stem}_part{page_idx:02d}{out.suffix}")
            draw_grid(chunk, out_i, cols=args.cols, rows=rows, dpi=args.dpi)
            print(f"[ok] Wrote {out_i} (page {page_idx}/{total_pages}, {len(chunk)} tiles)")

if __name__ == "__main__":
    main()