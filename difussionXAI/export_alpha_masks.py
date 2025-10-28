# export_alpha_masks_v2.py
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def to_rgba(mask_bool: np.ndarray) -> Image.Image:
    """
    mask_bool: np.bool or 0/1 array shaped [F, T]
    Returns an RGBA image where mask is red with full alpha, background transparent.
    """
    m = (mask_bool > 0).astype(np.uint8)
    h, w = m.shape  # h=freq, w=time
    R = (m * 255).astype(np.uint8)
    G = np.zeros_like(R)
    B = np.zeros_like(R)
    A = (m * 255).astype(np.uint8)
    rgba = np.stack([R, G, B, A], axis=-1)  # [H, W, 4]
    return Image.fromarray(rgba, mode="RGBA")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir that contains .../mask95_smoothed__<spoof>.npy files")
    ap.add_argument("--suffix", default="mask95_smoothed_alpha", help="Filename stem to save (suffix before __<spoof>.png)")
    ap.add_argument("--flip", dest="flip", action="store_true", help="Flip vertically (freq axis) [default]")
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.set_defaults(flip=True)
    ap.add_argument("--size", type=str, default=None,
                    help="Force output size as WxH (e.g., 676x546). Nearest-neighbor resize.")
    ap.add_argument("--like-png", type=str, default=None,
                    help="Resize each mask to match the pixel size of this PNG (e.g., your paper-style overlay).")
    ap.add_argument("--out-root", type=str, default=None,
                    help="If given, write outputs under this root mirroring the input tree; "
                         "otherwise save next to each .npy.")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else None

    # Optional: parse explicit WxH
    target_size = None
    if args.size:
        try:
            w_str, h_str = args.size.lower().split("x")
            target_size = (int(w_str), int(h_str))  # PIL expects (W, H)
        except Exception:
            raise ValueError("--size must be like 676x546")

    like_size = None
    if args.like_png:
        like_im = Image.open(args.like_png)
        like_size = like_im.size  # (W, H)
        like_im.close()

    for npy_path in root.rglob("mask95_smoothed__*.npy"):
        mask = np.load(npy_path)
        # mask comes as [F, T] (freq, time). Ensure 2D boolean.
        if mask.ndim != 2:
            mask = np.squeeze(mask)
        mask = (mask > 0)

        # 1) flip vertically if requested (so 0 Hz is at the bottom)
        if args.flip:
            mask = np.flipud(mask)

        # to RGBA
        im = to_rgba(mask)

        # 2) optional resize
        resize_to = target_size or like_size
        if resize_to is not None:
            # Use nearest to retain pixelated blobs
            im = im.resize(resize_to, resample=Image.NEAREST)

        # build output name
        # mask95_smoothed__<spoof>.npy  ->  {suffix}__<spoof>.png
        stem = npy_path.stem  # e.g., "mask95_smoothed__hifigan_LA_D_1024892"
        rest = stem.split("__", 1)[1] if "__" in stem else stem
        out_name = f"{args.suffix}__{rest}.png"

        if out_root:
            rel = npy_path.parent.relative_to(root)
            (out_root / rel).mkdir(parents=True, exist_ok=True)
            out_path = out_root / rel / out_name
        else:
            out_path = npy_path.parent / out_name

        im.save(out_path)
        print(f"[ok] {out_path}")

if __name__ == "__main__":
    main()