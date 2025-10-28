#!/usr/bin/env python3
# viz_concept_exemplars.py
# Visualize top-N concept exemplars produced by reference_exemplars_from_suggested.py

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

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

# ---------- display helpers ----------
def paper_gray_from_db(db_img, lo_pct=5.0, hi_pct=95.0, out_lo=0.62, out_hi=0.96, gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

RED_MASK_CMAP  = ListedColormap([[0, 0, 0, 0], [0.80, 0.00, 0.00, 1.0]])
BLUE_MASK_CMAP = ListedColormap([[0, 0, 0, 0], [0.00, 0.40, 0.95, 1.0]])

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

# ---------- geometry ----------
def hz_per_bin(n_freq_bins: int, sr: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def rows_for_fmax(arr_2d: np.ndarray, sr: int, fmax_hz: float) -> int:
    hz_bin = hz_per_bin(arr_2d.shape[0], sr)
    return max(1, int(round(fmax_hz / hz_bin)))

def bbox_idx_to_sechz(bbox_idx: List[int], sr: int, hop: int, H: int) -> Tuple[float,float,float,float]:
    t0,t1,f0,f1 = [int(x) for x in bbox_idx]
    hz_bin = hz_per_bin(H, sr)
    return (t0 * hop / sr, t1 * hop / sr, f0 * hz_bin, f1 * hz_bin)

# ---------- stft meta resolver ----------
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

# ---------- drawing ----------
def draw_one_crop(out_path: Path, Mb_db: np.ndarray, add_mask: Optional[np.ndarray], miss_mask: Optional[np.ndarray],
                  sr: int, hop: int, fmax: float, crop: Dict[str, Any], title: str, dpi: int = 130):
    H, W = Mb_db.shape
    max_row = rows_for_fmax(Mb_db, sr, fmax)
    Mb_db = Mb_db[:max_row, :]
    if add_mask is not None:  add_mask  = add_mask[:max_row, :]
    if miss_mask is not None: miss_mask = miss_mask[:max_row, :]

    t_max = W * hop / sr
    extent = [0.0, t_max, 0.0, fmax]
    bg = paper_gray_from_db(Mb_db)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.imshow(bg, extent=extent, aspect="auto", origin="lower",
              cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    if miss_mask is not None:
        ax.imshow(np.ma.masked_where(~miss_mask.astype(bool), miss_mask.astype(float)),
                  extent=extent, aspect="auto", origin="lower",
                  cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    if add_mask is not None:
        ax.imshow(np.ma.masked_where(~add_mask.astype(bool), add_mask.astype(float)),
                  extent=extent, aspect="auto", origin="lower",
                  cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)

    # Single crop rectangle
    if crop.get("bbox_sec_hz"):
        t0, t1, f0, f1 = [float(x) for x in crop["bbox_sec_hz"]]
    else:
        t0, t1, f0, f1 = bbox_idx_to_sechz(crop["bbox_idx"], sr, hop, H)

    # clip to display region
    f0c, f1c = max(0.0, min(f0, fmax)), max(0.0, min(f1, fmax))
    t0c, t1c = max(0.0, min(t0, t_max)), max(0.0, min(t1, t_max))
    if not (f1c <= 0 or f0c >= fmax or t1c <= 0 or t0c >= t_max):
        rect = Rectangle((t0c, f0c), max(1e-6, t1c - t0c), max(1e-6, f1c - f0c),
                         lw=2.0, ec=(0.07, 0.35, 0.95), fc=(0,0,0,0))
        ax.add_patch(rect)

    # label
    cname = crop.get("concept", "concept")
    conf = float(crop.get("confidence", 0.0))
    lbl = f"{cname} ({conf*100:.1f}%)"
    tx = t0c + 0.01 * (t1c - t0c + 1e-6)
    ty = f0c + 0.05 * (f1c - f0c + 1e-6)
    ax.text(tx, ty, lbl, fontsize=10, color="white",
            bbox=dict(facecolor=(0,0,0,0.45), edgecolor="none", pad=2.5))

    ax.set_xlim(0.0, t_max); ax.set_ylim(0.0, fmax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(out_path)

# ---------- concept JSON discovery ----------
def find_top_file(concept_root: Path, concept: str) -> Optional[Path]:
    # choose largest N if multiple topN exist
    cands = sorted(concept_root.glob(f"{concept}_top*.json"))
    if not cands:
        return None
    # prefer higher N by numeric suffix if present
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
    # bare list?
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    # common keys
    for k in ("items", "topN", "topn", "exemplars", "hits", "crops"):
        v = data.get(k)
        if isinstance(v, list):
            return v
    return []

def main():
    ap = argparse.ArgumentParser(description="Visualize top-N concept exemplars as individual images.")
    ap.add_argument("--concept-json-root", type=str, required=True,
                    help="Folder with <concept>_topN.json and/or <concept>_95plus.json")
    ap.add_argument("--suggested-root", type=str, required=True,
                    help="Root with per-pair metadata: <BONA>/<VOCODER>/{suggested_crops.json,k_summary.json}")
    ap.add_argument("--crops-root", type=str, default="./crops_sweep",
                    help="Root with k{K}/<BONA>/<VOCODER>/crops/crops_manifest.json (for spoof_stem resolution)")
    ap.add_argument("--array-root", type=str, required=True,
                    help="Arrays root: <root>/<BONA>/<VOCODER>/{Mb_*.npy, mask95_*}.npy")
    ap.add_argument("--take-first", type=int, default=10, help="Render only the first N items per concept.")
    ap.add_argument("--fmax", type=float, default=6000.0, help="Frequency cap for display in Hz.")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--debug", action="store_true", default=False)
    args = ap.parse_args()

    concept_root = Path(args.concept_json_root)
    suggested_root = Path(args.suggested_root)
    crops_root = Path(args.crops_root)
    array_root = Path(args.array_root)

    for concept in CONCEPTS:
        # pick source file
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
                # ensure highest-confidence first
                items = sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                src_label = fallback.name

        if not items:
            print(f"[warn] no items for {concept} (looked for {concept}_top*.json / {concept}_95plus.json)")
            continue

        if args.debug:
            print(f"[debug] {concept}: using {src_label} with {len(items)} items")

        # slice to requested count
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

            # Resolve the exact spoof_stem from the per-K manifest, if available
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

            # Robust Mb lookup (prefer Mb_db/Mb_smooth with stem, then generic)
            Mb_path = safe_find_first_with_stem(
                pair_dir, stem,
                stemmed=["Mb_db__{stem}.npy", "Mb_smooth__{stem}.npy"],
                generic=["Mb_db.npy", "Mb_smooth.npy", "Mb.npy"]
            )
            if not Mb_path:
                print(f"[warn] {concept}: No Mb_* found under {pair_dir} (stem={stem}), skipping {bona}/{voc}")
                continue
            Mb_db = np.load(Mb_path)

            # Prefer stemmed add/miss; fall back to any; if neither present, try combined mask
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

            draw_one_crop(out_path, Mb_db, add_mask, miss_mask, sr, hop, args.fmax, crop, title, dpi=args.dpi)

if __name__ == "__main__":
    main()