#!/usr/bin/env python3
# cluster_by_npy_and_viz.py  (with phone-class color bars)
# 1) cluster by mask95_smoothed__*.npy
# 2) save cluster -> audio names JSON
# 3) visualize smoothed_mask95__*.png per-cluster in grids
# 4) NEW: draw a time-aligned color bar (Vowel / Approximant / Consonant / Silence)
#         using TextGrids at /home/opc/aligned_textgrids/<BONA>/<BONA>.TextGrid

import argparse, re, json, csv, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

# --- optional feature libs (falls back gracefully) ---
try:
    from skimage.feature import hog
    HAVE_HOG = True
except Exception:
    HAVE_HOG = False

try:
    from skimage.transform import resize as sk_resize
    HAVE_RESIZE = True
except Exception:
    HAVE_RESIZE = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BONA_RX = re.compile(r"(LA_[TDE]_\d{7})")

# ------------------ TextGrid parsing ------------------

PHONE_TIER_RX_DEFAULT = re.compile(r"(phone|phoneme|phones|phn|segment)", re.I)

def _read_textgrid(path: Path) -> Dict[str, List[Tuple[float,float,str]]]:
    """
    Minimal, robust TextGrid reader (IntervalTier only).
    Returns {tier_name: [(xmin, xmax, text), ...], ...}
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    tiers: Dict[str, List[Tuple[float,float,str]]] = {}
    # Split by items
    for item_block in re.split(r"\n\s*item \[\d+\]:\s*\n", txt):
        if "IntervalTier" not in item_block:
            continue
        # name
        mname = re.search(r'name\s*=\s*"(.*?)"', item_block)
        tname = (mname.group(1) if mname else "UNKNOWN").strip()
        # intervals
        entries: List[Tuple[float,float,str]] = []
        for m in re.finditer(
            r"intervals\s*\[\d+\]:\s*?\n\s*xmin\s*=\s*([0-9.+\-eE]+)\s*?\n\s*xmax\s*=\s*([0-9.+\-eE]+)\s*?\n\s*text\s*=\s*\"(.*?)\"",
            item_block, flags=re.S):
            xmin = float(m.group(1)); xmax = float(m.group(2)); lab = m.group(3)
            entries.append((xmin, xmax, lab))
        if entries:
            tiers[tname] = entries
    return tiers

_ARPA_VOWELS = {
    "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX","AXR","IX","UX"
}
_ARPA_APPROX = {"L","R","W","Y","J"}  # 'ER' treated as vowel in ARPAbet

_IPA_APPROX = {"w","j","l","r","ɹ","ʎ","ʝ","ɰ","ʋ"}
_IPA_VOWEL_CHARS = set("aeiouɑɐɒæʌəɚɝɜɛɘɞøœyʏɯɪʊœ̈" "ɨ")

_SIL_LABS = {"sil","sp","spn","pau","nsn","h#", "silence", ""}

def _norm_phone_label(lab: str) -> str:
    s = lab.strip().strip('"').strip("'")
    return s

def _phone_class(lab: str) -> str:
    """
    Map a phone label to one of: 'V' (Vowel), 'APP' (Approximant), 'C' (Consonant), 'S' (Silence/Other)
    Tries ARPAbet first, then IPA-ish heuristics.
    """
    s = _norm_phone_label(lab)
    low = s.lower()
    if low in _SIL_LABS:
        return "S"
    # ARPAbet?
    arpa = re.sub(r"\d", "", s.upper())  # drop stress digits
    if arpa in _ARPA_VOWELS:
        return "V"
    if arpa in _ARPA_APPROX:
        return "APP"
    # IPA-ish
    if any(ch in _IPA_VOWEL_CHARS for ch in low):
        return "V"
    if low in _IPA_APPROX:
        return "APP"
    # else consonant
    return "C"

def _bona_from_any(text: str) -> Optional[str]:
    m = BONA_RX.search(text)
    return m.group(1) if m else None

def _textgrid_for_bona(tg_root: Path, bona_id: str) -> Path:
    return tg_root / bona_id / f"{bona_id}.TextGrid"

def _bar_colors():
    # Vowel, Approximant, Consonant, Silence
    return {
        "V": np.array([0.13, 0.69, 0.30], dtype=float),   # green
        "APP": np.array([1.00, 0.55, 0.00], dtype=float), # orange
        "C": np.array([0.20, 0.40, 1.00], dtype=float),   # blue
        "S": np.array([0.60, 0.60, 0.60], dtype=float),   # grey
    }

def build_phone_bar_from_textgrid(tg_path: Path, width_px: int, bar_h: int,
                                  phone_tier_rx: re.Pattern) -> Optional[np.ndarray]:
    if not tg_path.exists():
        return None
    tiers = _read_textgrid(tg_path)
    if not tiers:
        return None
    # choose tier
    tier_name = None
    for name in tiers.keys():
        if phone_tier_rx.search(name):
            tier_name = name; break
    if tier_name is None:
        # fallback: first interval tier we saw
        tier_name = list(tiers.keys())[0]
    intervals = tiers[tier_name]
    if not intervals:
        return None

    # figure out global duration from intervals
    tmin = min(x[0] for x in intervals)
    tmax = max(x[1] for x in intervals)
    dur = max(1e-9, tmax - tmin)

    # make bar
    colors = _bar_colors()
    bar = np.ones((bar_h, width_px, 3), dtype=float)  # start white
    # fill by intervals
    for xmin, xmax, lab in intervals:
        cls = _phone_class(lab)
        c = colors.get(cls, colors["C"])
        x0 = int(round((max(tmin, xmin) - tmin) / dur * width_px))
        x1 = int(round((min(tmax, xmax) - tmin) / dur * width_px))
        x0 = max(0, min(width_px, x0)); x1 = max(0, min(width_px, x1))
        if x1 > x0:
            bar[:, x0:x1, :] = c
    # top border to separate from image
    bar[0:1, :, :] = 0.0
    return bar

# ---------- discovery ----------
def list_vocoders(root: Path) -> List[str]:
    vocs = set()
    # Looking for .../<BONA>/<VOCODER>/mask95_smoothed__*.npy
    for p in root.glob("**/*"):
        if p.is_dir() and p.parent.is_dir():
            if any(p.glob("mask95_smoothed__*.npy")):
                vocs.add(p.name)
    return sorted(vocs)

def find_npys(root: Path, vocoder: str) -> List[Path]:
    return sorted(root.glob(f"**/{vocoder}/mask95_smoothed__*.npy"))

def spoof_stem_from_mask(npy_path: Path) -> str:
    # mask95_smoothed__{spoof_stem}.npy  -> {spoof_stem}
    return npy_path.stem.replace("mask95_smoothed__", "", 1)

def png_for_mask(npy_path: Path) -> Path:
    # same folder, name switch
    stem = spoof_stem_from_mask(npy_path)
    return npy_path.parent / f"smoothed_mask95__{stem}.png"

def bona_from_any(text: str) -> Optional[str]:
    m = BONA_RX.search(text)
    return m.group(1) if m else None

def bona_from_path(p: Path) -> str:
    # Prefer folder two levels up, else regex from file name
    try:
        return p.parent.parent.name
    except Exception:
        bid = bona_from_any(p.name) or "UNK"
        return bid

# ---------- image helpers ----------
def simple_resize(img: np.ndarray, out_hw: Tuple[int,int]) -> np.ndarray:
    H, W = out_hw
    if HAVE_RESIZE:
        return sk_resize(img, (H, W), anti_aliasing=True, preserve_range=True)
    # Mean-pool fallback
    h = max(1, img.shape[0] // H)
    w = max(1, img.shape[1] // W)
    img = img[:H*h, :W*w]
    return img.reshape(H, h, W, w).mean(axis=(1,3))

def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Accept grayscale or RGBA and return float RGB in [0,1]."""
    a = arr
    if a.ndim == 2:  # gray
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:  # RGBA -> RGB
        a = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4])
    # normalize if needed
    if a.dtype != np.float32 and a.dtype != np.float64:
        a = a.astype(np.float32) / 255.0
    return a

def image_with_phone_bar(png_path: Path, bona_id: str, tg_root: Path,
                         bar_h: int, phone_tier_rx: re.Pattern) -> np.ndarray:
    try:
        img = mpimg.imread(str(png_path))
        img = _ensure_rgb(img)
    except Exception:
        # fallback: white card with filename
        img = np.ones((200, 300, 3), dtype=float)
    H, W, _ = img.shape
    tg_path = _textgrid_for_bona(tg_root, bona_id)
    bar = build_phone_bar_from_textgrid(tg_path, W, bar_h, phone_tier_rx)
    if bar is None:
        return img
    # stack: [image; bar]
    combo = np.vstack([img, bar])
    return combo

# ---------- features from npy mask ----------
def features_from_mask(mask: np.ndarray, target_hw=(128,128)) -> np.ndarray:
    m = (mask > 0.5).astype(float)
    m = simple_resize(m, target_hw)
    feats = []

    if HAVE_HOG:
        fv = hog(m, orientations=8, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)
        feats.append(fv)
    else:
        feats.append(simple_resize(m, (64,64)).ravel())

    # row/col projections (shape signatures)
    rowproj = m.mean(axis=1)
    colproj = m.mean(axis=0)
    feats.append(simple_resize(rowproj.reshape(-1,1), (64,1)).ravel())
    feats.append(simple_resize(colproj.reshape(1,-1), (1,64)).ravel())

    return np.concatenate(feats, axis=0)

# ---------- clustering helpers ----------
def choose_k_auto(X: np.ndarray, k_min=3, k_max=12, seed=0) -> int:
    n = X.shape[0]
    if n <= 2:  # edge cases
        return n
    best_k, best_s = k_min, -1.0
    for k in range(k_min, min(k_max, n) + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
            lbl = km.fit_predict(X)
            s = silhouette_score(X, lbl) if k > 1 and len(set(lbl)) > 1 else -1.0
        except Exception:
            s = -1.0
        if s > best_s:
            best_k, best_s = k, s
    return best_k

# ---------- grids ----------
def draw_grid(img_paths: List[Path], out_path: Path, cols: int, rows: int, dpi: int,
              tg_root: Path, bar_h: int, phone_tier_rx: re.Pattern):
    if not img_paths:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(2.0*cols, 2.25*rows))
    axes = axes.flatten()
    for ax in axes: ax.axis("off")

    for i, p in enumerate(img_paths[:rows*cols]):
        ax = axes[i]
        # determine bona id and overlay bar
        bona = bona_from_path(p)
        try:
            combo = image_with_phone_bar(p, bona, tg_root, bar_h, phone_tier_rx)
            ax.imshow(combo, aspect="auto")
            ax.set_title(bona, fontsize=7)
        except Exception:
            ax.text(0.5,0.5,p.name, ha="center", va="center", fontsize=7)
        ax.axis("off")

    # legend once per page
    handles = [
        Patch(facecolor=_bar_colors()["V"], label="Vowel"),
        Patch(facecolor=_bar_colors()["APP"], label="Approximant"),
        Patch(facecolor=_bar_colors()["C"], label="Consonant"),
        Patch(facecolor=_bar_colors()["S"], label="Silence/Other"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(pad=0.6, rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi)
    plt.close()

def paginate(lst: List[Path], page_size: int):
    for i in range(0, len(lst), page_size):
        yield lst[i:i+page_size], (i // page_size) + 1

# ---------- main routine per vocoder ----------
def run_vocoder(root: Path, vocoder: str, out_root: Path,
                k: int, seed: int, per_grid: int, cols: int,
                tg_root: Path, bar_h: int, phone_tier_rx: re.Pattern):

    npys = find_npys(root, vocoder)
    if not npys:
        print(f"[warn] No npy masks for vocoder {vocoder}")
        return

    feats, keep_npys = [], []
    for p in tqdm(npys, desc=f"{vocoder} feature"):
        try:
            m = np.load(p)
            if m.ndim == 3: m = m.squeeze()
            fv = features_from_mask(m)
            feats.append(fv); keep_npys.append(p)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    if len(keep_npys) == 0:
        print(f"[warn] nothing to cluster for {vocoder}")
        return

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1]), random_state=seed)
    Z  = pca.fit_transform(X)

    if k <= 0:
        k = choose_k_auto(Z, k_min=3, k_max=12, seed=seed)
        k = max(1, min(k, len(keep_npys)))  # final safety
        print(f"[info] chosen K={k} for {vocoder}")

    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(Z)

    voc_out = out_root / vocoder / f"clusters_k{k}"
    voc_out.mkdir(parents=True, exist_ok=True)

    # ---- build JSON payloads ----
    clusters: Dict[int, List[Dict]] = {c: [] for c in range(k)}
    for npy_path, c in zip(keep_npys, labels):
        stem   = spoof_stem_from_mask(npy_path)
        png    = png_for_mask(npy_path)
        bona   = bona_from_any(stem) or bona_from_path(npy_path)
        clusters[int(c)].append({
            "bona_id": bona,
            "spoof_stem": stem,
            "vocoder": vocoder,
            "mask_path": str(npy_path),
            "png_path": str(png)
        })

    # Minimal (just audio names)
    audio_json = {str(c): [item["bona_id"] for item in clusters[c]] for c in sorted(clusters)}
    with open(voc_out / "clusters_audio.json", "w") as f:
        json.dump(audio_json, f, indent=2)

    # Full metadata
    with open(voc_out / "clusters_full.json", "w") as f:
        json.dump({
            "vocoder": vocoder,
            "k": k,
            "clusters": {str(c): clusters[c] for c in sorted(clusters)}
        }, f, indent=2)

    print(f"[ok] wrote {voc_out/'clusters_audio.json'} and clusters_full.json")

    # ---- contact sheets (PNG side-by-side) ----
    rows = max(1, per_grid // cols)
    page_size = rows * cols
    for c in range(k):
        paths = [Path(it["png_path"]) for it in clusters[c] if Path(it["png_path"]).exists()]
        if not paths:
            # fallback: draw text card if no PNGs present
            note = voc_out / f"cluster_c{c:02d}_EMPTY.txt"
            note.write_text("No corresponding PNGs found for this cluster.\n")
            continue
        for chunk, page in paginate(paths, page_size):
            out_img = voc_out / f"cluster_c{c:02d}_part{page:02d}.png"
            draw_grid(chunk, out_img, cols=cols, rows=rows, dpi=130,
                      tg_root=tg_root, bar_h=bar_h, phone_tier_rx=phone_tier_rx)

    # Optional: 2D PCA scatter
    if Z.shape[1] >= 2:
        plt.figure(figsize=(5,4))
        for c in range(k):
            pts = Z[labels==c]
            plt.scatter(pts[:,0], pts[:,1], s=10, label=f"C{c}")
        plt.legend(fontsize=8)
        plt.title(f"{vocoder} PCA (first 2 comps)")
        plt.tight_layout(); plt.savefig(voc_out / "pca_scatter.png", dpi=130); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Root with <BONA>/<VOCODER>/mask95_smoothed__*.npy")
    ap.add_argument("--out-root", default="/home/opc/difussionXAI/viz_compare_clusters_npy",
                    help="Output root")
    ap.add_argument("--vocoder", default=None, help="Single vocoder; if omitted, run all")
    ap.add_argument("--k", type=int, default=0, help="Clusters K (0 = auto by silhouette)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-grid", type=int, default=20)
    ap.add_argument("--cols", type=int, default=10)

    # NEW: TextGrid + overlay options
    ap.add_argument("--tg-root", default="/home/opc/aligned_textgrids",
                    help="Aligned TextGrids root: <tg-root>/<BONA>/<BONA>.TextGrid")
    ap.add_argument("--bar-h", type=int, default=8, help="Phone bar height in pixels")
    ap.add_argument("--phone-tier-regex", default="(phone|phoneme|phones|phn|segment)",
                    help="Regex to choose the phone tier from TextGrid names (case-insensitive)")

    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    tg_root = Path(args.tg_root)
    phone_tier_rx = re.compile(args.phone_tier_regex, re.I)

    if args.vocoder:
        run_vocoder(root, args.vocoder, out_root, args.k, args.seed, args.per_grid, args.cols,
                    tg_root=tg_root, bar_h=args.bar_h, phone_tier_rx=phone_tier_rx)
    else:
        vocs = list_vocoders(root)
        print(f"[info] found {len(vocs)} vocoders: {', '.join(vocs)}")
        for v in vocs:
            run_vocoder(root, v, out_root, args.k, args.seed, args.per_grid, args.cols,
                        tg_root=tg_root, bar_h=args.bar_h, phone_tier_rx=phone_tier_rx)

if __name__ == "__main__":
    main()