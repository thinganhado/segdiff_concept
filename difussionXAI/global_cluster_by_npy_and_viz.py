#!/usr/bin/env python3
# global_cluster_dim_sweep.py
#  - Load ALL vocoders' mask95_smoothed__*.npy
#  - Build features, scale, PCA
#  - Sweep PCA dimension to choose D* that best separates clusters
#  - Auto-pick K (silhouette for k-means; BIC for GMM if chosen)
#  - Cluster in D*, save JSONs + contact sheets
#  - Visualize in 3D (or 2D) + parallel-coordinates of cluster centroids

import argparse, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D proj

# Optional feature libs
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

try:
    from sklearn.mixture import GaussianMixture
    HAVE_GMM = True
except Exception:
    HAVE_GMM = False

try:
    import umap  # pip install umap-learn
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

BONA_RX = re.compile(r"(LA_[TDE]_\d{7})")

# ---------- discovery ----------
def iter_all_masks(root: Path):
    for npy in root.rglob("mask95_smoothed__*.npy"):
        vocoder = npy.parent.name
        bona_id = npy.parent.parent.name
        stem = npy.stem.replace("mask95_smoothed__", "", 1)
        png = npy.parent / f"smoothed_mask95__{stem}.png"
        yield {"vocoder": vocoder, "bona_id": bona_id, "spoof_stem": stem,
               "mask_path": str(npy), "png_path": str(png)}

# ---------- image helpers ----------
def simple_resize(img: np.ndarray, out_hw: Tuple[int,int]) -> np.ndarray:
    H, W = out_hw
    if HAVE_RESIZE:
        return sk_resize(img, (H, W), anti_aliasing=True, preserve_range=True)
    h = max(1, img.shape[0] // H)
    w = max(1, img.shape[1] // W)
    img = img[:H*h, :W*w]
    return img.reshape(H, h, W, w).mean(axis=(1,3))

# ---------- features ----------
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

    rowproj = m.mean(axis=1)
    colproj = m.mean(axis=0)
    feats.append(simple_resize(rowproj.reshape(-1,1), (64,1)).ravel())
    feats.append(simple_resize(colproj.reshape(1,-1), (1,64)).ravel())

    nz = float(m.sum())
    v_balance = float(rowproj[:len(rowproj)//2].sum() - rowproj[len(rowproj)//2:].sum())
    h_balance = float(colproj[:len(colproj)//2].sum() - colproj[len(colproj)//2:].sum())
    feats.append(np.array([nz, v_balance, h_balance], dtype=float))

    return np.concatenate(feats, axis=0)

# ---------- auto-K ----------
def auto_k_kmeans(X: np.ndarray, k_min=3, k_max=12, seed=0) -> Tuple[int, float]:
    best_k, best_s = k_min, -1.0
    upper = min(k_max, max(2, X.shape[0] - 1))
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        lbl = km.fit_predict(X)
        if len(set(lbl)) < 2: 
            continue
        try:
            s = silhouette_score(X, lbl)
        except Exception:
            s = -1.0
        if s > best_s:
            best_k, best_s = k, s
    if best_k < 2: best_k = 2
    return best_k, best_s

def auto_k_gmm(X: np.ndarray, k_min=2, k_max=12, seed=0) -> Tuple[int, float]:
    if not HAVE_GMM:
        return auto_k_kmeans(X, seed=seed)
    best_k, best_bic = k_min, np.inf
    for k in range(k_min, min(k_max, X.shape[0]) + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_k, best_bic = k, bic
        except Exception:
            pass
    if best_k < 2: best_k = 2
    return best_k, -best_bic

# ---------- grids ----------
def draw_grid(img_paths: List[Path], out_path: Path, cols=10, rows=2, dpi=130):
    if not img_paths:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(2.0*cols, 2.2*rows))
    axes = axes.flatten()
    for ax in axes: ax.axis("off")
    for i, p in enumerate(img_paths[:rows*cols]):
        ax = axes[i]
        try:
            ax.imshow(mpimg.imread(str(p)), aspect="auto")
            voc = p.parent.name
            bona = p.parent.parent.name
            ax.set_title(f"{voc}:{bona}", fontsize=6)
        except Exception:
            ax.text(0.5,0.5,p.name, ha="center", va="center", fontsize=7)
        ax.axis("off")
    plt.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi); plt.close()

def paginate(lst: List[Path], page_size: int):
    for i in range(0, len(lst), page_size):
        yield lst[i:i+page_size], (i // page_size) + 1

# ---------- dimension sweep ----------
def dimension_sweep(Xs: np.ndarray, max_dims: int, algo: str, seed: int, k_override: int = 0):
    """
    Returns dict with per-d metrics and the chosen D*.
    Strategy: pick smallest D reaching >=95% of max Silhouette
              and <=105% of min DB. Fallback: argmax Silhouette.
    """
    max_dims = min(max_dims, Xs.shape[1])
    results = []
    for d in range(2, max_dims + 1):
        pca_d = PCA(n_components=d, random_state=seed).fit(Xs)
        Zd = pca_d.transform(Xs)
        if k_override > 0:
            K = k_override
        else:
            if algo == "gmm":
                K, _ = auto_k_gmm(Zd, seed=seed)
            else:
                K, _ = auto_k_kmeans(Zd, seed=seed)
        if algo == "gmm" and HAVE_GMM:
            model = GaussianMixture(n_components=K, covariance_type="full", random_state=seed)
            lbl = model.fit_predict(Zd)
        else:
            model = KMeans(n_clusters=K, n_init="auto", random_state=seed)
            lbl = model.fit_predict(Zd)

        unique = len(set(lbl))
        if unique < 2:
            sil, ch, db = -1.0, 0.0, np.inf
        else:
            sil = silhouette_score(Zd, lbl)
            ch  = calinski_harabasz_score(Zd, lbl)
            db  = davies_bouldin_score(Zd, lbl)
        results.append({"d": d, "k": K, "silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)})

    # choose D*
    sils = np.array([r["silhouette"] for r in results])
    dbs  = np.array([r["davies_bouldin"] for r in results])
    sil_max = np.nanmax(sils)
    db_min  = np.nanmin(dbs)
    cand = [r for r in results if (r["silhouette"] >= 0.95 * sil_max) and (r["davies_bouldin"] <= 1.05 * db_min)]
    if cand:
        D_star = min(r["d"] for r in cand)
    else:
        D_star = int(results[int(np.nanargmax(sils))]["d"])
    return {"results": results, "D_star": D_star}

# ---------- 2D/3D embedding for plots ----------
def embed_2d_or_3d(Z, method: str, dims: int, seed: int):
    assert dims in (2,3)
    if method == "pca":
        return Z[:, :dims]
    if method == "tsne":
        return TSNE(n_components=dims, random_state=seed, init="pca", learning_rate="auto").fit_transform(Z)
    if method == "umap" and HAVE_UMAP:
        return umap.UMAP(n_components=dims, random_state=seed, n_neighbors=30, min_dist=0.1).fit_transform(Z)
    # fallback
    return Z[:, :dims]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/opc/difussionXAI/mask_outputs_crop_phonetier")
    ap.add_argument("--out-root", default="/home/opc/difussionXAI/viz_compare_clusters_global")
    ap.add_argument("--algo", choices=["kmeans","gmm"], default="kmeans")
    ap.add_argument("--embed", choices=["pca","tsne","umap"], default="pca")
    ap.add_argument("--embed-dims", type=int, default=3, help="2 or 3 for plotting")
    ap.add_argument("--max-dims", type=int, default=30, help="max PCA dims to consider in sweep")
    ap.add_argument("--per-grid", type=int, default=20)
    ap.add_argument("--no-sheets", action="store_true", help="Skip creating contact-sheet PNGs")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=0, help="override K; 0=auto")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_dir = out_root / "global_dim_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load items + features
    items, feats = [], []
    for it in tqdm(list(iter_all_masks(root)), desc="feature"):
        try:
            m = np.load(it["mask_path"])
            if m.ndim == 3: m = m.squeeze()
            fv = features_from_mask(m)
            feats.append(fv); items.append(it)
        except Exception as e:
            print(f"[skip] {it['mask_path']}: {e}")
    if not items:
        print("[warn] no items found"); return

    X = np.vstack(feats)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(100, Xs.shape[1]), random_state=args.seed)
    Z = pca.fit_transform(Xs)

    # 2) Dimension sweep -> choose D*
    sweep = dimension_sweep(Xs, max_dims=args.max_dims, algo=args.algo, seed=args.seed, k_override=args.k)
    with open(out_dir / "dim_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)
    D_star = int(sweep["D_star"])
    print(f"[info] Chosen D* = {D_star}")

    # plot sweep curves
    ds   = [r["d"] for r in sweep["results"]]
    sils = [r["silhouette"] for r in sweep["results"]]
    chs  = [r["calinski_harabasz"] for r in sweep["results"]]
    dbs  = [r["davies_bouldin"] for r in sweep["results"]]
    fig, ax = plt.subplots(3,1, figsize=(6,7), sharex=True)
    ax[0].plot(ds, sils); ax[0].set_ylabel("Silhouette (↑)")
    ax[1].plot(ds, chs);  ax[1].set_ylabel("Calinski-Harabasz (↑)")
    ax[2].plot(ds, dbs);  ax[2].set_ylabel("Davies-Bouldin (↓)"); ax[2].set_xlabel("PCA dims")
    for a in ax: a.axvline(D_star, linestyle="--")
    fig.tight_layout(); fig.savefig(out_dir / "dim_sweep.png", dpi=130); plt.close(fig)

    # 3) Cluster in D*
    Zd = Z[:, :D_star]
    if args.k > 0:
        K = args.k
    else:
        if args.algo == "gmm":
            K, _ = auto_k_gmm(Zd, seed=args.seed)
        else:
            K, _ = auto_k_kmeans(Zd, seed=args.seed)
    if args.algo == "gmm" and HAVE_GMM:
        model = GaussianMixture(n_components=K, covariance_type="full", random_state=args.seed)
        labels = model.fit_predict(Zd)
    else:
        model = KMeans(n_clusters=K, n_init="auto", random_state=args.seed)
        labels = model.fit_predict(Zd)

    # 4) JSON outputs (global)
    clusters: Dict[int, List[Dict]] = {c: [] for c in range(K)}
    for it, c in zip(items, labels):
        clusters[int(c)].append({
            "vocoder": it["vocoder"], "bona_id": it["bona_id"], "spoof_stem": it["spoof_stem"],
            "mask_path": it["mask_path"], "png_path": it["png_path"]
        })
    audio_json = {str(c): [{"vocoder": d["vocoder"], "bona_id": d["bona_id"], "spoof_stem": d["spoof_stem"]}
                           for d in clusters[c]] for c in range(K)}
    with open(out_dir / "global_clusters_audio.json", "w") as f:
        json.dump(audio_json, f, indent=2)
    with open(out_dir / "global_clusters_full.json", "w") as f:
        json.dump({"k": K, "D_star": D_star, "algo": args.algo,
                   "clusters": {str(c): clusters[c] for c in range(K)}}, f, indent=2)
    print(f"[ok] wrote global cluster JSONs (K={K}, D*={D_star})")

    # 5) 3D / 2D embedding scatter (for visualization only)
    dims = 3 if args.embed_dims >= 3 else 2
    Zplot = embed_2d_or_3d(Zd, args.embed, dims, args.seed)
    if dims == 3:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        for c in range(K):
            pts = Zplot[labels==c]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=6, label=f"C{c}")
        ax.set_title(f"Global {args.embed.upper()} (3D) by cluster (K={K}, D*={D_star})")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"{args.embed}_3d_by_cluster.png", dpi=130); plt.close(fig)
    else:
        plt.figure(figsize=(6,5))
        for c in range(K):
            pts = Zplot[labels==c]
            plt.scatter(pts[:,0], pts[:,1], s=8, label=f"C{c}")
        plt.title(f"Global {args.embed.upper()} (2D) by cluster (K={K}, D*={D_star})")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out_dir / f"{args.embed}_2d_by_cluster.png", dpi=130); plt.close()

    # 6) Parallel-coordinates of cluster centroids over first min(10, D*) PCs
    d_show = min(10, D_star)
    centroids = np.zeros((K, d_show))
    for c in range(K):
        centroids[c] = Zd[labels==c, :d_show].mean(axis=0)
    # z-normalize per-dim for comparability
    mu = centroids.mean(axis=0); sd = centroids.std(axis=0) + 1e-9
    Cn = (centroids - mu) / sd
    plt.figure(figsize=(max(6, d_show*0.8), 4))
    for c in range(K):
        plt.plot(range(1, d_show+1), Cn[c], marker="o", label=f"C{c}")
    plt.xticks(range(1, d_show+1), [f"PC{i}" for i in range(1, d_show+1)])
    plt.title("Cluster centroid profiles (parallel coordinates)")
    plt.legend(fontsize=8, ncol=min(4,K))
    plt.tight_layout(); plt.savefig(out_dir / "parallel_coords_centroids.png", dpi=130); plt.close()

    # 7) Contact sheets per cluster (optional)
    if not args.no-sheets:
        rows = max(1, args.per_grid // args.cols)
        page_size = rows * args.cols
        for c in range(K):
            paths = [Path(d["png_path"]) for d in clusters[c] if Path(d["png_path"]).exists()]
            for chunk, page in paginate(paths, page_size):
                out_img = out_dir / f"cluster_c{c:02d}_part{page:02d}.png"
                draw_grid(chunk, out_img, cols=args.cols, rows=rows)

if __name__ == "__main__":
    main()