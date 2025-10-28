# compare_shaps_to_masks.py
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# paths
GT_ROOT = Path("/home/opc/difussionXAI/out_all")             # make_masks output
DS_ROOT = Path("/home/opc/work/SSL_Anti-spoofing/diffusion_xai_baselines/deepshap_outputs_vocv4")                   # DeepSHAP output
GS_ROOT = Path("/home/opc/work/SSL_Anti-spoofing/diffusion_xai_baselines/gradshap_outputs_vocv4")                   # GradientSHAP output
AS_ROOT = Path("/home/opc/work/SSL_Anti-spoofing/diffusion_xai_baselines/attnlrp_outputs_vocv4")                    # AttnLRP output
OUT_DIR = Path("/home/opc/work/SSL_Anti-spoofing/diffusion_xai_baselines/xai_vs_gt_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_npy(p: Path):
    a = np.load(p)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    return a

def iou_f1_prec_recall(gt: np.ndarray, pr: np.ndarray):
    gt = gt.astype(np.uint8)
    pr = pr.astype(np.uint8)
    tp = np.sum((gt == 1) & (pr == 1))
    fp = np.sum((gt == 0) & (pr == 1))
    fn = np.sum((gt == 1) & (pr == 0))
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(iou), float(f1), float(prec), float(rec)

def crop_to_min(*arrays):
    F = min(a.shape[0] for a in arrays if a is not None)
    T = min(a.shape[1] for a in arrays if a is not None)
    out = []
    for a in arrays:
        out.append(a[:F, :T] if a is not None else None)
    return out

def _stretch01(x):
    x = x - np.nanmin(x)
    vmax = np.percentile(x, 98) if x.size else 1.0
    if vmax > 0:
        x = np.clip(x / vmax, 0, 1)
    return x

def save_heptaptych(bg_b: np.ndarray, gt_mask: np.ndarray,
                    ds_heat: np.ndarray, ds_mask: np.ndarray,
                    gs_heat: np.ndarray, gs_mask: np.ndarray,
                    as_heat: np.ndarray, as_mask: np.ndarray,
                    out_png: Path, title: str):
    plt.figure(figsize=(24, 3.8))

    # 1, ground truth overlay on bona fide smoothed spec
    ax1 = plt.subplot(1, 7, 1)
    bg_disp = _stretch01(bg_b.copy())
    ax1.imshow(bg_disp, origin="lower", aspect="auto", cmap="gray", interpolation="nearest")
    ax1.imshow(np.ma.masked_where(gt_mask == 0, gt_mask.astype(float)),
               origin="lower", aspect="auto", cmap="Reds", alpha=0.9, vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("Ground truth mask, 95 percent")
    ax1.set_xlabel("time frames"); ax1.set_ylabel("frequency bins")

    # 2, DeepSHAP heatmap
    ax2 = plt.subplot(1, 7, 2)
    if ds_heat is not None:
        disp = np.tanh(ds_heat / (np.std(ds_heat) + 1e-8))
        ax2.imshow(disp, origin="lower", aspect="auto", interpolation="nearest")
    ax2.set_title("DeepSHAP heatmap")
    ax2.set_xlabel("time frames"); ax2.set_ylabel("")

    # 3, DeepSHAP mask
    ax3 = plt.subplot(1, 7, 3)
    if ds_mask is not None:
        ax3.imshow(ds_mask.astype(float), origin="lower", aspect="auto", cmap="Reds",
                   interpolation="nearest", vmin=0, vmax=1)
    ax3.set_title("DeepSHAP mask, 95 percent")
    ax3.set_xlabel("time frames"); ax3.set_ylabel("")

    # 4, GradientSHAP heatmap
    ax4 = plt.subplot(1, 7, 4)
    if gs_heat is not None:
        disp_g = np.tanh(gs_heat / (np.std(gs_heat) + 1e-8))
        ax4.imshow(disp_g, origin="lower", aspect="auto", interpolation="nearest")
    ax4.set_title("GradientSHAP heatmap")
    ax4.set_xlabel("time frames"); ax4.set_ylabel("")

    # 5, GradientSHAP mask
    ax5 = plt.subplot(1, 7, 5)
    if gs_mask is not None:
        ax5.imshow(gs_mask.astype(float), origin="lower", aspect="auto", cmap="Reds",
                   interpolation="nearest", vmin=0, vmax=1)
    ax5.set_title("GradientSHAP mask, 95 percent")
    ax5.set_xlabel("time frames"); ax5.set_ylabel("")

    # 6, AttnLRP heatmap
    ax6 = plt.subplot(1, 7, 6)
    if as_heat is not None:
        disp_a = np.tanh(as_heat / (np.std(as_heat) + 1e-8))
        ax6.imshow(disp_a, origin="lower", aspect="auto", interpolation="nearest")
    ax6.set_title("AttnLRP heatmap")
    ax6.set_xlabel("time frames"); ax6.set_ylabel("")

    # 7, AttnLRP mask
    ax7 = plt.subplot(1, 7, 7)
    if as_mask is not None:
        ax7.imshow(as_mask.astype(float), origin="lower", aspect="auto", cmap="Reds",
                   interpolation="nearest", vmin=0, vmax=1)
    ax7.set_title("AttnLRP mask, 95 percent")
    ax7.set_xlabel("time frames"); ax7.set_ylabel("")

    plt.suptitle(title, y=1.03, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

def main():
    rows = []
    missing_any = 0

    for bona_dir in sorted(GT_ROOT.iterdir()):
        if not bona_dir.is_dir():
            continue
        bona_id = bona_dir.name
        for voc_dir in sorted(bona_dir.iterdir()):
            if not voc_dir.is_dir():
                continue
            voc = voc_dir.name

            for p_mask in sorted(voc_dir.glob("mask95_smoothed__*.npy")):
                spoof_stem = p_mask.stem.replace("mask95_smoothed__", "")

                # DeepSHAP files
                ds_dir = DS_ROOT / bona_id / voc
                p_ds_heat = ds_dir / f"attr_deepshap__{spoof_stem}.npy"
                p_ds_mask = ds_dir / f"attr_deepshap_mask95__{spoof_stem}.npy"

                # GradientSHAP files
                gs_dir = GS_ROOT / bona_id / voc
                p_gs_heat = gs_dir / f"attr_gradshap__{spoof_stem}.npy"
                p_gs_mask = gs_dir / f"attr_gradshap_mask95__{spoof_stem}.npy"

                # AttnLRP files
                as_dir = AS_ROOT / bona_id / voc
                p_as_heat = as_dir / f"attr_attnlrp__{spoof_stem}.npy"
                p_as_mask = as_dir / f"attr_attnlrp_mask95__{spoof_stem}.npy"
                # If you saved only heatmaps for AttnLRP, we can derive mask95 here
                have_as_heat = p_as_heat.exists()
                have_as_mask = p_as_mask.exists()

                have_any = any([
                    p_ds_heat.exists() and p_ds_mask.exists(),
                    p_gs_heat.exists() and p_gs_mask.exists(),
                    have_as_heat and (have_as_mask or True)  # allow mask creation below
                ])
                if not have_any:
                    missing_any += 1
                    continue

                # ground truth and background
                gt_mask = load_npy(p_mask)
                Mb = load_npy(voc_dir / "Mb_smooth.npy")

                # DeepSHAP
                ds_heat = load_npy(p_ds_heat) if p_ds_heat.exists() else None
                ds_mask = load_npy(p_ds_mask) if p_ds_mask.exists() else None

                # GradientSHAP
                gs_heat = load_npy(p_gs_heat) if p_gs_heat.exists() else None
                gs_mask = load_npy(p_gs_mask) if p_gs_mask.exists() else None

                # AttnLRP
                as_heat = load_npy(p_as_heat) if have_as_heat else None
                if have_as_mask:
                    as_mask = load_npy(p_as_mask)
                else:
                    as_mask = None
                    if as_heat is not None:
                        flat = np.abs(as_heat).ravel()
                        tau = np.quantile(flat, 0.95) if flat.size else 0.0
                        as_mask = (np.abs(as_heat) >= tau).astype(np.uint8)

                # align shapes
                Mb_c, gt_c, ds_heat_c, ds_mask_c, gs_heat_c, gs_mask_c, as_heat_c, as_mask_c = crop_to_min(
                    Mb, gt_mask, ds_heat, ds_mask, gs_heat, gs_mask, as_heat, as_mask
                )

                # metrics
                ds_iou = ds_f1 = ds_prec = ds_rec = np.nan
                gs_iou = gs_f1 = gs_prec = gs_rec = np.nan
                as_iou = as_f1 = as_prec = as_rec = np.nan
                if ds_mask_c is not None:
                    ds_iou, ds_f1, ds_prec, ds_rec = iou_f1_prec_recall(gt_c, ds_mask_c)
                if gs_mask_c is not None:
                    gs_iou, gs_f1, gs_prec, gs_rec = iou_f1_prec_recall(gt_c, gs_mask_c)
                if as_mask_c is not None:
                    as_iou, as_f1, as_prec, as_rec = iou_f1_prec_recall(gt_c, as_mask_c)

                out_png = OUT_DIR / f"{bona_id}__{voc}__{spoof_stem}.png"
                save_heptaptych(Mb_c, gt_c, ds_heat_c, ds_mask_c, gs_heat_c, gs_mask_c, as_heat_c, as_mask_c,
                                out_png,
                                title=(f"{bona_id}  ({voc})  {spoof_stem}\n"
                                       f"DeepSHAP IoU {ds_iou:.3f} F1 {ds_f1:.3f}   "
                                       f"GradSHAP IoU {gs_iou:.3f} F1 {gs_f1:.3f}   "
                                       f"AttnLRP IoU {as_iou:.3f} F1 {as_f1:.3f}"))

                rows.append({
                    "bona_id": bona_id,
                    "vocoder": voc,
                    "spoof_stem": spoof_stem,

                    "ds_iou": ds_iou, "ds_f1": ds_f1,
                    "ds_precision": ds_prec, "ds_recall": ds_rec,

                    "gs_iou": gs_iou, "gs_f1": gs_f1,
                    "gs_precision": gs_prec, "gs_recall": gs_rec,

                    "as_iou": as_iou, "as_f1": as_f1,
                    "as_precision": as_prec, "as_recall": as_rec,

                    "figure_png": str(out_png),
                    "gt_mask": str(p_mask),

                    "ds_heat": str(p_ds_heat) if p_ds_heat.exists() else "",
                    "ds_mask": str(p_ds_mask) if p_ds_mask.exists() else "",

                    "gs_heat": str(p_gs_heat) if p_gs_heat.exists() else "",
                    "gs_mask": str(p_gs_mask) if p_gs_mask.exists() else "",

                    "as_heat": str(p_as_heat) if have_as_heat else "",
                    "as_mask": str(p_as_mask) if p_as_mask.exists() else "",
                })

    if rows:
        out_csv = OUT_DIR / "metrics_all_xais_vs_gt.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {len(rows)} comparisons to {out_csv}")
        if missing_any:
            print(f"Skipped {missing_any} items that had no matching XAI files")
    else:
        print("No matches found, check GT_ROOT, DS_ROOT, GS_ROOT, AS_ROOT")

if __name__ == "__main__":
    main()