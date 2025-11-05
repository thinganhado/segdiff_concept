# improved_diffusion/sampling_util.py
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import dist_util
from .metrics import FBound_metric, WCov_metric
from datasets.voc_v4 import VocV4Dataset
from .utils import set_random_seed_for_iterations

# ---- Visualization palette (binary): 0 -> black (background), 1 -> red (mask)
binary_palette = [0, 0, 0, 255, 0, 0] + [0, 0, 0] * (256 - 2)


def _to_uint8(arr01: np.ndarray) -> np.ndarray:
    """[0,1] -> uint8 [0,255]"""
    return (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_condition_gray(condition_tensor: torch.Tensor, out_path: str):
    """
    condition_tensor: (3,H,W) or (1,H,W) in [-1,1]; save as grayscale PNG.
    We just take channel 0 because VOC-v4 replicates grayscale to 3 channels.
    """
    if condition_tensor.dim() == 3:
        chan0 = condition_tensor[0]  # [-1,1]
    else:
        chan0 = condition_tensor[0, 0]
    cond01 = (chan0.detach().cpu().numpy() + 1.0) / 2.0  # [0,1]
    Image.fromarray(_to_uint8(cond01), mode="L").save(out_path)


def _save_mask_palette(mask_tensor01: torch.Tensor, out_path_palette: str, out_path_raw: str):
    """
    mask_tensor01: (1,H,W) in {0.0,1.0} (float) or [0,1] float -> will binarize at 0.5
    Saves paletted preview (black/red) and raw binary PNG (0/1).
    """
    mask = (mask_tensor01.detach().cpu().numpy() > 0.5).astype(np.uint8)[0]  # H×W in {0,1}

    # Paletted preview
    im_p = Image.fromarray(mask, mode="P")
    im_p.putpalette(binary_palette)
    im_p.save(out_path_palette)

    # Raw binary (still a viewable image; pixel values 0 or 1)
    Image.fromarray(mask * 255, mode="L").save(out_path_raw)


def calculate_metrics(x, gt):
    """
    x, gt: tensors shaped (H,W) with values in {0,1}.
    Returns: F1, IoU (Jaccard), WCov, BoundF
    """
    predict = x.detach().cpu().numpy().astype("uint8")
    target = gt.detach().cpu().numpy().astype("uint8")
    return (
        f1_score(target.flatten(), predict.flatten()),
        jaccard_score(target.flatten(), predict.flatten()),
        WCov_metric(predict, target),
        FBound_metric(predict, target),
    )


def sampling_major_vote_func(
    diffusion_model,
    ddp_model,
    output_folder,
    dataset,
    logger,
    clip_denoised,
    step,
    n_rounds=3,
):
    """
    Runs n_rounds of sampling with 'majority vote' (mean over N samples then round).
    Assumes dataset is VocV4Dataset producing:
      - gt_mask in [-1,1], which we map to {0,1}
      - condition_on["conditioned_image"] in [-1,1], 3×H×W
    """
    ddp_model.eval()
    batch_size = 1
    major_vote_number = 9
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_iter = iter(loader)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for _ in tqdm(range(n_rounds), desc="Generating samples / evaluating"):
            gt_mask, condition_on, name = next(loader_iter)
            set_random_seed_for_iterations(step + int(name[0].split("_")[1]))

            # Ground truth mask in {0,1}
            gt_mask01 = ((gt_mask + 1.0) / 2.0).clamp(0.0, 1.0).round()
            # Condition tensor on device
            cond = condition_on["conditioned_image"]  # (B,3,H,W) in [-1,1]
            cond = cond.to(dist_util.dev())

            # Save visualization for this batch (B=1)
            i = 0
            # Save GT (paletted red/black + raw)
            _save_mask_palette(
                gt_mask01[i : i + 1],  # (1,1,H,W) -> inside helper becomes (1,H,W)
                os.path.join(output_folder, f"{name[i]}_gt_palette.png"),
                os.path.join(output_folder, f"{name[i]}_gt.png"),
            )
            # Save conditioned spectrogram as grayscale
            _save_condition_gray(
                cond[i], os.path.join(output_folder, f"{name[i]}_condition_gray.png")
            )

            # ---- Major-vote generation (replicate the same condition N times)
            model_kwargs = {"conditioned_image": torch.cat([cond] * major_vote_number)}
            x = diffusion_model.p_sample_loop(
                ddp_model,
                (
                    major_vote_number,
                    gt_mask01.shape[1],
                    cond.shape[2],
                    cond.shape[3],
                ),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )

            # Map generated to [0,1], resize if needed, clamp, then majority vote
            x = (x + 1.0) / 2.0
            if x.shape[2] != gt_mask01.shape[2] or x.shape[3] != gt_mask01.shape[3]:
                x = F.interpolate(x, gt_mask01.shape[2:], mode="bilinear", align_corners=False)
            x = torch.clamp(x, 0.0, 1.0)

            # Majority vote over N samples -> single (1,1,H,W), then binarize
            x = x.mean(dim=0, keepdim=True).round()

            # Save model output (paletted + raw)
            _save_mask_palette(
                x[0],
                os.path.join(output_folder, f"{name[i]}_model_output_palette.png"),
                os.path.join(output_folder, f"{name[i]}_model_output.png"),
            )

            # ---- Metrics per item (here batch size is 1)
            for index, (gt_im, out_im) in enumerate(zip(gt_mask01, x)):
                # Convert to scalar {0,1} (H,W) for metrics
                gt_bin = (gt_im[0] > 0.5).float()
                out_bin = (out_im[0] > 0.5).float()

                f1, miou, wcov, fbound = calculate_metrics(out_bin, gt_bin)
                f1_score_list.append(f1)
                miou_list.append(miou)
                wcov_list.append(wcov)
                fbound_list.append(fbound)

                logger.info(
                    f"{name[index]} iou {miou_list[-1]:.4f}, f1 {f1_score_list[-1]:.4f}, "
                    f"WCov {wcov_list[-1]:.4f}, boundF {fbound_list[-1]:.4f}"
                )

    # ---- Distributed aggregation (assumes dist has been initialized via dist_util.setup_dist())
    my_length = len(miou_list)
    length_of_data = torch.tensor(my_length, device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    def _pad_to(tlist):
        return torch.tensor(
            tlist + [torch.tensor(-1.0)] * (max_len.item() - len(tlist)),
            device=dist_util.dev(),
            dtype=torch.float32,
        )

    iou_tensor = _pad_to(miou_list)
    f1_tensor = _pad_to(f1_score_list)
    wcov_tensor = _pad_to(wcov_list)
    boundf_tensor = _pad_to(fbound_list)

    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_wcov = [torch.ones_like(wcov_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_boundf = [torch.ones_like(boundf_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)
    dist.all_gather(gathered_wcov, wcov_tensor)
    dist.all_gather(gathered_boundf, boundf_tensor)

    logger.info("measure total avg")
    gathered_miou = torch.cat(gathered_miou)
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean():.4f}")

    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean():.4f}")

    gathered_wcov = torch.cat(gathered_wcov)
    gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean WCov {gathered_wcov.mean():.4f}")

    gathered_boundf = torch.cat(gathered_boundf)
    gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean boundF {gathered_boundf.mean():.4f}")

    dist.barrier()
    return gathered_miou.mean().item()
