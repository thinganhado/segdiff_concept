# deepshap_run.py
import os
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from captum.attr import DeepLiftShap

# make sure we can import the SSL model and the wrapper
import sys
SSL_REPO = "/home/opc/SSL_Anti-spoofing"
sys.path.insert(0, SSL_REPO)

from model import Model
from xai_rawwrapper import RawWrapperISTFT

# ------------- paths, match make_masks -------------
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_PER_AUDIO_ROOT = Path("/home/opc/datasets/asvspoof2019/LA/per_audio_folders")
PROTOCOL_TXT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")
OUT_DIR = Path("./deepshap_outputs_vocv4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------- config -------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/opc/asvspoof/pretrained/LA_model.pth"

SR = 16000
N_FFT, HOP, WIN, CENTER = 1024, 256, 1024, True
LOG_EPS = 1e-8

EXPECTED_LEN = 64600
K_REFS = 20
TARGET_IDX = 1

# optional, fix randomness
SEED = 123
random.seed(SEED)

# ------------- cli -------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--only-vocoder", type=str, default=None,
                    help="Comma separated vocoder filter, for example hifigan,waveglow")
parser.add_argument("--limit", type=int, default=0,
                    help="Process at most N items, 0 means no limit")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing npy or png files")
parser.add_argument("--no-png", action="store_true",
                    help="Skip writing PNGs to save time")
parser.add_argument("--rebuild-refs", action="store_true",
                    help="Ignore reference cache and rebuild")
parser.add_argument("--baseline-chunk", type=int, default=4,
                    help="How many baselines to process per Captum call, lower uses less memory")
args = parser.parse_args()

# ------------- helpers, shared with make_masks -------------
import re
line_rx = re.compile(r"^(LA_\d{4})\s+([A-Za-z0-9_]+)\s+-\s+([A-Za-z0-9_-]+)\s+spoof")

def bona_id_from_spoof(stem: str) -> Optional[str]:
    if "_" not in stem:
        return None
    candidate = stem.split("_", 1)[1]
    return candidate if re.match(r"^LA_[TDE]_\d{7}$", candidate) else None

def build_bona_index(root: Path) -> Dict[str, Path]:
    if not root.exists():
        print(f"[error] ASV19_PER_AUDIO_ROOT not found, {root}")
        return {}
    idx: Dict[str, Path] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not re.match(r"^LA_[TDE]_\d{7}$", name):
            continue
        flac = d / f"{name}.flac"
        wav = d / f"{name}.wav"
        if flac.exists():
            idx[name] = flac
        elif wav.exists():
            idx[name] = wav
    return idx

def find_spoof_audio(spoof_stem: str) -> Optional[Path]:
    p_wav = VOC_V4_AUDIO_ROOT / f"{spoof_stem}.wav"
    if p_wav.exists():
        return p_wav
    p_flac = VOC_V4_AUDIO_ROOT / f"{spoof_stem}.flac"
    if p_flac.exists():
        return p_flac
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.wav"))
    if hits:
        return hits[0]
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.flac"))
    if hits:
        return hits[0]
    return None

def iter_pairs(protocol_path: Path, vocoder_filter=None, bona_index: Dict[str, Path] = None):
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            m = line_rx.match(line.strip())
            if not m:
                continue
            spk, spoof_stem, vocoder = m.groups()
            vocoder_lc = vocoder.lower()
            if vocoder_filter and vocoder_lc not in vocoder_filter:
                continue

            sp_path = find_spoof_audio(spoof_stem)
            if sp_path is None:
                continue

            bona_id = bona_id_from_spoof(spoof_stem)
            if bona_id is None:
                continue

            bf_path = bona_index.get(bona_id)
            if bf_path is None:
                continue

            yield spk, vocoder_lc, spoof_stem, sp_path, bona_id, bf_path

def load_mono_resample(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(path, always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32, copy=False)

def pad_or_crop_1d_torch(x: torch.Tensor, length: int) -> torch.Tensor:
    n = x.numel()
    if n < length:
        return F.pad(x, (0, length - n))
    if n > length:
        return x[:length]
    return x

# ------------- STFT -------------
_HANN = None
def stft_mag_phase_torch(wav_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    global _HANN
    if _HANN is None or _HANN.device != wav_1d.device:
        _HANN = torch.hann_window(WIN, periodic=True, device=wav_1d.device)
    X = torch.stft(
        wav_1d.unsqueeze(0),
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=_HANN, center=CENTER, return_complex=True
    )  # [1, F, T]
    mag = X.abs()
    phase = X.angle()
    M_log = torch.log(mag + LOG_EPS)
    return M_log.squeeze(0), phase.squeeze(0)  # [F, T]

# ------------- model build -------------
def build_model_and_wrapper(ckpt_path: str):
    class Args: pass
    base = Model(Args(), DEVICE).to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location=DEVICE)
    base.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint {ckpt_path}")

    wrap = RawWrapperISTFT(
        base_model=base,
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        center=CENTER, use_conv_istft=False,
        expected_len=EXPECTED_LEN, is_log_mag=True,
    ).to(DEVICE).eval()
    return wrap

# ------------- plotting -------------
def save_attr_png(A: np.ndarray, out_png: Path, title: str):
    plt.figure(figsize=(7.0, 3.2))
    A_disp = np.tanh(A / (np.std(A) + 1e-8))
    plt.imshow(A_disp, origin="lower", aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("time frames")
    plt.ylabel("frequency bins")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

# ------------- reference cache -------------
REFS_CACHE = OUT_DIR / "refs_cache.npy"

def build_or_load_refs(ref_paths: List[Path], device: str) -> torch.Tensor:
    if REFS_CACHE.exists() and not args.rebuild_refs:
        arr = np.load(REFS_CACHE, allow_pickle=False)
        refs = torch.from_numpy(arr).to(device)
        print(f"Loaded refs from cache {REFS_CACHE}, shape {tuple(refs.shape)}")
        return refs
    refs_M = []
    for rp in ref_paths:
        y = load_mono_resample(rp, SR)
        wav = torch.from_numpy(y).to(device)
        wav = pad_or_crop_1d_torch(wav, EXPECTED_LEN)
        M_log, _ = stft_mag_phase_torch(wav)
        refs_M.append(M_log.unsqueeze(0))
    refs = torch.cat(refs_M, dim=0).to(device)  # [K, F, T]
    np.save(REFS_CACHE, refs.detach().cpu().numpy())
    print(f"Built refs and cached to {REFS_CACHE}, shape {tuple(refs.shape)}")
    with open(OUT_DIR / "refs_used.txt", "w") as f:
        for p in ref_paths:
            f.write(str(p) + "\n")
    return refs

# ------------- main -------------
def main():
    # optional single thread settings to reduce OpenBLAS storms
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = build_model_and_wrapper(CKPT)

    # build bona fide index and iterate pairs from protocol
    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)

    voc_filter = None
    if args.only_vocoder:
        voc_filter = {v.strip().lower() for v in args.only_vocoder.split(",") if v.strip()}

    pairs_iter = iter_pairs(PROTOCOL_TXT, vocoder_filter=voc_filter, bona_index=bona_index)
    pairs_all = list(pairs_iter)
    if not pairs_all:
        raise RuntimeError("No parallel pairs found from protocol and roots")

    if args.limit and args.limit > 0:
        pairs_all = pairs_all[:args.limit]

    # collect unique bona files for references
    all_real_paths = [bf_path for _, _, _, _, _, bf_path in pairs_all]
    ref_pool = sorted({str(p) for p in all_real_paths})
    if len(ref_pool) < K_REFS:
        print(f"[warn] only {len(ref_pool)} bona fide files available, using all as references")
        ref_paths = [Path(p) for p in ref_pool]
    else:
        ref_paths = [Path(p) for p in random.sample(ref_pool, K_REFS)]

    refs_M = build_or_load_refs(ref_paths, DEVICE)
    print(f"Refs ready, shape {tuple(refs_M.shape)}")

    explainer = DeepLiftShap(model)

    processed = 0
    for idx, (spk, vocoder, spoof_stem, sp_path, bona_id, bf_path) in enumerate(pairs_all, start=1):
        try:
            out_dir = OUT_DIR / bona_id / vocoder
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(sp_path).stem
            out_npy = out_dir / f"attr_deepshap__{stem}.npy"
            out_png = out_dir / f"attr_deepshap__{stem}.png"
            out_mask = out_dir / f"attr_deepshap_mask95__{stem}.npy"

            if not args.overwrite and out_npy.exists() and out_mask.exists() and (args.no_png or out_png.exists()):
                print(f"[skip] exists, {stem}")
                continue

            ys = load_mono_resample(sp_path, SR)
            wav_s = torch.from_numpy(ys).to(DEVICE)
            wav_s = pad_or_crop_1d_torch(wav_s, EXPECTED_LEN)
            Mq, Pq = stft_mag_phase_torch(wav_s)          # [F, T]
            Mq = Mq.unsqueeze(0).requires_grad_(True)     # [1, F, T]
            Pq = Pq.unsqueeze(0).detach()                 # [1, F, T]

            # baselines, refs for magnitude and repeat query phase K times
            K = refs_M.size(0)
            accum = None
            count = 0
            chunk = max(1, int(args.baseline_chunk))

            for start in range(0, K, chunk):
                end = min(start + chunk, K)
                bs_mag = refs_M[start:end]                 # [chunk, F, T]
                bs_phase = Pq.expand(end - start, -1, -1)  # [chunk, F, T]

                atts_chunk = explainer.attribute(
                    inputs=(Mq, Pq),
                    baselines=(bs_mag, bs_phase),
                    target=TARGET_IDX,
                )
                A_chunk = atts_chunk[0]  # [1, F, T], averaged over this chunk

                if accum is None:
                    accum = A_chunk.detach() * (end - start)
                else:
                    accum += A_chunk.detach() * (end - start)
                count += (end - start)

                del atts_chunk, A_chunk, bs_mag, bs_phase
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            A = (accum / max(count, 1)).squeeze(0).cpu().numpy()  # [F, T]

            np.save(out_npy, A.astype(np.float32))

            if not args.no_png:
                save_attr_png(A, out_png, f"DeepSHAP, spoof={stem}")

            flat = np.abs(A).ravel()
            tau = np.quantile(flat, 0.95) if flat.size else 0.0
            np.save(out_mask, (np.abs(A) >= tau).astype(np.uint8))

            with torch.no_grad():
                logits = model(Mq, Pq)
            print(f"[{idx:05d}] {stem}, logits={logits.squeeze(0).tolist()}, saved to {out_dir}")
            processed += 1

            # hygiene between items
            del Mq, Pq
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom] {sp_path}, trying to recover")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"[err] {sp_path}, {e}")
                continue
        except Exception as e:
            print(f"[err] {sp_path}, {e}")
            continue

    print(f"Done, processed {processed} of {len(pairs_all)} items")

if __name__ == "__main__":
    main()