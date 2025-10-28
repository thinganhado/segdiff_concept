# gradientshap_run.py
import os
import csv
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from captum.attr import GradientShap

# import SSL model and RawWrapper
import sys
SSL_REPO = "/home/opc/SSL_Anti-spoofing"  # change if needed
sys.path.insert(0, SSL_REPO)
from model import Model
from xai_rawwrapper import RawWrapperISTFT

# ------------- roots, match make_masks -------------
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_PER_AUDIO_ROOT = Path("/home/opc/datasets/asvspoof2019/LA/per_audio_folders")
PROTOCOL_TXT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")

# outputs
OUT_DIR = Path("./gradshap_outputs_vocv4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------- config -------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/opc/asvspoof/pretrained/LA_model.pth"

SR = 16000
N_FFT, HOP, WIN, CENTER = 1024, 256, 1024, True
LOG_EPS = 1e-8

EXPECTED_LEN = 64600
TARGET_IDX = 1                 # spoof logit
N_SAMPLES = 20                 # paper, 20 samples per utterance
STDEVS = 0.0                   # zero vector baseline, no noise

# optional, fix randomness
SEED = 123
random.seed(SEED)

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

def stft_mag_phase_torch(wav_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(WIN, periodic=True, device=wav_1d.device)
    X = torch.stft(
        wav_1d.unsqueeze(0),
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=window, center=CENTER, return_complex=True
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

# ------------- main -------------
def main():
    model = build_model_and_wrapper(CKPT)

    # discover pairs, same as make_masks
    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)
    pairs = list(iter_pairs(PROTOCOL_TXT, vocoder_filter=None, bona_index=bona_index))
    if not pairs:
        raise RuntimeError("No parallel pairs found from protocol and roots")

    explainer = GradientShap(model)

    # process each spoof
    for idx, (spk, vocoder, spoof_stem, sp_path, bona_id, bf_path) in enumerate(pairs, start=1):
        try:
            ys = load_mono_resample(sp_path, SR)
            wav_s = torch.from_numpy(ys).to(DEVICE)
            wav_s = pad_or_crop_1d_torch(wav_s, EXPECTED_LEN)
            Mq, Pq = stft_mag_phase_torch(wav_s)          # [F, T]
            Mq = Mq.unsqueeze(0).requires_grad_(True)     # [1, F, T]
            Pq = Pq.unsqueeze(0).detach()                 # [1, F, T]

            # zero baselines, paper setup, 20 samples per utterance
            base_M = torch.zeros_like(Mq)                 # [1, F, T]
            base_P = Pq.clone()                           # keep phase fixed
            atts_M, atts_P = explainer.attribute(
                inputs=(Mq, Pq),
                baselines=(base_M, base_P),
                target=TARGET_IDX,
                n_samples=N_SAMPLES,
                stdevs=STDEVS,
            )

            A = atts_M.squeeze(0).detach().cpu().numpy()  # [F, T], magnitude attribution

            # save under the same structure as make_masks for easy comparison
            out_dir = OUT_DIR / bona_id / vocoder
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(sp_path).stem
            np.save(out_dir / f"attr_gradshap__{stem}.npy", A.astype(np.float32))
            save_attr_png(A, out_dir / f"attr_gradshap__{stem}.png",
                          f"GradientSHAP, spoof={stem}")

            # 95 percent binarized mask for Figure 3 style
            flat = np.abs(A).ravel()
            tau = np.quantile(flat, 0.95) if flat.size else 0.0
            mask95 = (np.abs(A) >= tau).astype(np.uint8)
            np.save(out_dir / f"attr_gradshap_mask95__{stem}.npy", mask95)

            with torch.no_grad():
                logits = model(Mq, Pq)
            print(f"[{idx:04d}] {stem}, logits={logits.squeeze(0).tolist()}, saved to {out_dir}")

        except Exception as e:
            print(f"[{idx:04d}] {sp_path}, error {e}")

if __name__ == "__main__":
    # quiet down OpenBLAS warnings, and keep CPU threads low
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    main()