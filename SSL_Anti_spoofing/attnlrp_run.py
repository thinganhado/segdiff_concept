# attnlrp_run.py — AttnLRP-like (γ-LRP) via gradient hooks on Conv1d / Linear
import os
import re
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- SSL model + RawWrapper ----
import sys
SSL_REPO = "/home/opc/SSL_Anti-spoofing"
sys.path.insert(0, SSL_REPO)
from model import Model
from xai_rawwrapper import RawWrapperISTFT

# ---------- paths/datasets ----------
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_PER_AUDIO_ROOT = Path("/home/opc/datasets/asvspoof2019/LA/per_audio_folders")
PROTOCOL_TXT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")

OUT_ROOT = Path("./attnlrp_outputs_vocv4")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/opc/asvspoof/pretrained/LA_model.pth"

SR = 16000
N_FFT, HOP, WIN, CENTER = 1024, 256, 1024, True
EXPECTED_LEN = 64600
TARGET_IDX = 1  # spoof class logit

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--only-vocoder", type=str, default=None, help="e.g. hifigan,waveglow (comma-separated)")
ap.add_argument("--limit", type=int, default=0, help="process at most N items (0 = all)")
ap.add_argument("--overwrite", action="store_true", help="overwrite existing npy/png")
ap.add_argument("--no-png", action="store_true", help="skip PNGs to save time")
ap.add_argument("--gamma-conv1d", type=float, default=10.0, help="γ for Conv1d (paper uses 10)")
ap.add_argument("--gamma-linear", type=float, default=0.1, help="γ for Linear (paper uses 0.1)")
args = ap.parse_args()

# ---------- routing helpers (same pattern as make_masks) ----------
line_rx = re.compile(r"^(LA_\d{4})\s+([A-Za-z0-9_]+)\s+-\s+([A-Za-z0-9_-]+)\s+spoof")

def bona_id_from_spoof(stem: str) -> Optional[str]:
    if "_" not in stem:
        return None
    cand = stem.split("_", 1)[1]
    return cand if re.match(r"^LA_[TDE]_\d{7}$", cand) else None

def build_bona_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not root.exists():
        print(f"[error] ASV19_PER_AUDIO_ROOT not found: {root}")
        return idx
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not re.match(r"^LA_[TDE]_\d{7}$", name):
            continue
        flac = d / f"{name}.flac"
        wav = d / f"{name}.wav"
        if flac.exists(): idx[name] = flac
        elif wav.exists(): idx[name] = wav
    return idx

def find_spoof_audio(spoof_stem: str) -> Optional[Path]:
    p = VOC_V4_AUDIO_ROOT / f"{spoof_stem}.wav"
    if p.exists(): return p
    p = VOC_V4_AUDIO_ROOT / f"{spoof_stem}.flac"
    if p.exists(): return p
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.wav"))
    if hits: return hits[0]
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.flac"))
    if hits: return hits[0]
    return None

def iter_pairs(protocol_path: Path, vocoder_filter=None, bona_index: Dict[str, Path] = None):
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            m = line_rx.match(line.strip())
            if not m:
                continue
            spk, spoof_stem, vocoder = m.groups()
            v = vocoder.lower()
            if vocoder_filter and v not in vocoder_filter:
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
            yield spk, v, spoof_stem, sp_path, bona_id, bf_path

# ---------- I/O + STFT ----------
def load_wav_strict(path: Path, sr: int = SR) -> torch.Tensor:
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x.mean(axis=1)
    if file_sr != sr:
        raise ValueError(f"SR mismatch: {file_sr} != {sr} for {path}")
    return torch.from_numpy(x)

def pad_or_crop_1d(x: torch.Tensor, length: int) -> torch.Tensor:
    n = x.numel()
    if n < length: return F.pad(x, (0, length - n))
    if n > length: return x[:length]
    return x

_HANN = None
def stft_mag_phase(wav_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    global _HANN
    if _HANN is None or _HANN.device != wav_1d.device:
        _HANN = torch.hann_window(WIN, periodic=True, device=wav_1d.device)
    X = torch.stft(
        wav_1d.unsqueeze(0),
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=_HANN, center=CENTER, return_complex=True
    )
    mag = X.abs()
    phase = X.angle()
    M_log = torch.log(mag + 1e-8)
    return M_log.squeeze(0), phase.squeeze(0)

def save_attr_png(A: np.ndarray, out_png: Path, title: str):
    plt.figure(figsize=(7.0, 3.2))
    A_disp = np.tanh(A / (np.std(A) + 1e-8))
    plt.imshow(A_disp, origin="lower", aspect="auto", interpolation="nearest")
    plt.title(title); plt.xlabel("time frames"); plt.ylabel("frequency bins")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

# ---------- model ----------
def build_model_and_wrapper(ckpt_path: str):
    class _Args: pass
    base = Model(_Args(), DEVICE).to(DEVICE).eval()
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

# ---------- γ-LRP style hooks ----------
def _gamma_scale_pos(x: torch.Tensor, gamma: float) -> torch.Tensor:
    # boost positive part by (1 + gamma), leave negative part unchanged
    return (1.0 + gamma) * torch.relu(x) - torch.relu(-x)

def register_gamma_hooks(
    model: torch.nn.Module,
    gamma_conv1d: float,
    gamma_linear: float,
) -> List[torch.utils.hooks.RemovableHandle]:

    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(gamma: float):
        def _hook(mod, grad_input, grad_output):
            # grad_input is a tuple; index 0 is dL/d(input)
            if not grad_input or grad_input[0] is None:
                return None
            gi = grad_input[0]
            return ( _gamma_scale_pos(gi, gamma), *grad_input[1:] )
        return _hook

    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            hooks.append(m.register_full_backward_hook(make_hook(args.gamma_conv1d)))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_full_backward_hook(make_hook(args.gamma_linear)))

    return hooks

def remove_hooks(hooks: List[torch.utils.hooks.RemovableHandle]):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

# ---------- main ----------
def main():
    # keep BLAS polite
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Build model
    model = build_model_and_wrapper(CKPT)

    # Dataset iteration
    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)
    voc_filter = None
    if args.only_vocoder:
        voc_filter = {v.strip().lower() for v in args.only_vocoder.split(",") if v.strip()}
    pairs = list(iter_pairs(PROTOCOL_TXT, vocoder_filter=voc_filter, bona_index=bona_index))
    if args.limit and args.limit > 0:
        pairs = pairs[:args.limit]

    processed = 0
    for idx, (_spk, voc, spoof_stem, sp_path, bona_id, _bf_path) in enumerate(pairs, start=1):
        out_dir = OUT_ROOT / bona_id / voc
        out_dir.mkdir(parents=True, exist_ok=True)
        out_heat = out_dir / f"attr_attnlrp__{spoof_stem}.npy"
        out_mask = out_dir / f"attr_attnlrp_mask95__{spoof_stem}.npy"
        out_png  = out_dir / f"attr_attnlrp__{spoof_stem}.png"

        if (not args.overwrite) and out_heat.exists() and out_mask.exists() and (args.no_png or out_png.exists()):
            print(f"[skip] exists {spoof_stem}")
            continue

        try:
            # Prepare inputs
            wav = load_wav_strict(sp_path).to(DEVICE)
            wav = pad_or_crop_1d(wav, EXPECTED_LEN)
            Mq, Pq = stft_mag_phase(wav)             # [F,T]
            Mq = Mq.unsqueeze(0).requires_grad_(True)
            Pq = Pq.unsqueeze(0).detach()

            # Register gamma hooks (Conv1d / Linear)
            hooks = register_gamma_hooks(model, args.gamma_conv1d, args.gamma_linear)

            # Forward + backprop spoof score
            model.zero_grad(set_to_none=True)
            if Mq.grad is not None: Mq.grad.zero_()
            logits = model(Mq, Pq)
            score = logits[:, TARGET_IDX].sum()
            score.backward()

            # Collect attribution as input gradient under γ scaling
            if Mq.grad is None:
                raise RuntimeError("Backward produced no input gradient (relevance).")
            A = Mq.grad.squeeze(0).detach().cpu().numpy()  # [F,T]

            # Cleanup hooks
            remove_hooks(hooks)

            # Save heat + 95% mask
            np.save(out_heat, A.astype(np.float32))
            flat = np.abs(A).ravel()
            tau = np.quantile(flat, 0.95) if flat.size else 0.0
            mask95 = (np.abs(A) >= tau).astype(np.uint8)
            np.save(out_mask, mask95)
            if not args.no_png:
                save_attr_png(A, out_png, f"AttnLRP (γ hooks), spoof={spoof_stem}")

            processed += 1
            print(f"[{idx:04d}] {spoof_stem} -> {out_dir}")

            # hygiene
            del Mq, Pq, logits, score
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom] {spoof_stem}: trying to recover")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            print(f"[err] {spoof_stem}: {e}")
        except Exception as e:
            print(f"[err] {spoof_stem}: {e}")

    print(f"Done, wrote {processed} items to {OUT_ROOT}")

if __name__ == "__main__":
    main()