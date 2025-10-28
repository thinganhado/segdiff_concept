# save as attnlrp_overlay.py
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ---- repo paths ----
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "core_scripts"))
sys.path.insert(0, str(REPO_ROOT / "core_scripts" / "config_parse"))
from core_scripts.config_parse.arg_parse import f_args_parsed
from model import Model
from xai_rawwrapper import RawWrapperISTFT

# ---- defaults (paper-like) ----
SR = 16000
N_FFT_DEF = 320
HOP_DEF = 160
WIN_DEF = 320
CENTER_DEF = True
LOG_EPS = 1e-8
EXPECTED_LEN_DEFAULT = 64600

# ----------------- I/O helpers -----------------
def load_wav(path: Path, sr: int = SR) -> np.ndarray:
    x, s = sf.read(str(path), dtype="float32", always_2d=False)
    if s != sr:
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return x

def stft_mag_db(x: np.ndarray, n_fft: int, hop: int, win: int, center: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, center=center, window="hann")
    mag = np.abs(S) + 1e-10
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag_db.shape[1]), sr=SR, hop_length=hop, n_fft=n_fft)
    return mag_db, freqs, times

def make_gray_bg(ax, mag_db, freqs, times, alpha: float = 0.35):
    ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        cmap="gray",
        vmin=mag_db.min(),
        vmax=mag_db.max(),
        alpha=alpha,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

def overlay_bipolar(ax, grid, freqs, times, alpha_scale: float = 0.85, clip_q: float = 0.985):
    v = np.quantile(np.abs(grid), clip_q) + 1e-9
    norm = np.clip(grid / v, -1, 1)
    pos = np.clip(norm, 0, 1)
    neg = np.clip(-norm, 0, 1)
    alpha = (pos + neg) * alpha_scale
    rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = neg      # red (class 0 evidence)
    rgba[..., 1] = pos      # green (class 1 evidence)
    rgba[..., 2] = 0.0
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
    )

def overlay_single(ax, grid, freqs, times, clip_q: float = 0.985, show_colorbar: bool = False):
    # Non-negative heatmap, paper-style
    grid = np.clip(grid, 0, None)
    v = np.quantile(grid, clip_q) + 1e-9
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        cmap="Reds",
        vmin=0.0,
        vmax=v,
        interpolation="nearest",
    )
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relevance")

def _pad_or_crop_1d(y: torch.Tensor, length: int) -> torch.Tensor:
    n = y.numel()
    if n < length:
        return F.pad(y, (0, length - n))
    if n > length:
        return y[:length]
    return y

def _avgpool_and_mask(A: np.ndarray, k_t: int, k_f: int, top_q: float) -> np.ndarray:
    if k_t > 1 or k_f > 1:
        x = torch.from_numpy(A).unsqueeze(0).unsqueeze(0)  # [1,1,F,T]
        y = F.avg_pool2d(x, kernel_size=(k_f, k_t), stride=(k_f, k_t), ceil_mode=True)
        y = F.interpolate(y, size=A.shape, mode="bilinear", align_corners=False)
        A = y.squeeze().numpy()
    if 0.0 < top_q < 1.0:
        tau = np.quantile(np.abs(A), top_q)
        A[np.abs(A) < tau] = 0.0
    return A

# ----------------- AttnLRP-like core (γ-scaled input gradient) -----------------
def _gamma_scale_pos(x: torch.Tensor, gamma: float) -> torch.Tensor:
    return (1.0 + gamma) * torch.relu(x) - torch.relu(-x)

def _register_gamma_hooks(model: torch.nn.Module, gamma_conv1d: float, gamma_linear: float):
    hooks = []
    def make_hook(gamma: float):
        def _hook(mod, grad_input, grad_output):
            if not grad_input or grad_input[0] is None:
                return None
            gi0 = grad_input[0]
            return (_gamma_scale_pos(gi0, gamma), *grad_input[1:])
        return _hook
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            hooks.append(m.register_full_backward_hook(make_hook(gamma_conv1d)))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_full_backward_hook(make_hook(gamma_linear)))
    return hooks

def _remove_hooks(hooks):
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

def attnlrp_like_attr_M(model: torch.nn.Module,
                        x_np: np.ndarray,
                        expected_len: int,
                        target_idx: int,
                        n_fft: int,
                        hop: int,
                        win: int,
                        center: bool,
                        gamma_conv1d: float,
                        gamma_linear: float) -> np.ndarray:
    """
    Compute γ-scaled input gradient on log-magnitude M (fast AttnLRP-like).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    wrapper = RawWrapperISTFT(
        base_model=model,
        n_fft=n_fft, hop_length=hop, win_length=win, center=center,
        use_conv_istft=False, expected_len=expected_len, is_log_mag=True,
    ).to(device).eval()

    y = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = _pad_or_crop_1d(y, expected_len)

    window = torch.hann_window(win, periodic=True, device=device)
    X = torch.stft(
        y.unsqueeze(0),
        n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=center, return_complex=True
    )  # [1, F, T]
    M_log = torch.log(X.abs() + LOG_EPS)   # [1, F, T]
    P = X.angle()                          # [1, F, T]
    M_log = M_log.requires_grad_(True)

    hooks = _register_gamma_hooks(wrapper, gamma_conv1d, gamma_linear)
    try:
        wrapper.zero_grad(set_to_none=True)
        if M_log.grad is not None:
            M_log.grad.zero_()
        logits = wrapper(M_log, P)                 # [1, 2]
        score = logits[:, target_idx].sum()
        score.backward()
        if M_log.grad is None:
            raise RuntimeError("No input gradient obtained.")
        A = M_log.grad.squeeze(0).detach().cpu().numpy()  # [F, T]
    finally:
        _remove_hooks(hooks)
    return A

# ----------------- metadata helpers -----------------
def _read_label_from_meta(meta_path: Optional[Path], utt_id: str) -> Optional[int]:
    """
    Return class index (spoof=0, bonafide=1) if found in metadata, else None.
    Matches either a plain token (e.g., LA_E_1000048) or 'LA2021-LA_E_1000048'.
    """
    if not meta_path or not meta_path.exists():
        return None
    key = utt_id
    alt_key = f"LA2021-{utt_id}"
    try:
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if key in line or alt_key in line:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    # last meaningful token containing class string
                    # metadata often has "... spoof ... eval" or "... bonafide ... eval"
                    # find first token that equals those
                    label = None
                    for tok in tokens:
                        t = tok.lower()
                        if t == "spoof":
                            label = 0
                            break
                        if t == "bonafide":
                            label = 1
                            break
                    return label
    except Exception:
        return None
    return None

def _model_argmax_class(model: torch.nn.Module, x_np: np.ndarray,
                        expected_len: int, n_fft: int, hop: int, win: int, center: bool) -> int:
    device = next(model.parameters()).device
    y = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = _pad_or_crop_1d(y, expected_len)
    window = torch.hann_window(win, periodic=True, device=device)
    X = torch.stft(y.unsqueeze(0), n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, center=center, return_complex=True)
    M_log = torch.log(X.abs() + LOG_EPS)
    P = X.angle()
    wrapper = RawWrapperISTFT(model, n_fft=n_fft, hop_length=hop, win_length=win,
                              center=center, use_conv_istft=False,
                              expected_len=expected_len, is_log_mag=True).to(device).eval()
    with torch.no_grad():
        logits = wrapper(M_log, P)  # [1,2]
    return int(torch.argmax(logits, dim=1).item())

# ----------------- main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True, help="SSL_Anti-spoofing checkpoint path")
    p.add_argument("--out", type=Path, required=True)

    # class handling / metadata
    p.add_argument("--meta", type=Path, default=None, help="trial_metadata.txt to resolve class when --target=auto")
    p.add_argument("--target", choices=["auto", "diff", "0", "1"], default="auto",
                   help="auto=use metadata (or model argmax fallback), diff=class1-class0, or fixed class 0/1")
    p.add_argument("--show_colorbar", action="store_true", help="show colorbar in single-class mode")

    # STFT grid
    p.add_argument("--n_fft", type=int, default=N_FFT_DEF)
    p.add_argument("--hop", type=int, default=HOP_DEF)
    p.add_argument("--win", type=int, default=WIN_DEF)
    p.add_argument("--center", type=int, default=1)

    # model input length
    p.add_argument("--expected_len", type=int, default=EXPECTED_LEN_DEFAULT)

    # gamma settings
    p.add_argument("--gamma_conv1d", type=float, default=10.0)
    p.add_argument("--gamma_linear", type=float, default=0.1)

    # viz / smoothing
    p.add_argument("--pool_t", type=int, default=3)
    p.add_argument("--pool_f", type=int, default=3)
    p.add_argument("--mask_q", type=float, default=0.95)
    p.add_argument("--clip_q", type=float, default=0.985)
    p.add_argument("--bg_alpha", type=float, default=0.35)
    args = p.parse_args()

    # --- audio ---
    x = load_wav(args.wav, SR)
    audio_stem = Path(args.wav).stem
    # normalize to base utt id like LA_E_1000048 from LA_E_1000048-flac variant names
    base_id = audio_stem.split("-")[0]

    # --- device & model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    repo_args = f_args_parsed([
        "--module-model", "model",
        "--module-config", "config",
        "--batch-size", "1",
        "--inference",
    ])
    setattr(repo_args, "trained_model", str(args.ckpt))
    setattr(repo_args, "save_model_dir", "./")
    setattr(repo_args, "eval_mode_for_validation", True)

    model = Model(repo_args, device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # --- STFT params ---
    n_fft = int(args.n_fft); hop = int(args.hop); win = int(args.win); center = bool(args.center)

    # --- choose target / compute attribution ---
    title_suffix = ""
    if args.target == "diff":
        A1 = attnlrp_like_attr_M(model, x, args.expected_len, 1, n_fft, hop, win, center,
                                 args.gamma_conv1d, args.gamma_linear)
        A0 = attnlrp_like_attr_M(model, x, args.expected_len, 0, n_fft, hop, win, center,
                                 args.gamma_conv1d, args.gamma_linear)
        A = A1 - A0
        mode = "diff"
        title_suffix = "class1 minus class0"
        subfolder = None
    else:
        if args.target in ("0", "1"):
            tgt = int(args.target)
            resolved = tgt
            resolved_src = f"fixed {tgt}"
        else:
            # auto: try metadata first, else model argmax
            label = _read_label_from_meta(args.meta, base_id)
            if label is None:
                label = _model_argmax_class(model, x, args.expected_len, n_fft, hop, win, center)
                resolved_src = f"model-argmax ({label})"
            else:
                resolved_src = f"metadata ({'spoof' if label==0 else 'bonafide'})"
            resolved = int(label)
        A = attnlrp_like_attr_M(model, x, args.expected_len, resolved, n_fft, hop, win, center,
                                args.gamma_conv1d, args.gamma_linear)
        mode = "single"
        cls_name = "spoof" if resolved == 0 else "bonafide"
        title_suffix = f"class {resolved} [{cls_name}] • {resolved_src}"
        subfolder = cls_name

    # smoothing & sparsifying
    A = _avgpool_and_mask(A, k_t=max(1, args.pool_t), k_f=max(1, args.pool_f), top_q=float(args.mask_q))

    # background spectrogram
    mag_db, freqs, times = stft_mag_db(x, n_fft=n_fft, hop=hop, win=win, center=center)

    # --- resolve output path ---
    out_path = args.out
    if out_path.suffix.lower() == ".png":
        final_out = out_path
    else:
        # treat as directory
        if mode == "single" and subfolder is not None:
            final_out = out_path / subfolder / f"{audio_stem}.png"
        else:
            final_out = out_path / f"{audio_stem}.png"
    final_out.parent.mkdir(parents=True, exist_ok=True)

    # --- plot ---
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(2, 1, 1)
    make_gray_bg(ax1, mag_db, freqs, times, alpha=float(args.bg_alpha))
    ax1.set_title("(a) magnitude spectrogram")

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(freqs[0] / 1000.0, freqs[-1] / 1000.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (kHz)")
    make_gray_bg(ax2, mag_db, freqs, times, alpha=float(args.bg_alpha))

    if mode == "diff":
        overlay_bipolar(ax2, A, freqs, times, alpha_scale=0.90, clip_q=float(args.clip_q))
        ax2.set_title(f"(b) AttnLRP-like relevance, {title_suffix}")
    else:
        overlay_single(ax2, A, freqs, times, clip_q=float(args.clip_q), show_colorbar=bool(args.show_colorbar))
        ax2.set_title(f"(b) AttnLRP-like relevance, {title_suffix}")

    plt.tight_layout()
    plt.savefig(str(final_out), dpi=200)
    print(f"Saved {final_out}")

if __name__ == "__main__":
    main()