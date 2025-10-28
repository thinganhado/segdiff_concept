# save as gradshap_overlay.py
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

# NumPy alias shim for older libs that use np.float, np.int, np.complex
import numpy as _np
if not hasattr(_np, "float"): _np.float = float
if not hasattr(_np, "int"): _np.int = int
if not hasattr(_np, "complex"): _np.complex = _np.complex128
if not hasattr(_np, "bool"): _np.bool = bool

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from captum.attr import GradientShap

# repo paths
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "core_scripts"))
sys.path.insert(0, str(REPO_ROOT / "core_scripts" / "config_parse"))
from core_scripts.config_parse.arg_parse import f_args_parsed

# wrapper
from xai_rawwrapper import RawWrapperISTFT

# defaults
SR = 16000
N_FFT_DEF = 320
HOP_DEF = 160
WIN_DEF = 320
CENTER_DEF = True
LOG_EPS = 1e-8
EXPECTED_LEN_DEFAULT = 64600

# ---------------- I/O helpers ----------------
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

def make_gray_bg(ax, mag_db, freqs, times, bg_alpha: float = 0.35):
    ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        cmap="gray",
        vmin=mag_db.min(),
        vmax=mag_db.max(),
        alpha=bg_alpha,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

def overlay_single_heat(ax, shap_nonneg, freqs, times, clip_q: float, alpha: float, cmap: str, add_cbar: bool):
    """Sequential colormap, darker means larger SHAP, overlaid on faded spectrogram."""
    vmax = np.quantile(shap_nonneg, clip_q) + 1e-9
    im = ax.imshow(
        shap_nonneg,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        alpha=alpha,
    )
    if add_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("The value of SHAP")

def _pad_or_crop_1d(y: torch.Tensor, length: int) -> torch.Tensor:
    n = y.numel()
    if n < length:
        return F.pad(y, (0, length - n))
    if n > length:
        return y[:length]
    return y

def _avgpool_and_mask(A: np.ndarray, k_t: int, k_f: int, top_q: float) -> np.ndarray:
    """Average pool A with kernel [k_f, k_t], then keep top quantile by abs value."""
    if k_t > 1 or k_f > 1:
        import torch as _torch
        x = _torch.from_numpy(A).unsqueeze(0).unsqueeze(0)  # [1,1,F,T]
        y = F.avg_pool2d(x, kernel_size=(k_f, k_t), stride=(k_f, k_t), ceil_mode=True)
        y = F.interpolate(y, size=A.shape, mode="bilinear", align_corners=False)
        A = y.squeeze().numpy()
    if 0.0 < top_q < 1.0:
        tau = np.quantile(np.abs(A), top_q)
        A[np.abs(A) < tau] = 0.0
    return A

# ---------------- metadata utils ----------------
def parse_metadata(meta_path: Optional[Path]) -> Dict[str, int]:
    """
    Return dict: utt_id -> class_idx, where 0 = spoof, 1 = bonafide.
    Accepts many token formats, for example:
      LA_E_1000048, LA2021-LA_E_1000048, LA_E_1000048-opus-loc_tx
    """
    mapping: Dict[str, int] = {}
    if not meta_path:
        return mapping
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            toks = line.strip().split()
            core = None
            for t in toks:
                if "LA_E_" in t:
                    start = t.find("LA_E_")
                    frag = t[start:]
                    core = frag.split("-", 1)[0]
                    core = core.split(".", 1)[0]
                    break
            if core is None:
                continue
            is_bona = any(tok.lower() == "bonafide" for tok in toks)
            is_spoof = any(tok.lower() == "spoof" for tok in toks)
            if is_bona:
                mapping[core] = 1
            elif is_spoof:
                mapping[core] = 0
    return mapping

def id_from_path(audio_path: Path) -> str:
    """Return bare id like LA_E_1000048 from file name."""
    stem = audio_path.stem
    if "LA_E_" in stem:
        i = stem.find("LA_E_")
        stem = stem[i:]
    return stem.split("-", 1)[0]

# ---------------- model utils ----------------
def predict_class(model, x_np: np.ndarray, expected_len: int) -> Tuple[int, float]:
    """Return (pred_class, margin). Margin is softmax difference between top and runner up."""
    device = next(model.parameters()).device
    y = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = _pad_or_crop_1d(y, expected_len)
    y = y.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    with torch.no_grad():
        logits = model(y)  # [1, 2]
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    order = probs.argsort()[::-1]
    top, second = probs[order[0]], probs[order[1]]
    return int(order[0]), float(top - second)

# ---------------- SHAP core ----------------
def gradshap_mag_with_fixed_phase(model: torch.nn.Module,
                                  x_np: np.ndarray,
                                  n_samples: int,
                                  expected_len: int,
                                  target_idx: int,
                                  n_fft: int,
                                  hop: int,
                                  win: int,
                                  center: bool,
                                  stdevs: float = 0.0) -> np.ndarray:
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

    explainer = GradientShap(wrapper)
    torch.manual_seed(0)
    base_M = torch.zeros_like(M_log)
    base_P = P.clone()

    atts_M, _ = explainer.attribute(
        inputs=(M_log, P),
        baselines=(base_M, base_P),
        target=target_idx,
        n_samples=n_samples,
        stdevs=stdevs,
    )
    A = atts_M.squeeze(0).detach().cpu().numpy()  # [F, T]
    return A

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True, help="SSL_Anti-spoofing checkpoint path")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--expected_len", type=int, default=EXPECTED_LEN_DEFAULT)
    # STFT grid
    p.add_argument("--n_fft", type=int, default=N_FFT_DEF)
    p.add_argument("--hop", type=int, default=HOP_DEF)
    p.add_argument("--win", type=int, default=WIN_DEF)
    p.add_argument("--center", type=int, default=1, help="1 or 0")
    # display
    p.add_argument("--pool_t", type=int, default=3, help="avg pool kernel in time, frames")
    p.add_argument("--pool_f", type=int, default=3, help="avg pool kernel in freq, bins")
    p.add_argument("--mask_q", type=float, default=0.95, help="keep top quantile by abs value")
    p.add_argument("--clip_q", type=float, default=0.985, help="quantile for color clipping")
    p.add_argument("--bg_alpha", type=float, default=0.35, help="spectrogram fade")
    p.add_argument("--heat_alpha", type=float, default=0.90, help="overlay heat transparency")
    p.add_argument("--colorbar", action="store_true", help="show colorbar")
    # class selection
    p.add_argument("--meta", type=Path, default=None, help="trial_metadata.txt path, optional")
    p.add_argument("--target", choices=["auto", "meta", "pred", "0", "1"], default="auto",
                   help="which class to attribute, 0 spoof, 1 bonafide")
    p.add_argument("--require_correct", action="store_true",
                   help="if both meta and prediction exist, skip saving unless they match")
    p.add_argument("--min_margin", type=float, default=0.0,
                   help="skip saving unless prediction margin is at least this value")
    args = p.parse_args()

    # import model only now
    from model import Model

    # load audio
    x = load_wav(args.wav, SR)
    audio_stem = Path(args.wav).stem

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load downstream checkpoint
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # build model (same way as before, no edits to your model.py)
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

    # metadata class
    meta_map = parse_metadata(args.meta) if args.meta else {}
    utt_id = id_from_path(args.wav)
    meta_cls = meta_map.get(utt_id, None)  # 0 spoof, 1 bonafide, or None

    # model prediction
    pred_cls, margin = predict_class(model, x, args.expected_len)

    # choose target class
    if args.target in ("0", "1"):
        target_cls = int(args.target)
        chosen_via = "cli"
    elif args.target == "meta" and meta_cls is not None:
        target_cls = meta_cls
        chosen_via = "meta"
    elif args.target == "pred":
        target_cls = pred_cls
        chosen_via = "pred"
    else:  # auto
        if meta_cls is not None:
            target_cls = meta_cls
            chosen_via = "auto(meta)"
        else:
            target_cls = pred_cls
            chosen_via = "auto(pred)"

    # optional gating: require correct prediction and margin
    if args.require_correct and meta_cls is not None and pred_cls != meta_cls:
        print(f"[skip] {utt_id} prediction {pred_cls} disagrees with meta {meta_cls}")
        return
    if margin < float(args.min_margin):
        print(f"[skip] {utt_id} low margin {margin:.3f} < {args.min_margin}")
        return

    # unpack STFT grid
    n_fft = int(args.n_fft)
    hop = int(args.hop)
    win = int(args.win)
    center = bool(args.center)

    # SHAP for the chosen class
    A = gradshap_mag_with_fixed_phase(
        model, x,
        n_samples=args.n_samples,
        expected_len=args.expected_len,
        target_idx=target_cls,
        n_fft=n_fft, hop=hop, win=win, center=center,
        stdevs=0.0
    )

    # display a single class map like the paper: nonnegative values only
    A_disp = np.clip(A, a_min=0.0, a_max=None)
    A_disp = _avgpool_and_mask(A_disp, k_t=max(1, args.pool_t), k_f=max(1, args.pool_f), top_q=float(args.mask_q))

    # background spectrogram
    mag_db, freqs, times = stft_mag_db(x, n_fft=n_fft, hop=hop, win=win, center=center)

    # --- decide color and subfolder by class ---
    cls_name = "bonafide" if target_cls == 1 else "spoof"
    cmap_for_class = "Blues" if target_cls == 1 else "Reds"

    # --- resolve output path ---
    out_path = args.out
    if out_path.suffix.lower() == ".png":
        final_out = out_path
    else:
        # treat as directory, split by class subfolder
        final_out = out_path / cls_name / f"{audio_stem}.png"
    final_out.parent.mkdir(parents=True, exist_ok=True)

    # figure
    plt.figure(figsize=(10, 7.2))

    # top: faded spectrogram
    ax1 = plt.subplot(2, 1, 1)
    make_gray_bg(ax1, mag_db, freqs, times, bg_alpha=float(args.bg_alpha))
    ax1.set_title("(a) magnitude spectrogram")

    # bottom: spectrogram + single class SHAP heat
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(freqs[0] / 1000.0, freqs[-1] / 1000.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (kHz)")
    make_gray_bg(ax2, mag_db, freqs, times, bg_alpha=float(args.bg_alpha))

    overlay_single_heat(
        ax2,
        shap_nonneg=A_disp,
        freqs=freqs,
        times=times,
        clip_q=float(args.clip_q),
        alpha=float(args.heat_alpha),
        cmap=cmap_for_class,              # auto: blue for bonafide, red for spoof
        add_cbar=bool(args.colorbar),
    )
    ax2.set_title(f"(b) SHAP values, class {target_cls} [{cls_name}] ({chosen_via}, margin {margin:.3f})")

    plt.tight_layout()
    plt.savefig(str(final_out), dpi=200)
    print(f"Saved {final_out}")

if __name__ == "__main__":
    main()