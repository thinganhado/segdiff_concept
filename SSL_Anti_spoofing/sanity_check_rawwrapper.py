# sanity_check_rawwrapper.py
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf

from model import Model
from xai_rawwrapper import RawWrapperISTFT

# ===== config, adjust only if needed =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/opc/asvspoof/pretrained/LA_model.pth"
PAIRS_CSV = "/home/opc/pairs20/pairs20_paths.csv"

SR = 16000
N_FFT, HOP, WIN, CENTER = 1024, 256, 1024, True
# ========================================

def load_wav_mono(path: str, target_sr: int = SR):
    x, srr = sf.read(path, dtype="float32", always_2d=False)
    if srr != target_sr:
        raise ValueError(f"SR mismatch, file {path} has {srr}, expected {target_sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return torch.from_numpy(x)

def stft_mag_phase(wav_1d: torch.Tensor, n_fft=N_FFT, hop=HOP, win=WIN, center=CENTER):
    window = torch.hann_window(win, periodic=True, device=wav_1d.device)
    X = torch.stft(
        wav_1d.unsqueeze(0),  # [1, N]
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=center,
        return_complex=True,
    )  # [1, F, T]
    mag = X.abs()
    phase = X.angle()
    M_log = torch.log(mag + 1e-8)
    return M_log.squeeze(0), phase.squeeze(0)  # [F, T] each

def build_base_model(ckpt_path: str):
    class Args: pass
    base = Model(Args(), DEVICE).to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location=DEVICE)
    base.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint {ckpt_path}")
    return base

def compare_one_file(path: str, base_model: torch.nn.Module):
    # load raw waveform
    wav = load_wav_mono(path).to(DEVICE)                 # [N]
    N = wav.numel()

    # build wrapper with expected_len equal to this file length
    wrapper = RawWrapperISTFT(
        base_model=base_model,
        n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        center=CENTER, use_conv_istft=False,
        expected_len=N, is_log_mag=True,
    ).to(DEVICE).eval()

    # raw path
    with torch.no_grad():
        logits_raw = base_model(wav.unsqueeze(0).unsqueeze(-1))  # [1, 2]

    # wrapped path
    M_log, phase = stft_mag_phase(wav)                           # [F, T]
    M_log = M_log.unsqueeze(0)                                   # [1, F, T]
    phase = phase.unsqueeze(0)                                   # [1, F, T]
    with torch.no_grad():
        logits_wrap = wrapper(M_log, phase)                      # [1, 2]

    # compare
    cos = F.cosine_similarity(logits_raw, logits_wrap, dim=-1)[0].item()
    l1 = torch.mean(torch.abs(logits_raw - logits_wrap), dim=-1)[0].item()
    return logits_raw[0].tolist(), logits_wrap[0].tolist(), cos, l1

def read_pairs(csv_path: str, limit_per_col: int = 4):
    reals, spoofs = [], []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("real_path"):
                reals.append(row["real_path"])
            if row.get("spoof_path"):
                spoofs.append(row["spoof_path"])
    # take a few from each to keep runtime short
    return reals[:limit_per_col] + spoofs[:limit_per_col]

def main():
    paths = read_pairs(PAIRS_CSV, limit_per_col=4)
    assert len(paths) > 0, f"No paths read from {PAIRS_CSV}"
    base = build_base_model(CKPT)

    print("\n=== Sanity check, raw vs wrapper on ISTFT(STFT(x)) ===")
    agg_cos, agg_l1, cnt = 0.0, 0.0, 0
    for i, p in enumerate(paths):
        try:
            logits_raw, logits_wrap, cos, l1 = compare_one_file(p, base)
            print(f"[{i}] {p}")
            print(f"  raw logits    {logits_raw}")
            print(f"  wrap logits   {logits_wrap}")
            print(f"  cosine sim    {cos:.6f}")
            print(f"  L1 difference {l1:.6f}")
            agg_cos += cos
            agg_l1 += l1
            cnt += 1
        except Exception as e:
            print(f"[{i}] {p}  ERROR {e}")

    if cnt:
        print("\nAggregate")
        print(f"  mean cosine sim = {agg_cos / cnt:.6f}")
        print(f"  mean L1 diff    = {agg_l1 / cnt:.6f}")

if __name__ == "__main__":
    main()