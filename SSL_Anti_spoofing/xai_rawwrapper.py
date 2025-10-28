# xai_rawwrapper.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_complex(mag_or_log: torch.Tensor, phase: torch.Tensor, is_log_mag: bool):
    """
    Build complex STFT from magnitude and phase.
    mag_or_log: [B, F, T], either linear magnitude or log magnitude.
    phase: [B, F, T] in radians.
    is_log_mag: True if mag_or_log is log magnitude.
    Returns [..., 2] where the last dim packs (real, imag).
    """
    mag = mag_or_log.exp() if is_log_mag else mag_or_log
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.stack([real, imag], dim=-1)


class ISTFTTorch(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        win = torch.hann_window(win_length, periodic=True)
        self.register_buffer("window", win)

    def forward(self, stft_complex: torch.Tensor):
        """
        stft_complex [B, F, T, 2], F equals n_fft // 2 + 1.
        Returns waveform [B, N].
        """
        return torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            return_complex=False,
        )


class ISTFTConv1D(nn.Module):
    """
    ISTFT implemented with ConvTranspose1d and fixed sinusoid kernels.
    Input [B, 2F, T] where channels are [real bins, imag bins].
    Output [B, N] waveform.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        assert win_length == n_fft, "set win_length equal to n_fft for clean overlap add"
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.F = n_fft // 2 + 1

        t = torch.arange(win_length).float()
        hann = torch.hann_window(win_length, periodic=True)
        kernels = []
        for k in range(self.F):
            w = 2.0 * math.pi * k / n_fft
            cosw = torch.cos(w * t) * hann
            sinw = torch.sin(w * t) * hann
            kernels.append(cosw)
            kernels.append(sinw)
        weight = torch.stack(kernels, dim=0).unsqueeze(1)  # [2F, 1, W]
        self.register_buffer("weight", weight)

        # precompute overlap add normalization for Hann with hop equal to n_fft // 4
        ola = torch.zeros(win_length + 8 * hop_length)
        i = 0
        while i + win_length <= ola.numel():
            ola[i:i + win_length] += hann.pow(2)
            i += hop_length
        self.register_buffer("ola_norm", ola)

    def forward(self, x: torch.Tensor):
        # x [B, 2F, T]
        y = F.conv_transpose1d(x, self.weight, stride=self.hop_length)  # [B, 1, N]
        y = y.squeeze(1)
        norm = self.ola_norm[: y.size(-1)].clamp_min(1e-8)
        return y / norm


class RawWrapperISTFT(nn.Module):
    """
    Wrap a raw waveform model to accept spectrogram inputs for XAI.
    Forward signature: forward(M, phase) where both are [B, F, T].
    If is_log_mag is True, M is log magnitude, otherwise M is linear magnitude.
    Returns logits from the base model.
    """
    def __init__(
        self,
        base_model: nn.Module,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        center: bool = True,
        use_conv_istft: bool = False,
        expected_len: Optional[int] = None,
        is_log_mag: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.expected_len = expected_len
        self.is_log_mag = is_log_mag

        # store STFT params so callers can keep their STFT producer aligned
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        self.F = n_fft // 2 + 1
        if use_conv_istft:
            self.istft = ISTFTConv1D(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            self.conv_mode = True
        else:
            self.istft = ISTFTTorch(n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
            self.conv_mode = False

    def forward(self, M: torch.Tensor, phase: torch.Tensor):
        """
        M: [B, F, T], magnitude in linear or log domain depending on is_log_mag.
        phase: [B, F, T] in radians.
        """
        # basic shape checks to catch misaligned STFT configs
        if M.dim() != 3 or phase.dim() != 3:
            raise ValueError(f"M and phase must be [B, F, T], got {M.shape} and {phase.shape}")
        if M.shape != phase.shape:
            raise ValueError(f"M and phase must have the same shape, got {M.shape} vs {phase.shape}")
        if M.size(1) != self.F:
            raise ValueError(f"Frequency bins F must equal n_fft//2+1 = {self.F}, got {M.size(1)}")

        # build complex STFT
        C = _to_complex(M, phase, self.is_log_mag)  # [B, F, T, 2]

        if self.conv_mode:
            real = C[..., 0]
            imag = C[..., 1]
            x_ch = torch.cat([real, imag], dim=1)    # [B, 2F, T]
            wav = self.istft(x_ch)                   # [B, N]
        else:
            wav = self.istft(C)                      # [B, N]

        # the SSL_Anti spoofing Model expects [B, N, 1]
        wav = wav.unsqueeze(-1)

        # align length if needed
        tgt = self.expected_len
        if tgt is not None:
            N = wav.size(1)
            if N < tgt:
                wav = F.pad(wav, (0, 0, 0, tgt - N))
            elif N > tgt:
                wav = wav[:, :tgt, :]

        return self.base(wav)