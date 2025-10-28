# annotate_diff_regions.py  (old region defaults + Tier2/Tier3 features)
import json, csv, math, argparse, re, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter1d, gaussian_filter,
    binary_closing, binary_opening, label, find_objects
)

# Optional Praat F0
try:
    import parselmouth
    HAVE_PRAAT = True
except Exception:
    HAVE_PRAAT = False

# ---------------------------- constants / defaults ----------------------------
TARGET_SR = 16000
N_FFT = 1024
HOP = 256
WIN = 1024
CENTER = True

# smoothing, separable Gaussian [time, freq] — same as before
VAR_T, VAR_F = 3.0, 5.0
SIGMA_T, SIGMA_F = math.sqrt(VAR_T), math.sqrt(VAR_F)
TRUNC_T = ((3 - 1) / 2.0) / max(SIGMA_T, 1e-6)   # ksize=3 time
TRUNC_F = ((11 - 1) / 2.0) / max(SIGMA_F, 1e-6)  # ksize=11 freq

EPS = 1e-8
BAND_BINS = [(0.0, 1500.0), (1500.0, 3000.0), (3000.0, 8000.0)]

# For Tier 3 (phone classes)
VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
SIL_TOKENS = {"sil","sp","spn","pau","nsn"}

# ---------------------------- helpers: basic STFT & geometry ----------------------------
def hz_per_bin(sr=TARGET_SR, n_fft=N_FFT):
    return sr / 2.0 / (n_fft // 2)

def time_per_col(hop=HOP, sr=TARGET_SR):
    return hop / sr

def _extent(frames, sr=TARGET_SR, hop=HOP, n_fft=N_FFT):
    t_max = frames * hop / sr
    f_max = sr / 2.0
    return [0.0, t_max, 0.0, f_max]

def _stretch01(x):
    x = x - np.nanmin(x)
    vmax = np.percentile(x, 98)
    if vmax > 0:
        x = np.clip(x / vmax, 0, 1)
    return x

def smooth_2d(S: np.ndarray):
    St = gaussian_filter1d(S, sigma=SIGMA_T, axis=1, truncate=TRUNC_T)
    Sf = gaussian_filter1d(St, sigma=SIGMA_F, axis=0, truncate=TRUNC_F)
    return Sf

# ---------------------------- load precomputed D/mask ----------------------------
def load_precomputed(pair_dir: str, spoof_stem: str):
    pair = Path(pair_dir)
    mpath = pair / f"mask95_smoothed__{spoof_stem}.npy"
    if not mpath.exists():
        raise FileNotFoundError(f"mask file not found, {mpath}")
    mask = np.load(mpath).astype(bool)

    G_Mb = None
    mb_path = pair / "Mb_smooth.npy"
    if mb_path.exists():
        G_Mb = np.load(mb_path).astype(np.float32)

    # density proxy
    ms_path = pair / f"Ms_smooth__{spoof_stem}.npy"
    if ms_path.exists() and G_Mb is not None:
        G_Ms = np.load(ms_path).astype(np.float32)
        D = np.abs(G_Ms - G_Mb) / (np.abs(G_Mb) + EPS)
    else:
        # fallback: smoothed mask as density
        D = gaussian_filter(mask.astype(np.float32), sigma=(2.0, 1.0))

    D = np.where(mask, D, 0.0).astype(np.float32)
    return G_Mb, D, mask

# ---------------------------- region data ----------------------------
@dataclass
class Region:
    bbox: Tuple[int, int, int, int]   # r0, c0, r1, c1
    sal_mean: float
    sal_sum: float
    sal_peak: float
    area_px: int
    t_start: float
    t_end: float
    f_low: float
    f_high: float
    band_bin: str
    mask_cov: float
    rid: int = -1
    # ---- Tier 2 (geometry/intensity) ----
    width_cols: int = 0
    height_bins: int = 0
    aspect_ft: float = 0.0
    center_t: float = 0.0
    center_f: float = 0.0
    compactness: float = 0.0
    sal95: float = 0.0
    # ---- Tier 3 (speech, energy, F0, phones, words) ----
    speech_frac: float = 0.0
    ls_frac: float = 0.0
    ms_frac: float = 0.0
    hs_frac: float = 0.0
    f0_mean: float = 0.0
    f0_median: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    phone_major: str = "none"
    phone_entropy: float = 0.0
    words_covered: int = 0

# ---------------------------- region ops ----------------------------
def band_of_bbox(bbox):
    r0, c0, r1, c1 = bbox
    f0 = r0 * hz_per_bin(); f1 = r1 * hz_per_bin()
    mid = 0.5 * (f0 + f1)
    for i, (lo, hi) in enumerate(BAND_BINS):
        if lo <= mid < hi:
            return ["low", "mid", "high"][i]
    return "mid"

def regions_from_mask(D: np.ndarray,
                      mask: np.ndarray,
                      min_area_px: int,
                      min_dur_s: float,
                      min_bw_hz: float,
                      min_cov: float) -> List[Region]:
    assert D.shape == mask.shape, f"D and mask shapes must match, got {D.shape} vs {mask.shape}"
    lab, _ = label(mask)
    slices = find_objects(lab)
    regs: List[Region] = []
    fbin = hz_per_bin(); tbin = time_per_col()

    for slc in slices:
        if slc is None:
            continue
        r0, r1 = slc[0].start, slc[0].stop
        c0, c1 = slc[1].start, slc[1].stop

        sub_mask = mask[r0:r1, c0:c1]
        sub_D    = D[r0:r1, c0:c1]
        if sub_mask.size == 0 or sub_D.size == 0:
            continue
        if sub_mask.shape != sub_D.shape:
            h = min(sub_mask.shape[0], sub_D.shape[0])
            w = min(sub_mask.shape[1], sub_D.shape[1])
            sub_mask = sub_mask[:h, :w]
            sub_D    = sub_D[:h, :w]

        area = int(sub_mask.sum())
        if area < min_area_px:
            continue

        dur = (c1 - c0) * tbin
        bw  = (r1 - r0) * fbin
        if dur < min_dur_s or bw < min_bw_hz:
            continue

        cov = float(sub_mask.mean())
        if cov < min_cov:
            continue

        masked_vals = sub_D[sub_mask]
        sal_sum  = float(masked_vals.sum()) if masked_vals.size else 0.0
        sal_mean = float(masked_vals.mean()) if masked_vals.size else 0.0
        sal_peak = float(sub_D.max()) if sub_D.size else 0.0

        width_cols  = int(c1 - c0)
        height_bins = int(r1 - r0)
        aspect_ft   = (height_bins * fbin) / max(width_cols * tbin, 1e-9)
        center_t    = (c0 + c1) * 0.5 * tbin
        center_f    = (r0 + r1) * 0.5 * fbin
        # compactness ~ perimeter^2 / area; approximate perimeter on mask bbox
        peri = 2.0 * (width_cols + height_bins)
        compact = float(peri * peri / max(area, 1))
        # 95th percentile inside the mask
        sal95 = float(np.percentile(masked_vals, 95)) if masked_vals.size else 0.0

        regs.append(Region(
            bbox=(r0, c0, r1, c1),
            sal_mean=sal_mean, sal_sum=sal_sum, sal_peak=sal_peak,
            area_px=area,
            t_start=c0 * tbin, t_end=c1 * tbin,
            f_low=r0 * fbin,  f_high=r1 * fbin,
            band_bin=band_of_bbox((r0, c0, r1, c1)),
            mask_cov=cov,
            width_cols=width_cols,
            height_bins=height_bins,
            aspect_ft=aspect_ft,
            center_t=center_t,
            center_f=center_f,
            compactness=compact,
            sal95=sal95
        ))

    regs.sort(key=lambda r: (r.sal_sum, r.sal_mean, r.sal_peak), reverse=True)
    return regs

def pick_topk(regs: List[Region], k: int, ensure_band_diversity: bool):
    if not regs:
        return []
    if not ensure_band_diversity or k <= 1:
        out = regs[:k]
    else:
        by_band = {"low": [], "mid": [], "high": []}
        for r in regs:
            by_band[r.band_bin].append(r)
        out = []
        ptr = 0
        bands = [b for b in ["low", "mid", "high"] if by_band[b]]
        while len(out) < k and any(by_band.values()):
            b = bands[ptr % len(bands)]
            if by_band[b]:
                out.append(by_band[b].pop(0))
            ptr += 1
    out.sort(key=lambda r: (r.t_start, r.f_low))
    for i, r in enumerate(out, start=1):
        r.rid = i
    return out

# ---------------------------- Tier 3 helpers (audio/TextGrid/F0) ----------------------------
def load_audio_16k(path: str, target_sr=TARGET_SR):
    try:
        y, sr = sf.read(path, always_2d=False)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, sr

def vad_mask_webrtc(y, sr, frame_ms=20, aggressiveness=2):
    import webrtcvad
    assert sr in [8000, 16000, 32000]
    vad = webrtcvad.Vad(int(aggressiveness))
    frame_len = int(sr * frame_ms / 1000)
    pcm = (np.int16(np.clip(y, -1.0, 1.0) * 32767)).tobytes()
    n_frames = len(y) // frame_len
    speech = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        b0 = i * frame_len * 2
        b1 = b0 + frame_len * 2
        fb = pcm[b0:b1]
        if len(fb) < frame_len * 2:
            break
        speech[i] = vad.is_speech(fb, sr)
    return speech, frame_len

def stft_energy_L2(y, sr, frame_ms=20):
    hop = int(sr * frame_ms / 1000)
    win = hop
    n_fft = max(512, 1 << (win - 1).bit_length())
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, center=False, window="hann")
    mag = np.abs(S)
    energy = np.linalg.norm(mag, ord=2, axis=0)
    return energy, hop, S.shape[1]

def normalize_log_energy(energy, speech_mask, eps=1e-8):
    speech_energy = energy[speech_mask]
    ref = (np.max(speech_energy) if speech_energy.size else np.max(energy)) + eps
    energy_db = 20.0 * np.log10(np.maximum(energy, eps) / ref)
    if speech_energy.size:
        min_db = 20.0 * np.log10(np.maximum(np.percentile(speech_energy, 5.0), eps) / ref)
    else:
        min_db = np.min(energy_db)
    en = (energy_db - min_db) / (0.0 - min_db + eps)
    return np.clip(en, 0.0, 1.0)

def f0_pyin(y, sr, fmin=75.0, fmax=500.0, hop=None):
    hop = 160 if hop is None else hop
    f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=2048, hop_length=hop, center=False)
    t = np.arange(len(f0)) * hop / sr
    f0 = f0.astype(float)
    f0[(~np.isfinite(f0)) | (f0 <= 0.0)] = np.nan
    return t, f0

def f0_praat(y, sr, fmin=75.0, fmax=500.0,
             silence_threshold=0.03, voicing_threshold=0.45,
             octave_cost=0.01, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14):
    if not HAVE_PRAAT:
        return None, None
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        pitch = snd.to_pitch_cc(
            time_step=None, pitch_floor=fmin,
            max_number_of_candidates=15,
            very_accurate=False,
            silence_threshold=silence_threshold,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost, octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
            pitch_ceiling=fmax,
        )
    except TypeError:
        pitch = snd.to_pitch_cc(time_step=None, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"].astype(float)
    t = pitch.xmin + np.arange(f0.shape[0]) * pitch.dx
    f0[(~np.isfinite(f0)) | (f0 <= 0.0)] = np.nan
    return t, f0

def phone_class(label: str) -> Optional[str]:
    lab = (label or "").upper()
    if lab in SIL_TOKENS or lab == "":
        return None
    base = "".join([c for c in lab if c.isalpha()])
    stress = "".join([c for c in lab if c.isdigit()])
    if base in VOWELS:
        if stress == "1": return "V1"
        if stress == "2": return "V2"
        return "V0"
    return "C"

def load_textgrid_intervals(tg_path: str, phones_tier="phones", words_tier="words"):
    try:
        import tgt
    except Exception:
        return [], []
    if not tg_path or not Path(tg_path).exists():
        return [], []
    tg = tgt.io.read_textgrid(tg_path)
    phs, wrds = [], []
    try:
        tier = tg.get_tier_by_name(phones_tier)
        phs = [(it.start_time, it.end_time, it.text.strip()) for it in tier]
    except Exception:
        phs = []
    try:
        tier = tg.get_tier_by_name(words_tier)
        wrds = [(it.start_time, it.end_time, it.text.strip()) for it in tier]
    except Exception:
        wrds = []
    return phs, wrds

def interval_overlap_frac(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    base = max(1e-9, a1 - a0)
    return inter / base

# ---------------------------- Tier 3 feature computation per region ----------------------------
def enrich_tier3_features(regs: List[Region],
                          y: np.ndarray, sr: int,
                          textgrid_path: Optional[str],
                          phones_tier: str, words_tier: str,
                          f0_method: str = "pyin",
                          f0_floor: float = 75.0, f0_ceiling: float = 500.0):
    # VAD + LS/MS/HS
    frame_ms = 20
    speech_mask, fsz = vad_mask_webrtc(y, sr, frame_ms=frame_ms, aggressiveness=2)
    energy, hop_e, _ = stft_energy_L2(y, sr, frame_ms=frame_ms)
    n = min(len(energy), len(speech_mask))
    energy, speech_mask = energy[:n], speech_mask[:n]
    en = normalize_log_energy(energy, speech_mask)
    t1, t2 = 1.0/3.0, 2.0/3.0
    ls_idx = (en < t1) & speech_mask
    ms_idx = (en >= t1) & (en < t2) & speech_mask
    hs_idx = (en >= t2) & speech_mask

    # F0
    if f0_method.lower() == "praat" and HAVE_PRAAT:
        t_f0, f0 = f0_praat(y, sr, fmin=f0_floor, fmax=f0_ceiling)
        if t_f0 is None:
            t_f0, f0 = f0_pyin(y, sr, fmin=f0_floor, fmax=f0_ceiling, hop=int(sr*frame_ms/1000))
    else:
        t_f0, f0 = f0_pyin(y, sr, fmin=f0_floor, fmax=f0_ceiling, hop=int(sr*frame_ms/1000))

    # TextGrid
    phones, words = load_textgrid_intervals(textgrid_path, phones_tier=phones_tier, words_tier=words_tier)

    # helpers to map time→frame idx
    def col_range_for_region(r: Region):
        # map region t_start/t_end (seconds) into VAD/energy frame indices
        s = max(0, int(math.floor(r.t_start / (frame_ms/1000.0))))
        e = max(s+1, int(math.ceil(r.t_end / (frame_ms/1000.0))))
        s = min(s, len(speech_mask)); e = min(e, len(speech_mask))
        return s, e

    def f0_mask_for_region(r: Region):
        if t_f0 is None or f0 is None or len(t_f0) == 0:
            return np.array([], dtype=bool), np.array([])
        mask = (t_f0 >= r.t_start) & (t_f0 <= r.t_end)
        return mask, f0[mask]

    for r in regs:
        s, e = col_range_for_region(r)
        # speech fractions
        denom = max(e - s, 1)
        r.speech_frac = float(np.count_nonzero(speech_mask[s:e])) / denom
        r.ls_frac = float(np.count_nonzero(ls_idx[s:e])) / denom
        r.ms_frac = float(np.count_nonzero(ms_idx[s:e])) / denom
        r.hs_frac = float(np.count_nonzero(hs_idx[s:e])) / denom

        # F0 stats
        fmask, fvals = f0_mask_for_region(r)
        if fvals.size:
            r.f0_mean = float(np.nanmean(fvals))
            r.f0_median = float(np.nanmedian(fvals))
            r.f0_std = float(np.nanstd(fvals))
            r.f0_min = float(np.nanmin(fvals))
            r.f0_max = float(np.nanmax(fvals))
        else:
            r.f0_mean = r.f0_median = r.f0_std = r.f0_min = r.f0_max = 0.0

        # Phones: majority class + entropy (fallback to "none")
        ph_counts: Dict[str, float] = {}
        for s0, s1, lab in phones:
            frac = interval_overlap_frac(r.t_start, r.t_end, s0, s1)
            if frac <= 0.0:
                continue
            cls = phone_class(lab)
            if cls is None:
                continue
            ph_counts[cls] = ph_counts.get(cls, 0.0) + frac
        if ph_counts:
            major = max(ph_counts.items(), key=lambda kv: kv[1])[0]
            total = sum(ph_counts.values())
            probs = [v/total for v in ph_counts.values()]
            entropy = -sum(p*math.log(p+1e-12) for p in probs)
            r.phone_major = major
            r.phone_entropy = float(entropy)
        else:
            r.phone_major = "none"
            r.phone_entropy = 0.0

        # Words: how many with any overlap
        r.words_covered = 0
        for s0, s1, w in words:
            if interval_overlap_frac(r.t_start, r.t_end, s0, s1) > 0.0:
                r.words_covered += 1

# ---------------------------- drawing ----------------------------
def draw_annotated_png(out_path, G_Mb, mask, regs):
    frames = mask.shape[1]
    ext = _extent(frames)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if G_Mb is not None:
        disp = _stretch01(G_Mb.copy())
        ax.imshow(disp, extent=ext, aspect="auto", origin="lower",
                  cmap="gray", interpolation="nearest", zorder=1)
    ax.imshow(np.ma.masked_where(mask == 0, mask.astype(float)),
              extent=ext, aspect="auto", origin="lower",
              cmap="Reds", interpolation="nearest", alpha=0.9, vmin=0, vmax=1, zorder=2)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Smoothed Annotation, selected regions")

    for r in regs:
        ax.add_patch(plt.Rectangle(
            (r.t_start, r.f_low),
            r.t_end - r.t_start,
            r.f_high - r.f_low,
            fill=False, linewidth=2.0, edgecolor="#1f77b4", zorder=5
        ))
        ax.text(
            r.t_start + 0.01 * max(r.t_end - r.t_start, 1e-6),
            r.f_high - 0.05 * max(r.f_high - r.f_low, 1e-6),
            f"{r.rid}",
            color="#1f77b4", fontsize=12, ha="left", va="top", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, lw=0.0)
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------------- paths for Tier 3 resources ----------------------------
def infer_bona_id_from_spoof(stem: str) -> Optional[str]:
    m = re.search(r"(LA_[TDE]_\d{7})", stem)
    return m.group(1) if m else None

def locate_spoof_wav(spoof_audio_root: str, spoof_stem: str) -> Optional[str]:
    p = Path(spoof_audio_root) / f"{spoof_stem}.wav"
    return str(p) if p.exists() else None

def locate_textgrid(textgrid_root: str, bona_id: str) -> Optional[str]:
    p = Path(textgrid_root) / bona_id / f"{bona_id}.TextGrid"
    return str(p) if p.exists() else None

def locate_transcript(transcript_root: str, bona_id: str) -> Optional[str]:
    p = Path(transcript_root) / bona_id / f"{bona_id}.txt"
    return str(p) if p.exists() else None  # not used in features, but reserved

# ---------------------------- single/batch entry points ----------------------------
def run_pair_precomputed(pair_dir: str, spoof_stem: str,
                         out_dir: str,
                         # morphology
                         close_iters: int = 1,
                         open_iters: int = 0,
                         # region filters
                         min_area_px: int = 60,
                         min_dur_s: float = 0.03,
                         min_bw_hz: float = 120.0,
                         min_cov: float = 0.20,
                         # selection
                         top_k: int = 4,
                         band_diversity: bool = True,
                         # Tier 3 resources
                         spoof_audio_root: str = "/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav",
                         textgrid_root: str = "/home/opc/aligned_textgrids",
                         transcript_root: str = "/home/opc/tmp_mfa_corpus",
                         phones_tier: str = "phones",
                         words_tier: str = "words",
                         f0_method: str = "pyin",
                         f0_floor: float = 75.0,
                         f0_ceiling: float = 500.0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    G_Mb, D, mask = load_precomputed(pair_dir, spoof_stem)

    # light morphology to form clumps  (same defaults)
    mask2 = binary_closing(mask, iterations=close_iters)
    if open_iters > 0:
        mask2 = binary_opening(mask2, iterations=open_iters)

    regs_all = regions_from_mask(D, mask2, min_area_px, min_dur_s, min_bw_hz, min_cov)
    regs_sel = pick_topk(regs_all, k=top_k, ensure_band_diversity=band_diversity)

    # ---- Tier 3 enrichment from audio/TextGrid/F0 (always produce values) ----
    bona_id = infer_bona_id_from_spoof(spoof_stem) or ""
    wav_path = locate_spoof_wav(spoof_audio_root, spoof_stem)
    tg_path  = locate_textgrid(textgrid_root, bona_id) if bona_id else None
    # transcript_path = locate_transcript(transcript_root, bona_id)  # reserved

    if wav_path and Path(wav_path).exists():
        y, sr = load_audio_16k(wav_path, TARGET_SR)
        enrich_tier3_features(
            regs_sel, y, sr,
            textgrid_path=tg_path,
            phones_tier=phones_tier, words_tier=words_tier,
            f0_method=f0_method, f0_floor=f0_floor, f0_ceiling=f0_ceiling
        )
    else:
        # fill with safe defaults (already initialized)
        pass

    # outputs
    csv_path = out / "regions.csv"
    field_order = [
        # original
        "rid","bbox","sal_mean","sal_sum","sal_peak","area_px",
        "t_start","t_end","f_low","f_high","band_bin","mask_cov",
        # Tier2
        "width_cols","height_bins","aspect_ft","center_t","center_f",
        "compactness","sal95",
        # Tier3
        "speech_frac","ls_frac","ms_frac","hs_frac",
        "f0_mean","f0_median","f0_std","f0_min","f0_max",
        "phone_major","phone_entropy","words_covered",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in regs_sel:
            row = asdict(r)
            # string-ify bbox for readability
            row["bbox"] = str(tuple(row["bbox"]))
            # ensure all fields exist
            for k in field_order:
                if k not in row:
                    row[k] = 0 if isinstance(k, (int,float)) else ""
            w.writerow({k: row[k] for k in field_order})

    json_path = out / "regions.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in regs_sel], f, indent=2)

    png_path = out / "annotated.png"
    draw_annotated_png(str(png_path), G_Mb, mask2, regs_sel)

    np.save(out / "mask.npy", mask2.astype(np.uint8))
    np.save(out / "density.npy", D.astype(np.float32))
    return {"csv": str(csv_path), "json": str(json_path), "png": str(png_path),
            "n_regions": len(regs_sel)}

def infer_spoof_stems_in_pair_dir(pair_dir: Path):
    stems = []
    for p in pair_dir.glob("mask95_smoothed__*.npy"):
        m = re.match(r"mask95_smoothed__([A-Za-z0-9_]+)\.npy", p.name)
        if m: stems.append(m.group(1))
    return stems

def run_batch(precomp_root: str, only_vocoder: str, out_root: str, **kwargs):
    precomp_root = Path(precomp_root)
    out_root = Path(out_root)
    count = 0
    for bona_dir in precomp_root.iterdir():
        if not bona_dir.is_dir(): continue
        voc_dir = bona_dir / only_vocoder
        if not voc_dir.exists(): continue
        for stem in infer_spoof_stems_in_pair_dir(voc_dir):
            out_dir = out_root / bona_dir.name / only_vocoder / stem
            run_pair_precomputed(str(voc_dir), stem, str(out_dir), **kwargs)
            count += 1
    return count

# ---------------------------- CLI ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", help="Folder with precomputed npys")
    ap.add_argument("--spoof-stem", help="Spoof stem like hifigan_LA_D_1024892")
    ap.add_argument("--out", help="Output folder for a single pair")

    # batch mode
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--precomp-root", help="Root containing <bona_id>/<vocoder>")
    ap.add_argument("--only-vocoder", help="Pick a vocoder (e.g. hifigan)")
    ap.add_argument("--out-root", help="Output root for batch")

    # region knobs (same defaults as older script)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--band-diversity", action="store_true", default=True)
    ap.add_argument("--close-iters", type=int, default=1)
    ap.add_argument("--open-iters", type=int, default=0)
    ap.add_argument("--min-area-px", type=int, default=60)
    ap.add_argument("--min-dur-s", type=float, default=0.03)
    ap.add_argument("--min-bw-hz", type=float, default=120.0)
    ap.add_argument("--min-cov", type=float, default=0.20)

    # Tier 3 roots/tiers/f0
    ap.add_argument("--spoof-audio-root", default="/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
    ap.add_argument("--textgrid-root", default="/home/opc/aligned_textgrids")
    ap.add_argument("--transcript-root", default="/home/opc/tmp_mfa_corpus")
    ap.add_argument("--phones-textgrid", default="phones")
    ap.add_argument("--words-textgrid", default="words")
    ap.add_argument("--f0-method", default="pyin", choices=["pyin","praat"])
    ap.add_argument("--f0-floor", type=float, default=75.0)
    ap.add_argument("--f0-ceiling", type=float, default=500.0)

    args = ap.parse_args()

    kwargs = dict(
        close_iters=args.close_iters,
        open_iters=args.open_iters,
        min_area_px=args.min_area_px,
        min_dur_s=args.min_dur_s,
        min_bw_hz=args.min_bw_hz,
        min_cov=args.min_cov,
        top_k=args.top_k,
        band_diversity=args.band_diversity,
        spoof_audio_root=args.spoof_audio_root,
        textgrid_root=args.textgrid_root,
        transcript_root=args.transcript_root,
        phones_tier=args.phones_textgrid,
        words_tier=args.words_textgrid,
        f0_method=args.f0_method,
        f0_floor=args.f0_floor,
        f0_ceiling=args.f0_ceiling,
    )

    if args.batch:
        if not args.precomp_root or not args.only_vocoder or not args.out_root:
            ap.error("with --batch, provide --precomp-root, --only-vocoder, --out-root")
        n = run_batch(args.precomp_root, args.only_vocoder, args.out_root, **kwargs)
        print(json.dumps({"processed_pairs": n}, indent=2))
        return

    # single
    if not args.pair_dir or not args.spoof_stem or not args.out:
        ap.error("for single run, provide --pair-dir, --spoof-stem, --out")
    res = run_pair_precomputed(args.pair_dir, args.spoof_stem, args.out, **kwargs)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()