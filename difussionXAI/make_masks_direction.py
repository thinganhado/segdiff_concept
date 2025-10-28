import re
import csv
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import shutil
import numpy as np
import soundfile as sf
import librosa
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter1d,
    binary_opening,
    binary_closing,
    label,
    find_objects,
)
from tqdm import tqdm

# Optional Praat TextGrid parser
try:
    import tgt
    HAVE_TGT = True
except Exception:
    HAVE_TGT = False

# Optional Praat, Parselmouth, for Picture rendering
try:
    import parselmouth as pm
    HAVE_PM = True
except Exception:
    HAVE_PM = False

# ------------- paths -------------
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_PER_AUDIO_ROOT = Path("/home/opc/datasets/asvspoof2019/LA/per_audio_folders")
PROTOCOL_TXT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")
OUT_DIR = Path("./mask_outputs_vocv4")

# ------------- STFT and smoothing -------------
target_sr = 16000
n_fft = 1024
hop = 256
win_length = 1024
center = True

ksize_t, ksize_f = 3, 11
var_t, var_f = 3.0, 5.0
sigma_t, sigma_f = np.sqrt(var_t), np.sqrt(var_f)
# per axis truncates so window length equals requested sizes
trunc_t = ((ksize_t - 1) / 2.0) / max(sigma_t, 1e-6)
trunc_f = ((ksize_f - 1) / 2.0) / max(sigma_f, 1e-6)

quantile_tau = 0.95
eps = 1e-8
use_dtw = False

# ------------ helpers ------------
def load_mono_resample(path, sr):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return y

def mag_spec(y):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center))

def smooth_2d(S):
    """Exact separable Gaussian, time then frequency, sizes [3, 11], variances [3, 5]."""
    St = gaussian_filter1d(S, sigma=sigma_t, axis=1, truncate=trunc_t)
    Sf = gaussian_filter1d(St, sigma=sigma_f, axis=0, truncate=trunc_f)
    return Sf

def dtw_time_align_same_length(A, B):
    A_n = librosa.util.normalize(A, axis=0)
    B_n = librosa.util.normalize(B, axis=0)
    _, wp = librosa.sequence.dtw(X=A_n, Y=B_n, metric="cosine")
    path = wp[::-1]
    a_idx = np.fromiter((i for i, _ in path), dtype=int)
    b_idx = np.fromiter((j for _, j in path), dtype=int)
    return A[:, a_idx], B[:, b_idx]

# parse lines like, LA_0079 hifigan_LA_T_1703395 , hifigan spoof
line_rx = re.compile(r"^(LA_\d{4})\s+([A-Za-z0-9_-]+)\s+-\s+([A-Za-z0-9_-]+)\s+spoof")

def bona_id_from_spoof(stem: str) -> Optional[str]:
    m = re.search(r"(LA_[TDE]_\d{7})", stem)
    return m.group(1) if m else None

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
    print(
        f"[info] Indexed bona folders at {root}, total {len(idx)} items"
        f" [T {sum(k.startswith('LA_T_') for k in idx)},"
        f" D {sum(k.startswith('LA_D_') for k in idx)},"
        f" E {sum(k.startswith('LA_E_') for k in idx)}]"
    )
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

def iter_pairs(protocol_path, vocoder_filter=None, bona_index: Dict[str, Path] = None):
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
                print(f"Skip, spoof audio not found, {spoof_stem}")
                continue

            bona_id = bona_id_from_spoof(stem=spoof_stem)
            if bona_id is None:
                print(f"Skip, cannot derive bona ID from spoof stem, {spoof_stem}")
                continue

            bf_path = bona_index.get(bona_id)
            if bf_path is None:
                folder = ASV19_PER_AUDIO_ROOT / bona_id
                if folder.exists():
                    print(f"Skip, bona folder exists but audio file missing, {folder}")
                else:
                    print(f"Skip, bona folder not found under {ASV19_PER_AUDIO_ROOT}, {bona_id}")
                continue

            yield spk, vocoder_lc, spoof_stem, sp_path, bona_id, bf_path

# ---------- TextGrid helpers ----------
def find_matching_textgrid(bona_id: str, root: Optional[str]) -> Optional[Path]:
    """
    Prefer <root>/<bona_id>/<bona_id>.TextGrid, your layout.
    Fallback to recursive search if not found.
    """
    if not root:
        return None
    root_path = Path(root)
    if not root_path.exists():
        return None
    direct = root_path / bona_id / f"{bona_id}.TextGrid"
    if direct.exists():
        return direct
    hits = list(root_path.rglob(f"{bona_id}.TextGrid"))
    return hits[0] if hits else None

def load_intervals_from_textgrid(tg_path: Path, tier_name: str) -> List[Tuple[float, float, str]]:
    if not HAVE_TGT:
        return []
    try:
        tg = tgt.io.read_textgrid(str(tg_path))
        tier = tg.get_tier_by_name(tier_name)
        out = []
        for it in tier.intervals:
            lab = (it.text or "").strip()
            out.append((float(it.start_time), float(it.end_time), lab))
        return out
    except Exception:
        return []

def labels_overlapping_interval(
    intervals: List[Tuple[float, float, str]],
    t0: float, t1: float,
    silence_set: set,
    min_overlap_ratio: float = 0.2
):
    """
    Generic overlap helper, returns (labels_list, detailed_spans, any_silence_overlap).
    labels_list excludes silence tokens.
    """
    found = []
    spans = []
    had_sil = False
    for s, e, w in intervals:
        if e <= t0 or s >= t1:
            continue
        ov = max(0.0, min(t1, e) - max(t0, s))
        dur = max(1e-6, e - s)
        if ov / dur >= min_overlap_ratio:
            wl = (w or "").strip()
            if wl.lower() in silence_set or wl == "":
                had_sil = True
            else:
                found.append(wl)
            spans.append({"label": wl, "start": float(s), "end": float(e), "overlap_sec": float(ov)})
    return found, spans, had_sil

# ---------- diff products ----------
def to_db(m):
    db = librosa.amplitude_to_db(m, ref=np.max)
    return np.clip(db, -60.0, 0.0)

def compute_products(y_bona, y_spoof):
    # magnitude spectrograms for mask math
    Mb = mag_spec(y_bona)   # |STFT|
    Ms = mag_spec(y_spoof)

    # align lengths or apply DTW if enabled
    if use_dtw:
        Mb, Ms = dtw_time_align_same_length(Mb, Ms)
    else:
        T = min(Mb.shape[1], Ms.shape[1])
        Mb, Ms = Mb[:, :T], Ms[:, :T]

    # smoothing G of size (3,11), var (3,5)
    G_Mb_lin = smooth_2d(Mb)
    G_Ms_lin = smooth_2d(Ms)

    # Eq. (2), |G(Ms) - G(Mb)| over |G(Mb)|
    signed = G_Ms_lin - G_Mb_lin
    diff_s = np.abs(signed)
    norm_s = diff_s / (G_Mb_lin + eps)
    flat_s = norm_s[np.isfinite(norm_s)].ravel()
    tau_s  = np.quantile(flat_s, quantile_tau) if flat_s.size else 0.0

    # binary mask and directional masks
    mask_s = norm_s > tau_s
    add_mask  = mask_s & (signed > 0)   # spoof has more energy
    miss_mask = mask_s & (signed < 0)   # spoof has less energy

    # optional unsmoothed branch
    diff_r = np.abs(Ms - Mb)
    norm_r = diff_r / (Mb + eps)
    flat_r = norm_r[np.isfinite(norm_r)].ravel()
    tau_r  = np.quantile(flat_r, quantile_tau) if flat_r.size else 0.0
    mask_r = norm_r > tau_r

    # dB for display
    G_Mb_db = to_db(G_Mb_lin)
    G_Ms_db = to_db(G_Ms_lin)
    Mb_db   = to_db(Mb)
    Ms_db   = to_db(Ms)

    return (G_Mb_db, G_Ms_db, mask_s, tau_s, mask_r, tau_r,
            Mb_db, Ms_db, add_mask, miss_mask, norm_s, G_Mb_lin, G_Ms_lin)

# ---------- paper style visualization ----------
def _extent(frames, sr, hop, n_fft):
    t_max = frames * hop / sr
    f_max = sr / 2.0
    return [0.0, t_max, 0.0, f_max]

def paper_gray_from_db(db_img,
                       lo_pct=5.0, hi_pct=95.0,
                       out_lo=0.62, out_hi=0.96,
                       gamma=1.0):
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z

# constant color maps, transparent where 0
RED_MASK_CMAP  = ListedColormap([[0, 0, 0, 0], [0.80, 0.00, 0.00, 1.0]])  # additions
BLUE_MASK_CMAP = ListedColormap([[0, 0, 0, 0], [0.00, 0.40, 0.95, 1.0]])  # misses

def _rows_for_fmax(arr_2d: np.ndarray, fmax_hz: float) -> int:
    """Map a desired display fmax to the number of spectrogram rows to keep."""
    F = arr_2d.shape[0]
    hz_per = (target_sr / 2.0) / max(1, F - 1)
    return max(1, int(round(fmax_hz / hz_per)))

def save_annotation_overlay(bg_db, add_mask, miss_mask, out_path, title, fmax_display=None):
    """Save a single panel with optional add or miss overlays, clamped to fmax_display in Hz."""
    if fmax_display is None:
        fmax_display = target_sr / 2.0

    max_row = _rows_for_fmax(bg_db, fmax_display)
    bg_db = bg_db[:max_row, :]
    add_mask = add_mask[:max_row, :]
    miss_mask = miss_mask[:max_row, :]

    frames = bg_db.shape[1]
    ext = [0.0, frames * hop / target_sr, 0.0, fmax_display]
    bg_disp = paper_gray_from_db(bg_db)

    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(bg_disp, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    # draw misses first, adds on top
    plt.imshow(np.ma.masked_where(~miss_mask, miss_mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    plt.imshow(np.ma.masked_where(~add_mask, add_mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()

def save_quicklook_paper_style(out_path, G_Mb_db, G_Ms_db, add_mask, miss_mask, fmax_display=None):
    """Three panel figure, real, fake, overlay, clamped to fmax_display in Hz."""
    if fmax_display is None:
        fmax_display = target_sr / 2.0

    max_row = _rows_for_fmax(G_Mb_db, fmax_display)
    G_Mb_db = G_Mb_db[:max_row, :]
    G_Ms_db = G_Ms_db[:max_row, :]
    add_mask = add_mask[:max_row, :]
    miss_mask = miss_mask[:max_row, :]

    frames = G_Mb_db.shape[1]
    ext = [0.0, frames * hop / target_sr, 0.0, fmax_display]
    disp_b = paper_gray_from_db(G_Mb_db)
    disp_s = paper_gray_from_db(G_Ms_db)

    plt.figure(figsize=(12, 3.8))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(disp_b, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax1.set_title("Smoothed Real"); ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Frequency [Hz]")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(disp_s, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title("Smoothed Fake"); ax2.set_xlabel("Time [s]"); ax2.set_ylabel("")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(disp_b, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax3.imshow(np.ma.masked_where(~miss_mask, miss_mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=BLUE_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax3.imshow(np.ma.masked_where(~add_mask, add_mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax3.set_title("Smoothed Annotation, 95 percent, red add, blue miss"); ax3.set_xlabel("Time [s]"); ax3.set_ylabel("")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()

# ---------- crops pipeline ----------
def hz_per_bin(n_freq_bins: int) -> float:
    return (target_sr / 2.0) / max(1, n_freq_bins - 1)

def frames_to_seconds(frames: int) -> float:
    return frames * hop / target_sr

def make_crops(
    out_dir: Path,
    pair_id: str,
    G_Mb_db: np.ndarray,
    G_Ms_db: np.ndarray,
    add_mask: np.ndarray,
    miss_mask: np.ndarray,
    norm_s: np.ndarray,
    G_Mb_lin: np.ndarray,
    # TextGrid mapping inputs
    words_intervals: Optional[List[Tuple[float, float, str]]] = None,
    phones_intervals: Optional[List[Tuple[float, float, str]]] = None,
    silence_set: Optional[set] = None,
    min_word_overlap_ratio: float = 0.2,
    min_phone_overlap_ratio: float = 0.2,
    # params
    crops_k: int = 8,
    gate_time_pct: float = 25.0,
    fmin_hz: float = 80.0,
    fmax_hz: float = 6000.0,
    grow_t_ms: float = 32.0,
    grow_f_hz: float = 300.0,
    min_area: int = 30,
    save_overlays: bool = False,
) -> List[dict]:
    """
    Create top K crops from the smoothed, normalized diff using,
    time and band gating on REAL energy, morphological open and close,
    connected components ranked by sum of norm_s.
    Also maps crops to word and phone spans from TextGrid if provided.
    """
    H, W = G_Mb_db.shape
    # 1) gating by real energy, time and freq band
    E_t = G_Mb_lin.sum(axis=0)
    thr = np.percentile(E_t, gate_time_pct)
    t_keep = E_t > thr
    f_axis = np.linspace(0.0, target_sr/2.0, H)
    f_keep = (f_axis >= fmin_hz) & (f_axis <= fmax_hz)
    keep = f_keep[:, None] & t_keep[None, :]

    mask = (norm_s > np.quantile(norm_s[np.isfinite(norm_s)], 0.95)) & keep

    # 2) clean up blobs
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((3, 3)))

    # 3) connected components
    lab, nlab = label(mask, structure=np.ones((3, 3)))
    if nlab == 0:
        return []

    slices = find_objects(lab)
    regions: List[Tuple[float, int, int, Tuple[slice, slice]]] = []
    for lab_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        region_mask = (lab[slc] == lab_id)
        area = int(region_mask.sum())
        if area < min_area:
            continue
        score = float(norm_s[slc][region_mask].sum())  # integral of continuous diff
        regions.append((score, area, lab_id, slc))

    if not regions:
        return []

    regions.sort(key=lambda x: x[0], reverse=True)
    top = regions[:crops_k]

    # 4) grow slices for context
    grow_t_frames = max(0, int(round((grow_t_ms / 1000.0) * target_sr / hop)))
    grow_f_bins = max(0, int(round(grow_f_hz / hz_per_bin(H))))

    def grow_slice(slc: Tuple[slice, slice]) -> Tuple[slice, slice]:
        fs, ts = slc
        f0, f1 = fs.start, fs.stop
        t0, t1 = ts.start, ts.stop
        f0 = max(0, f0 - grow_f_bins); f1 = min(H, f1 + grow_f_bins)
        t0 = max(0, t0 - grow_t_frames); t1 = min(W, t1 + grow_t_frames)
        return (slice(f0, f1), slice(t0, t1))

    # 5) export crops
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    frames_to_s = lambda fr: fr * hop / target_sr
    hz_bin = hz_per_bin(H)
    sil_set = (silence_set or set())

    for i, (score, area, lab_id, slc) in enumerate(top, start=1):
        g = grow_slice(slc)
        fs, ts = g
        # crops in dB space for consistent display
        real_crop = paper_gray_from_db(G_Mb_db[fs, ts])
        fake_crop = paper_gray_from_db(G_Ms_db[fs, ts])

        # times in seconds for overlaps
        t0_s = frames_to_s(ts.start)
        t1_s = frames_to_s(ts.stop)

        # save images
        ext = [t0_s, t1_s, fs.start * hz_bin, fs.stop * hz_bin]

        def save_panel(img, path, title):
            plt.figure(figsize=(4.2, 3.4))
            plt.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]],
                       aspect="auto", origin="lower",
                       cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            plt.title(title); plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]")
            plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

        crop_id = f"crop_{i:02d}"
        save_panel(real_crop, crops_dir / f"{crop_id}__A_real.png", "A Real")
        save_panel(fake_crop, crops_dir / f"{crop_id}__B_fake.png", "B Fake")

        if save_overlays:
            add_c = add_mask[fs, ts]
            miss_c = miss_mask[fs, ts]
            plt.figure(figsize=(4.2, 3.4))
            plt.imshow(real_crop, extent=[ext[0], ext[1], ext[2], ext[3]],
                       aspect="auto", origin="lower",
                       cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            plt.imshow(np.ma.masked_where(~miss_c, miss_c.astype(float)),
                       extent=[ext[0], ext[1], ext[2], ext[3]],
                       aspect="auto", origin="lower", cmap=BLUE_MASK_CMAP, vmin=0, vmax=1)
            plt.imshow(np.ma.masked_where(~add_c, add_c.astype(float)),
                       extent=[ext[0], ext[1], ext[2], ext[3]],
                       aspect="auto", origin="lower", cmap=RED_MASK_CMAP, vmin=0, vmax=1)
            plt.title("Overlay add=red miss=blue")
            plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]")
            plt.tight_layout(); plt.savefig(crops_dir / f"{crop_id}__overlay.png", dpi=130); plt.close()

        # word mapping
        words_list, word_spans, had_sil_w = [], [], False
        words_source = "energy_fallback"
        if words_intervals:
            words_list, word_spans, had_sil_w = labels_overlapping_interval(
                words_intervals, t0_s, t1_s, sil_set, min_overlap_ratio=min_word_overlap_ratio
            )
            words_source = "textgrid"

        # phones mapping
        phones_list, phone_spans, had_sil_p = [], [], False
        phones_source = None
        if phones_intervals:
            phones_list, phone_spans, had_sil_p = labels_overlapping_interval(
                phones_intervals, t0_s, t1_s, sil_set, min_overlap_ratio=min_phone_overlap_ratio
            )
            phones_source = "textgrid"

        # non speech fallback, unchanged
        non_speech = False
        if words_source == "energy_fallback":
            E_crop = G_Mb_lin.sum(axis=0)[ts.start:ts.stop]
            thrE = np.percentile(G_Mb_lin.sum(axis=0), 25.0)
            non_speech = bool(np.mean(E_crop) < thrE)
        else:
            non_speech = (len(words_list) == 0) and had_sil_w

        manifest.append({
            "pair_id": pair_id,
            "crop_id": crop_id,
            "score_sum_norm": float(score),
            "area_bins": int(area),
            "box_index": [int(ts.start), int(ts.stop), int(fs.start), int(fs.stop)],  # [t0,t1,f0,f1] indices
            "box_tfr": [t0_s, t1_s, fs.start * hz_bin, fs.stop * hz_bin],            # [t0,t1,f0,f1] sec and Hz
            "A_path": str(crops_dir / f"{crop_id}__A_real.png"),
            "B_path": str(crops_dir / f"{crop_id}__B_fake.png"),
            "overlay_path": str(crops_dir / f"{crop_id}__overlay.png") if save_overlays else None,
            "words": words_list if words_list else None,
            "word_spans": word_spans if word_spans else None,
            "words_source": words_source,
            "phones": phones_list if phones_list else None,
            "phone_spans": phone_spans if phone_spans else None,
            "phones_source": phones_source,
            "non_speech": bool(non_speech)
        })

    with open(crops_dir / "crops_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

# ---------- Praat based crop rendering, Picture ----------
# Matplotlib fallback for safety, used only if Praat draw fails
def _praat_like_spec(y, sr, t0, t1, fmax, win_ms, db_range, preemph_from=None):
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    seg = y[i0:i1].astype(np.float32, copy=False)
    win = int(round(sr * win_ms / 1000.0))
    n_fft_local = 1
    while n_fft_local < win:
        n_fft_local <<= 1
    hop_local = max(1, win // 2)
    S = np.abs(librosa.stft(seg, n_fft=n_fft_local, hop_length=hop_local,
                            win_length=win, center=False, window="hann")) ** 2
    freqs = np.linspace(0.0, sr / 2.0, S.shape[0])
    hi = np.searchsorted(freqs, fmax, side="right")
    S = S[:hi, :]
    SdB = librosa.power_to_db(S, ref=np.max)
    SdB = np.clip(SdB, -db_range, 0.0)
    dur = max(1e-9, (i1 - i0) / sr)
    extent = [0.0, dur, 0.0, fmax]
    return SdB, extent

def _praat_picture_clear():
    pm.praat.call("Erase all")

def _praat_picture_save_png(path: Path):
    pm.praat.call("Write to PNG file...", str(path))

def _paint_spectrogram_praat(seg_sound: "pm.Sound",
                             fmax: float, win_ms: float,
                             db_range: float, preemph_from: float):
    # Sound to Spectrogram in Praat, window length in seconds
    spec = pm.praat.call(seg_sound, "To Spectrogram", float(win_ms) / 1000.0, float(fmax))
    # Spectrogram, Paint, tmin tmax fmin fmax dynamicRange preEmphasisFrom garnish
    pm.praat.call(spec, "Paint", 0.0, 0.0, 0.0, float(fmax), float(db_range), float(preemph_from), "yes")

def _draw_pitch_praat(seg_sound: "pm.Sound",
                      f0_floor: float, f0_ceiling: float, f0_axis_max: float):
    pitch = pm.praat.call(seg_sound, "To Pitch", 0.0, float(f0_floor), float(f0_ceiling))
    pm.praat.call(pitch, "Draw", 0.0, 0.0, 0.0, float(f0_axis_max), "yes")

def _time_crop_sound(y: np.ndarray, sr: int, t0: float, t1: float) -> "pm.Sound":
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(y), int(round(t1 * sr)))
    seg = y[i0:i1].astype(np.float32, copy=False)
    return pm.Sound(seg, sampling_frequency=sr)

def render_praat_picture_crops(
    y_real: np.ndarray, y_fake: np.ndarray, sr: int,
    crops_dir: Path, manifest: list,
    add_mask_full: np.ndarray, miss_mask_full: np.ndarray,
    fmax: float, win_ms: float, db_range: float, preemph_from: float,
    draw_pitch: bool, f0_floor: float, f0_ceiling: float, f0_axis_max: float,
    overlay_on_praat: bool
):
    if not HAVE_PM:
        raise RuntimeError("Parselmouth, Praat, is not available. Install with, pip install praat-parselmouth")

    import matplotlib.image as mpimg

    def time_to_frame(t):  # column index in your STFT and masks, same hop
        return int(round(t * target_sr / hop))

    for entry in manifest:
        crop_id = entry["crop_id"]
        t0, t1 = float(entry["box_tfr"][0]), float(entry["box_tfr"][1])
        dur = max(1e-6, t1 - t0)
        extent = [0.0, dur, 0.0, fmax]

        # REAL, A
        try:
            sndA = _time_crop_sound(y_real, sr, t0, t1)
            _praat_picture_clear()
            _paint_spectrogram_praat(sndA, fmax, win_ms, db_range, preemph_from)
            if draw_pitch:
                try:
                    _draw_pitch_praat(sndA, f0_floor, f0_ceiling, f0_axis_max)
                except Exception:
                    pass
            outA = crops_dir / f"{crop_id}__A_praat.png"
            _praat_picture_save_png(outA)
            entry["praat_A_path"] = str(outA)
        except Exception:
            SdB, ext = _praat_like_spec(y_real, sr, t0, t1, fmax, win_ms, db_range)
            fig, ax = plt.subplots(figsize=(5.0, 3.6))
            ax.imshow(SdB, extent=ext, origin="lower", aspect="auto", cmap="gray", vmin=-db_range, vmax=0)
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]"); ax.set_ylim(0, fmax)
            outA = crops_dir / f"{crop_id}__A_praat.png"
            fig.tight_layout(); fig.savefig(outA, dpi=130); plt.close(fig)
            entry["praat_A_path"] = str(outA)

        # FAKE, B
        try:
            sndB = _time_crop_sound(y_fake, sr, t0, t1)
            _praat_picture_clear()
            _paint_spectrogram_praat(sndB, fmax, win_ms, db_range, preemph_from)
            if draw_pitch:
                try:
                    _draw_pitch_praat(sndB, f0_floor, f0_ceiling, f0_axis_max)
                except Exception:
                    pass
            outB = crops_dir / f"{crop_id}__B_praat.png"
            _praat_picture_save_png(outB)
            entry["praat_B_path"] = str(outB)
        except Exception:
            SdB, ext = _praat_like_spec(y_fake, sr, t0, t1, fmax, win_ms, db_range)
            fig, ax = plt.subplots(figsize=(5.0, 3.6))
            ax.imshow(SdB, extent=ext, origin="lower", aspect="auto", cmap="gray", vmin=-db_range, vmax=0)
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]"); ax.set_ylim(0, fmax)
            outB = crops_dir / f"{crop_id}__B_praat.png"
            fig.tight_layout(); fig.savefig(outB, dpi=130); plt.close(fig)
            entry["praat_B_path"] = str(outB)

        # optional, overlay add and miss masks onto Praat PNGs
        if overlay_on_praat:
            try:
                dur = max(1e-6, t1 - t0)
                ext = [0.0, dur, 0.0, fmax]
                c0, c1 = time_to_frame(t0), time_to_frame(t1)
                max_row = int(round(fmax / hz_per_bin(add_mask_full.shape[0])))
                add_c = add_mask_full[:max_row, c0:c1]
                miss_c = miss_mask_full[:max_row, c0:c1]

                def overlay_png(bg_path, out_path):
                    img = mpimg.imread(bg_path)
                    fig, ax = plt.subplots(figsize=(5.0, 3.6))
                    ax.imshow(img, extent=ext, origin="lower", aspect="auto")
                    ax.imshow(np.ma.masked_where(~miss_c, miss_c.astype(float)),
                              extent=ext, origin="lower", aspect="auto", cmap=BLUE_MASK_CMAP, vmin=0, vmax=1)
                    ax.imshow(np.ma.masked_where(~add_c, add_c.astype(float)),
                              extent=ext, origin="lower", aspect="auto", cmap=RED_MASK_CMAP, vmin=0, vmax=1)
                    ax.set_xlim(0.0, dur); ax.set_ylim(0.0, fmax)
                    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
                    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

                outAov = crops_dir / f"{crop_id}__A_praat_overlay.png"
                outBov = crops_dir / f"{crop_id}__B_praat_overlay.png"
                overlay_png(outA, outAov)
                overlay_png(outB, outBov)
                entry["praat_A_overlay_path"] = str(outAov)
                entry["praat_B_overlay_path"] = str(outBov)
            except Exception:
                pass

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-vocoder", type=str, default=None,
                        help="Pick a vocoder, for example hifigan, comma separated list allowed.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save the unsmoothed annotation overlay.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete each pair folder before writing outputs.")

    # crops options
    parser.add_argument("--make-crops", action="store_true",
                        help="Generate top K crops from the smoothed normalized diff.")
    parser.add_argument("--crops-k", type=int, default=8)
    parser.add_argument("--gate-time-pct", type=float, default=25.0,
                        help="Drop lowest energy frames of REAL by this percentile.")
    parser.add_argument("--fmin", type=float, default=80.0)
    parser.add_argument("--fmax", type=float, default=6000.0)
    parser.add_argument("--grow-t-ms", type=float, default=32.0)
    parser.add_argument("--grow-f-hz", type=float, default=300.0)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--save-crop-overlays", action="store_true")

    # TextGrid mapping options
    parser.add_argument("--textgrid-root", type=str, default="/home/opc/aligned_textgrids",
                        help="Root containing <bona_id>/<bona_id>.TextGrid or any descendant .TextGrid.")
    parser.add_argument("--words-tier", type=str, default="words",
                        help="Tier name for word intervals in TextGrid.")
    parser.add_argument("--phones-tier", type=str, default="phones",
                        help="Tier name for phones intervals in TextGrid.")
    parser.add_argument("--silence-tokens", type=str, default="sil,sp,spn,pau,nsn",
                        help="Comma separated labels that indicate silence or non speech.")
    parser.add_argument("--min-word-overlap-ratio", type=float, default=0.2,
                        help="Min fraction of a word duration that must overlap a crop to count.")
    parser.add_argument("--min-phone-overlap-ratio", type=float, default=0.2,
                        help="Min fraction of a phone duration that must overlap a crop to count.")

    # Praat rendering options, Picture
    parser.add_argument("--render-praat-crops", action="store_true",
                        help="Re render each crop using Praat Picture, fixed 0 to fmax Hz.")
    parser.add_argument("--praat-fmax", type=float, default=5000.0)
    parser.add_argument("--praat-win-ms", type=float, default=5.0,
                        help="Praat Spectrogram window length, ms, about 5 ms is wideband.")
    parser.add_argument("--praat-db-range", type=float, default=60.0)
    parser.add_argument("--praat-preemph-from", type=float, default=50.0)
    parser.add_argument("--draw-pitch", action="store_true",
                        help="Overlay Praat Pitch on Praat crops.")
    parser.add_argument("--f0-floor", type=float, default=60.0)
    parser.add_argument("--f0-ceiling", type=float, default=400.0)
    parser.add_argument("--f0-axis-max", type=float, default=800.0)
    parser.add_argument("--overlay-on-praat", action="store_true",
                        help="After Praat paints, add red and blue overlays via matplotlib.")

    args = parser.parse_args()

    # Prepare silence set
    silence_set = {tok.strip().lower() for tok in (args.silence_tokens or "").split(",") if tok.strip()}

    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)

    vocoder_filter = None
    if args.only_vocoder:
        vocoder_filter = {v.strip().lower() for v in args.only_vocoder.split(",") if v.strip()}

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # choose a single display cap for matplotlib figures
    display_fmax = args.praat_fmax if args.praat_fmax else (args.fmax if args.fmax else target_sr / 2.0)

    index_rows = []
    for spk, vocoder, spoof_stem, sp_path, bona_id, bf_path in tqdm(iter_pairs(PROTOCOL_TXT, vocoder_filter, bona_index)):
        try:
            ys = load_mono_resample(sp_path, target_sr)
            yb = load_mono_resample(bf_path, target_sr)

            (G_Mb, G_Ms, mask_s, tau_s, mask_r, tau_r,
             Mb_raw, Ms_raw, add_mask, miss_mask, norm_s, G_Mb_lin, G_Ms_lin) = compute_products(yb, ys)

            # Find matching TextGrid and pre load tiers
            words_intervals = []
            phones_intervals = []
            if args.textgrid_root:
                tg_path = find_matching_textgrid(bona_id, args.textgrid_root)
                if tg_path:
                    if args.words_tier:
                        words_intervals = load_intervals_from_textgrid(tg_path, args.words_tier)
                    if args.phones_tier:
                        phones_intervals = load_intervals_from_textgrid(tg_path, args.phones_tier)

            bf_dir = out_root / bona_id / vocoder
            if args.overwrite and bf_dir.exists():
                if out_root in bf_dir.parents:
                    shutil.rmtree(bf_dir)
            bf_dir.mkdir(parents=True, exist_ok=True)

            # npy outputs, smoothed branch
            np.save(bf_dir / "Mb_smooth.npy", G_Mb.astype(np.float32))
            np.save(bf_dir / f"Ms_smooth__{spoof_stem}.npy", G_Ms.astype(np.float32))

            # save combined and directional masks
            np.save(bf_dir / f"mask95_smoothed__{spoof_stem}.npy", mask_s.astype(np.uint8))
            np.save(bf_dir / f"mask95_add_smoothed__{spoof_stem}.npy", add_mask.astype(np.uint8))
            np.save(bf_dir / f"mask95_miss_smoothed__{spoof_stem}.npy", miss_mask.astype(np.uint8))
            np.save(bf_dir / f"norm_s__{spoof_stem}.npy", norm_s.astype(np.float32))

            # paper style images, now clamped to display_fmax
            save_annotation_overlay(G_Mb, add_mask, miss_mask, bf_dir / f"smoothed_mask95__{spoof_stem}.png",
                                    "Smoothed Annotation, 95 percent", fmax_display=display_fmax)
            save_quicklook_paper_style(bf_dir / f"quicklook__{spoof_stem}.png", G_Mb, G_Ms, add_mask, miss_mask,
                                       fmax_display=display_fmax)

            # optional, also save raw overlay
            if args.save_raw:
                np.save(bf_dir / f"mask95_raw__{spoof_stem}.npy", mask_r.astype(np.uint8))
                save_annotation_overlay(Mb_raw, mask_s, np.zeros_like(mask_s, dtype=bool),
                                        bf_dir / f"raw_mask95__{spoof_stem}.png",
                                        "Annotation, 95 percent", fmax_display=display_fmax)

            num_crops = 0
            manifest = None
            if args.make_crops:
                manifest = make_crops(
                    out_dir=bf_dir,
                    pair_id=f"{bona_id}__{spoof_stem}",
                    G_Mb_db=G_Mb,
                    G_Ms_db=G_Ms,
                    add_mask=add_mask,
                    miss_mask=miss_mask,
                    norm_s=norm_s,
                    G_Mb_lin=G_Mb_lin,
                    words_intervals=words_intervals,
                    phones_intervals=phones_intervals,
                    silence_set=silence_set,
                    min_word_overlap_ratio=args.min_word_overlap_ratio,
                    min_phone_overlap_ratio=args.min_phone_overlap_ratio,
                    crops_k=args.crops_k,
                    gate_time_pct=args.gate_time_pct,
                    fmin_hz=args.fmin,
                    fmax_hz=args.fmax,
                    grow_t_ms=args.grow_t_ms,
                    grow_f_hz=args.grow_f_hz,
                    min_area=args.min_area,
                    save_overlays=args.save_crop_overlays,
                )
                num_crops = len(manifest)

                # Re render crops using Praat Picture if requested
                if args.render_praat_crops and num_crops > 0:
                    crops_dir = bf_dir / "crops"
                    try:
                        render_praat_picture_crops(
                            y_real=yb, y_fake=ys, sr=target_sr,
                            crops_dir=crops_dir, manifest=manifest,
                            add_mask_full=add_mask, miss_mask_full=miss_mask,
                            fmax=args.praat_fmax, win_ms=args.praat_win_ms, db_range=args.praat_db_range,
                            preemph_from=args.praat_preemph_from,
                            draw_pitch=args.draw_pitch, f0_floor=args.f0_floor,
                            f0_ceiling=args.f0_ceiling, f0_axis_max=args.f0_axis_max,
                            overlay_on_praat=args.overlay_on_praat,
                        )
                        # persist updated manifest with Praat paths
                        with open(crops_dir / "crops_manifest.json", "w") as f:
                            json.dump(manifest, f, indent=2)
                    except Exception as e:
                        print(f"[warn] Praat crop rendering failed on {bona_id} vs {spoof_stem}, {e}")

            index_rows.append({
                "bona_id": bona_id,
                "vocoder": vocoder,
                "spoof_stem": spoof_stem,
                "tau_0_95_smoothed": float(tau_s),
                "tau_0_95_raw": float(tau_r),
                "frames": int(G_Mb.shape[1]),
                "freq_bins": int(G_Mb.shape[0]),
                "num_crops": int(num_crops),
                "out_dir": str(bf_dir),
            })
        except Exception as e:
            print(f"Error on {spoof_stem} vs {bona_id}, {e}")

    if index_rows:
        with open(out_root / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)

if __name__ == "__main__":
    main()