import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Set, Tuple
import shutil
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, zoom
from tqdm import tqdm

# White figure background like Praat
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

# Try Praat interface
try:
    import parselmouth as pm
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False

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
trunc_t = ((ksize_t - 1) / 2.0) / max(sigma_t, 1e-6)
trunc_f = ((ksize_f - 1) / 2.0) / max(sigma_f, 1e-6)

quantile_tau = 0.95
eps = 1e-8
use_dtw = True

# ------------ helpers ------------
def load_mono_resample(path, sr):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return y

def stft_mag(y):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center))

def logmag_spec(y):
    S = stft_mag(y)
    return np.log(S + 1e-6)

def smooth_2d(S):
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

# parse lines like: LA_0079 hifigan_LA_T_1703395 , hifigan spoof
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
                print(f"Skip, spoof audio not found: {spoof_stem}")
                continue

            bona_id = bona_id_from_spoof(spoof_stem)
            if bona_id is None:
                print(f"Skip, cannot derive bona ID from spoof stem: {spoof_stem}")
                continue

            bf_path = bona_index.get(bona_id)
            if bf_path is None:
                folder = ASV19_PER_AUDIO_ROOT / bona_id
                if folder.exists():
                    print(f"Skip, bona folder exists but audio file missing: {folder}")
                else:
                    print(f"Skip, bona folder not found under {ASV19_PER_AUDIO_ROOT}: {bona_id}")
                continue

            yield spk, vocoder_lc, spoof_stem, sp_path, bona_id, bf_path

def compute_products(y_bona, y_spoof):
    Mb = logmag_spec(y_bona)
    Ms = logmag_spec(y_spoof)

    if use_dtw:
        Mb, Ms = dtw_time_align_same_length(Mb, Ms)
    else:
        T = min(Mb.shape[1], Ms.shape[1])
        Mb, Ms = Mb[:, :T], Ms[:, :T]

    T = min(Mb.shape[1], Ms.shape[1])
    Mb, Ms = Mb[:, :T], Ms[:, :T]

    G_Mb = smooth_2d(Mb)
    G_Ms = smooth_2d(Ms)
    diff_s = np.abs(G_Ms - G_Mb)
    norm_s = diff_s / (np.abs(G_Mb) + eps)
    flat_s = norm_s[np.isfinite(norm_s)].ravel()
    tau_s = np.quantile(flat_s, quantile_tau) if flat_s.size else 0.0
    mask_s = norm_s > tau_s

    diff_r = np.abs(Ms - Mb)
    norm_r = diff_r / (np.abs(Mb) + eps)
    flat_r = norm_r[np.isfinite(norm_r)].ravel()
    tau_r = np.quantile(flat_r, quantile_tau) if flat_r.size else 0.0
    mask_r = norm_r > tau_r

    return G_Mb, G_Ms, mask_s, tau_s, mask_r, tau_r, Mb, Ms

# ---------- Praat-style display helpers ----------
def _extent(frames, sr, hop, fmax):
    t_max = frames * hop / sr
    return [0.0, t_max, 0.0, fmax]

def praat_display_parselmouth(y: np.ndarray,
                              sr: int,
                              fmax: float,
                              winlen: float,
                              timestep: float,
                              dyn_range_db: float,
                              pre_emph_from: float) -> Tuple[np.ndarray, list]:
    # Build Sound
    snd = pm.Sound(y, sampling_frequency=sr)

    # Apply Praat pre-emphasis first, then make spectrogram
    if pre_emph_from and pre_emph_from > 0:
        try:
            snd = pm.praat.call(snd, "Pre-emphasize", float(pre_emph_from))
        except Exception as e:
            print(f"[warn] Pre-emphasize failed, continuing without, {e}")

    # Use positional args in Praat order:
    # window_length, maximum_frequency, time_step, [frequency_step], [window_shape]
    spec = snd.to_spectrogram(float(winlen), float(fmax), float(timestep))

    V = np.maximum(spec.values, 1e-12)  # [freq, time]
    S_db = 10.0 * np.log10(V)
    max_db = float(np.max(S_db))
    floor = max_db - float(dyn_range_db)
    S_db = np.clip(S_db, floor, max_db)
    disp = (S_db - floor) / float(dyn_range_db)  # 0..1
    extent = [0.0, spec.xmax, 0.0, spec.ymax]
    return disp, extent

def praat_display_librosa(y: np.ndarray,
                          sr: int,
                          fmax: float,
                          dyn_range_db: float) -> Tuple[np.ndarray, list]:
    S = stft_mag(y)
    nyq = sr / 2.0
    fmax_eff = min(fmax, nyq)
    max_bin = int(np.floor(fmax_eff * n_fft / sr))
    S = S[: max_bin + 1, :]
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    max_db = np.max(S_db)
    floor = max_db - dyn_range_db
    S_db = np.clip(S_db, floor, max_db)
    disp = (S_db - floor) / dyn_range_db
    extent = _extent(S.shape[1], sr, hop, fmax_eff)
    return disp, extent

def make_praat_displays(yb: np.ndarray,
                        ys: np.ndarray,
                        sr: int,
                        use_praat_backend: bool,
                        fmax: float,
                        winlen: float,
                        timestep: float,
                        dyn_range_db: float,
                        pre_emph_from: float) -> Tuple[np.ndarray, np.ndarray, list]:
    if use_praat_backend and HAS_PRAAT:
        Db, ext_b = praat_display_parselmouth(yb, sr, fmax, winlen, timestep, dyn_range_db, pre_emph_from)
        Ds, ext_s = praat_display_parselmouth(ys, sr, fmax, winlen, timestep, dyn_range_db, pre_emph_from)
    else:
        Db, ext_b = praat_display_librosa(yb, sr, fmax, dyn_range_db)
        Ds, ext_s = praat_display_librosa(ys, sr, fmax, dyn_range_db)

    Lb = np.log(np.maximum(Db, 1e-8))
    Ls = np.log(np.maximum(Ds, 1e-8))
    if use_dtw:
        Lb_a, Ls_a = dtw_time_align_same_length(Lb, Ls)
        Db = np.exp(Lb_a)
        Ds = np.exp(Ls_a)

    frames = min(Db.shape[1], Ds.shape[1])
    Db, Ds = Db[:, :frames], Ds[:, :frames]
    extent = [0.0, frames * timestep, 0.0, min(fmax, sr / 2.0)]
    return Db, Ds, extent

def _stretch01(x):
    x = x - np.nanmin(x)
    vmax = np.percentile(x, 98)
    if vmax > 0:
        x = np.clip(x / vmax, 0, 1)
    return x

def save_annotation_overlay_disp(bg_disp, mask_disp, extent, out_path, title):
    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(bg_disp, extent=extent, aspect="auto", origin="lower",
               cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    plt.imshow(np.ma.masked_where(mask_disp == 0, mask_disp.astype(float)),
               extent=extent, aspect="auto", origin="lower",
               cmap="Reds", interpolation="nearest", alpha=0.9, vmin=0, vmax=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

def save_quicklook_praat_style(
    out_path,
    disp_b,
    disp_s,
    mask_disp,
    extent,
    width_in=16.0,   # wide like Praat
    height_in=9.0    # enough vertical space for 3 rows
):
    # 3 rows, 1 column, shared X for consistent time axis
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(width_in, height_in), sharex=True
    )

    # Common limits based on extent
    t0, t1, f0, f1 = extent[0], extent[1], extent[2], extent[3]

    # Top: Smoothed Real
    ax1.imshow(
        disp_b, extent=extent, aspect="auto", origin="lower",
        cmap="gray_r", interpolation="nearest", vmin=0, vmax=1
    )
    ax1.set_xlim(t0, t1); ax1.set_ylim(f0, f1)
    ax1.set_title("Smoothed Real")
    ax1.set_ylabel("Frequency [Hz]")

    # Middle: Smoothed Fake
    ax2.imshow(
        disp_s, extent=extent, aspect="auto", origin="lower",
        cmap="gray_r", interpolation="nearest", vmin=0, vmax=1
    )
    ax2.set_xlim(t0, t1); ax2.set_ylim(f0, f1)
    ax2.set_title("Smoothed Fake")
    ax2.set_ylabel("Frequency [Hz]")

    # Bottom: Annotation overlay
    ax3.imshow(
        disp_b, extent=extent, aspect="auto", origin="lower",
        cmap="gray_r", interpolation="nearest", vmin=0, vmax=1
    )
    ax3.imshow(
        np.ma.masked_where(mask_disp == 0, mask_disp.astype(float)),
        extent=extent, aspect="auto", origin="lower",
        cmap="Reds", interpolation="nearest", alpha=0.9, vmin=0, vmax=1
    )
    ax3.set_xlim(t0, t1); ax3.set_ylim(f0, f1)
    ax3.set_title("Smoothed Annotation, 95 percent")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Frequency [Hz]")

    # Clean simple look like Praat
    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

# ---------- legacy grayscale helpers (fallback) ----------
def save_annotation_overlay(bg, mask, out_path, title):
    frames = bg.shape[1]
    ext = _extent(frames, target_sr, hop, target_sr / 2.0)
    bg_disp = _stretch01(bg.copy())

    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(bg_disp, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    plt.imshow(np.ma.masked_where(mask == 0, mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap="Reds", interpolation="nearest", alpha=0.9, vmin=0, vmax=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

def save_quicklook_paper_style(out_path, G_Mb, G_Ms, mask):
    frames = G_Mb.shape[1]
    ext = _extent(frames, target_sr, hop, target_sr / 2.0)
    disp_b = _stretch01(G_Mb.copy())
    disp_s = _stretch01(G_Ms.copy())

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
    ax3.imshow(np.ma.masked_where(mask == 0, mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap="Reds", interpolation="nearest", alpha=0.9, vmin=0, vmax=1)
    ax3.set_title("Smoothed Annotation, 95 percent"); ax3.set_xlabel("Time [s]"); ax3.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

# ---------- args and main ----------
def parse_only_ids(arg_val: Optional[str]) -> Optional[Set[str]]:
    if not arg_val:
        return None
    ids = {s.strip() for s in arg_val.split(",") if s.strip()}
    cleaned = set()
    for s in ids:
        m = re.search(r"(LA_[TDE]_\d{7})", s)
        if m:
            cleaned.add(m.group(1))
    return cleaned if cleaned else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-vocoder", type=str, default=None,
                        help="Pick a vocoder, for example hifigan, comma separated list allowed.")
    parser.add_argument("--only-ids", type=str, default=None,
                        help="Comma separated bona IDs, for example LA_D_1024892")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save the unsmoothed annotation overlay.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete each pair folder before writing outputs.")

    # Praat-style display controls
    parser.add_argument("--praat-backend", action="store_true",
                        help="Use Praat via parselmouth for display if available, otherwise mimic.")
    parser.add_argument("--praat-fmax", type=float, default=5000.0,
                        help="Max frequency for Praat-style display in Hz [default 5000].")
    parser.add_argument("--praat-dynrange", type=float, default=50.0,
                        help="Dynamic range in dB for display [default 50].")
    parser.add_argument("--praat-winlen", type=float, default=0.005,
                        help="Praat window length in seconds [default 0.005].")
    parser.add_argument("--praat-preemph", type=float, default=50.0,
                        help="Praat pre-emphasis from frequency in Hz [default 50].")
    parser.add_argument("--praat-timestep", type=float, default=0.002,
                        help="Time step for Praat-style spectrogram in seconds [default 0.002].")

    args = parser.parse_args()

    if args.praat_backend and not HAS_PRAAT:
        print("[warn] --praat-backend requested but parselmouth not available, using librosa mimic.")

    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)

    vocoder_filter = None
    if args.only_vocoder:
        vocoder_filter = {v.strip().lower() for v in args.only_vocoder.split(",") if v.strip()}

    only_ids = parse_only_ids(args.only_ids)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for spk, vocoder, spoof_stem, sp_path, bona_id, bf_path in tqdm(iter_pairs(PROTOCOL_TXT, vocoder_filter, bona_index)):
        if only_ids and bona_id not in only_ids:
            continue
        try:
            ys = load_mono_resample(sp_path, target_sr)
            yb = load_mono_resample(bf_path, target_sr)

            G_Mb, G_Ms, mask_s, tau_s, mask_r, tau_r, Mb_raw, Ms_raw = compute_products(yb, ys)

            # Praat-like displays, aligned and cropped to fmax
            disp_b, disp_s, extent = make_praat_displays(
                yb, ys, target_sr, args.praat_backend, args.praat_fmax,
                args.praat_winlen, args.praat_timestep, args.praat_dynrange, args.praat_preemph
            )
            # Resize mask to the display grid for overlays
            mask_disp = zoom(mask_s.astype(float),
                             (disp_b.shape[0] / mask_s.shape[0], disp_b.shape[1] / mask_s.shape[1]),
                             order=0)

            bf_dir = out_root / bona_id / vocoder
            if args.overwrite and bf_dir.exists():
                if out_root in bf_dir.parents:
                    shutil.rmtree(bf_dir)
            bf_dir.mkdir(parents=True, exist_ok=True)

            # npy outputs, smoothed branch
            np.save(bf_dir / "Mb_smooth.npy", G_Mb.astype(np.float32))
            np.save(bf_dir / f"Ms_smooth__{spoof_stem}.npy", G_Ms.astype(np.float32))
            np.save(bf_dir / f"mask95_smoothed__{spoof_stem}.npy", mask_s.astype(np.uint8))

            # Praat-style images
            save_quicklook_praat_style(bf_dir / f"quicklook__{spoof_stem}.png",
                                       disp_b, disp_s, mask_disp, extent)
            save_annotation_overlay_disp(disp_b, mask_disp,
                                         extent,
                                         bf_dir / f"smoothed_mask95__{spoof_stem}.png",
                                         "Smoothed Annotation, 95 percent")

            # Optional raw overlay on legacy grid
            if args.save_raw:
                np.save(bf_dir / f"mask95_raw__{spoof_stem}.npy", mask_r.astype(np.uint8))
                save_annotation_overlay(G_Mb, mask_r, bf_dir / f"raw_mask95__{spoof_stem}.png",
                                        "Annotation, 95 percent")

            index_rows.append({
                "bona_id": bona_id,
                "vocoder": vocoder,
                "spoof_stem": spoof_stem,
                "tau_0_95_smoothed": float(tau_s),
                "tau_0_95_raw": float(tau_r),
                "frames": int(G_Mb.shape[1]),
                "freq_bins": int(G_Mb.shape[0]),
                "out_dir": str(bf_dir),
            })
        except Exception as e:
            print(f"Error on {spoof_stem} vs {bona_id}: {e}")

    if index_rows:
        with open(out_root / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)

if __name__ == "__main__":
    main()