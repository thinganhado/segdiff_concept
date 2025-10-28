import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict
import shutil
import numpy as np
import soundfile as sf
import librosa
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter 
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

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

# replace logmag_spec with magnitude for mask math
def mag_spec(y):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center))

def to_display(bg_mag):
    # for gray background only, stable visual dynamics
    return librosa.amplitude_to_db(bg_mag, ref=np.max)

def smooth_2d(S):
    """Exact separable Gaussian, time then frequency, sizes [3, 11], variances [3, 5]."""
    # S shape [freq, time]
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
    G_Mb = smooth_2d(Mb)
    G_Ms = smooth_2d(Ms)

    # Eq. (2): |G(Ms)-G(Mb)| / |G(Mb)| , on linear magnitude
    diff_s = np.abs(G_Ms - G_Mb)
    norm_s = diff_s / (G_Mb + eps)          # G_Mb is nonnegative
    flat_s = norm_s[np.isfinite(norm_s)].ravel()
    tau_s  = np.quantile(flat_s, quantile_tau) if flat_s.size else 0.0
    mask_s = norm_s > tau_s

    # optional unsmoothed branch
    diff_r = np.abs(Ms - Mb)
    norm_r = diff_r / (Mb + eps)
    flat_r = norm_r[np.isfinite(norm_r)].ravel()
    tau_r  = np.quantile(flat_r, quantile_tau) if flat_r.size else 0.0
    mask_r = norm_r > tau_r

    # prepare dB for display, fixed window
    def to_db(m):
        db = librosa.amplitude_to_db(m, ref=np.max)
        return np.clip(db, -60.0, 0.0)      # consistent contrast

    Mb_db   = to_db(Mb)
    Ms_db   = to_db(Ms)
    G_Mb_db = to_db(G_Mb)
    G_Ms_db = to_db(G_Ms)

    return G_Mb_db, G_Ms_db, mask_s, tau_s, mask_r, tau_r, Mb_db, Ms_db

# ---------- paper style visualization ----------
def _extent(frames, sr, hop, n_fft):
    t_max = frames * hop / sr
    f_max = sr / 2.0
    return [0.0, t_max, 0.0, f_max]

# 1) Light gray mapping that mimics the paper style
def paper_gray_from_db(db_img,
                       lo_pct=5.0, hi_pct=99.0,
                       out_lo=0.62, out_hi=0.96,
                       gamma=1.0):
    """
    Map dB image to a light gray 0..1 image.
    - Clamp by percentiles to avoid white slabs.
    - Put most values between out_lo and out_hi so background is light.
    - gamma >= 1 softens contrast, gamma < 1 boosts it.
    """
    lo = np.percentile(db_img, lo_pct)
    hi = np.percentile(db_img, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = -60.0, 0.0
    z = (db_img - lo) / (hi - lo + 1e-12)
    z = np.clip(z, 0.0, 1.0) ** gamma
    return out_lo + (out_hi - out_lo) * z  # 0.62..0.96 looks like the paper

# 2) Constant red for mask, transparent for zero
RED_MASK_CMAP = ListedColormap([[0, 0, 0, 0],   # 0 → fully transparent
                                [0.75, 0.0, 0.0, 1.0]])  # 1 → solid red

def save_annotation_overlay(bg_db, mask, out_path, title):
    frames = bg_db.shape[1]
    ext = _extent(frames, target_sr, hop, n_fft)

    bg_disp = paper_gray_from_db(bg_db)

    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(bg_disp, extent=ext, aspect="auto", origin="lower",
               cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    plt.imshow(np.ma.masked_where(mask == 0, mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()

def save_quicklook_paper_style(out_path, G_Mb_db, G_Ms_db, mask):
    frames = G_Mb_db.shape[1]
    ext = _extent(frames, target_sr, hop, n_fft)

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
    ax3.imshow(np.ma.masked_where(mask == 0, mask.astype(float)),
               extent=ext, aspect="auto", origin="lower",
               cmap=RED_MASK_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax3.set_title("Smoothed Annotation, 95 percent"); ax3.set_xlabel("Time [s]"); ax3.set_ylabel("")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-vocoder", type=str, default=None,
                        help="Pick a vocoder, for example hifigan, comma separated list allowed.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save the unsmoothed annotation overlay.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete each pair folder before writing outputs.")
    args = parser.parse_args()

    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)

    vocoder_filter = None
    if args.only_vocoder:
        vocoder_filter = {v.strip().lower() for v in args.only_vocoder.split(",") if v.strip()}

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for spk, vocoder, spoof_stem, sp_path, bona_id, bf_path in tqdm(iter_pairs(PROTOCOL_TXT, vocoder_filter, bona_index)):
        try:
            ys = load_mono_resample(sp_path, target_sr)
            yb = load_mono_resample(bf_path, target_sr)

            G_Mb, G_Ms, mask_s, tau_s, mask_r, tau_r, Mb_raw, Ms_raw = compute_products(yb, ys)

            bf_dir = out_root / bona_id / vocoder
            if args.overwrite and bf_dir.exists():
                if out_root in bf_dir.parents:
                    shutil.rmtree(bf_dir)
            bf_dir.mkdir(parents=True, exist_ok=True)

            # npy outputs, smoothed branch
            np.save(bf_dir / "Mb_smooth.npy", G_Mb.astype(np.float32))
            np.save(bf_dir / f"Ms_smooth__{spoof_stem}.npy", G_Ms.astype(np.float32))
            np.save(bf_dir / f"mask95_smoothed__{spoof_stem}.npy", mask_s.astype(np.uint8))

            # paper style images
            save_annotation_overlay(G_Mb, mask_s, bf_dir / f"smoothed_mask95__{spoof_stem}.png",
                                    "Smoothed Annotation, 95 percent")
            save_quicklook_paper_style(bf_dir / f"quicklook__{spoof_stem}.png", G_Mb, G_Ms, mask_s)

            # optional, also save raw overlay
            if args.save_raw:
                np.save(bf_dir / f"mask95_raw__{spoof_stem}.npy", mask_r.astype(np.uint8))
                save_annotation_overlay(Mb_raw, mask_r, bf_dir / f"raw_mask95__{spoof_stem}.png",
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