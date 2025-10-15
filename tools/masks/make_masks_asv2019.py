# tools/make_masks_min_asv19orig.py
import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict
import shutil
import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ----------- fixed paths (edit if needed) -----------
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_ORIG_ROOT   = Path("/home/opc/test/LA/LA")  # expects */ASVspoof2019_LA_{train,dev,eval}/flac/*.flac
PROTOCOL_TXT      = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")
OUT_DIR           = Path("./mask_outputs_vocv4")

# ----------- STFT & smoothing -----------
target_sr  = 16000
n_fft      = 1024
hop        = 256
win_length = 1024
center     = True

# separable Gaussian (time=3, freq=11) with variances (3, 5)
ksize_t, ksize_f = 3, 11
var_t, var_f     = 3.0, 5.0
sigma_t, sigma_f = np.sqrt(var_t), np.sqrt(var_f)
# truncation so output kernel length matches requested sizes
trunc_t = ((ksize_t - 1) / 2.0) / max(sigma_t, 1e-6)
trunc_f = ((ksize_f - 1) / 2.0) / max(sigma_f, 1e-6)

default_tau = 0.95
eps = 1e-8

# ------------ helpers ------------
def load_mono_resample(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y

def mag_spec(y: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center))

def smooth_2d(S: np.ndarray) -> np.ndarray:
    """Exact separable Gaussian, time then frequency."""
    St = gaussian_filter1d(S, sigma=sigma_t, axis=1, truncate=trunc_t)
    Sf = gaussian_filter1d(St, sigma=sigma_f, axis=0, truncate=trunc_f)
    return Sf

def to_db(mag: np.ndarray) -> np.ndarray:
    db = librosa.amplitude_to_db(mag, ref=np.max)
    return np.clip(db, -60.0, 0.0).astype(np.float32, copy=False)

# parse lines like: LA_0079 hifigan_LA_T_1703395 , hifigan spoof
line_rx = re.compile(r"^(LA_\d{4})\s+([A-Za-z0-9_-]+)\s+-\s+([A-Za-z0-9_-]+)\s+spoof")

def bona_id_from_spoof(stem: str) -> Optional[str]:
    m = re.search(r"(LA_[TDE]_\d{7})", stem)
    return m.group(1) if m else None

def build_bona_index_original(root: Path) -> Dict[str, Path]:
    """
    Build {ASV_ID : audio_path} from original ASVspoof2019-LA layout:
      root/
        ASVspoof2019_LA_train/flac/*.flac
        ASVspoof2019_LA_dev/flac/*.flac
        ASVspoof2019_LA_eval/flac/*.flac
    Also accepts *.wav if present.
    """
    idx: Dict[str, Path] = {}
    if not root.exists():
        print(f"[error] ASV19_ORIG_ROOT not found: {root}")
        return idx
    any_found = False
    for split in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev", "ASVspoof2019_LA_eval"]:
        d = root / split / "flac"
        if not d.is_dir():
            continue
        for pat in ("*.flac", "*.wav"):
            for p in d.glob(pat):
                idx[p.stem] = p
                any_found = True
    if not any_found:
        print(f"[error] No audio found under {root}/<split>/flac")
    else:
        print(f"[info] Indexed ASVspoof2019 original layout at {root}, total {len(idx)} items")
    return idx

def find_spoof_audio(spoof_stem: str) -> Optional[Path]:
    for ext in (".wav", ".flac"):
        p = VOC_V4_AUDIO_ROOT / f"{spoof_stem}{ext}"
        if p.exists():
            return p
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.wav")) or list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.flac"))
    return hits[0] if hits else None

def iter_pairs(protocol_path: Path, bona_index: Dict[str, Path]):
    with open(protocol_path, "r", encoding="utf-8") as f:
        for raw in f:
            m = line_rx.match(raw.strip())
            if not m:
                continue
            spk, spoof_stem, vocoder = m.groups()

            sp_path = find_spoof_audio(spoof_stem)
            if sp_path is None:
                print(f"[skip] spoof audio not found: {spoof_stem}")
                continue

            bona_id = bona_id_from_spoof(spoof_stem)
            if bona_id is None:
                print(f"[skip] cannot derive bona ID from spoof stem: {spoof_stem}")
                continue

            bf_path = bona_index.get(bona_id)
            if bf_path is None:
                print(f"[skip] bonafide audio not found under ASV19_ORIG_ROOT for {bona_id}")
                continue

            yield spk, vocoder.lower(), spoof_stem, sp_path, bona_id, bf_path

def compute_mask_and_specs(y_bona: np.ndarray, y_spoof: np.ndarray, tau: float):
    """
    Returns:
      G_Ms_db: float32 dB smoothed spoof spectrogram in [-60,0] (F x T)
      mask_s:  uint8 binary mask (F x T) with threshold at tau-quantile of |G(Ms)-G(Mb)| / |G(Mb)|
    """
    Mb = mag_spec(y_bona)
    Ms = mag_spec(y_spoof)

    T = min(Mb.shape[1], Ms.shape[1])
    Mb = Mb[:, :T]
    Ms = Ms[:, :T]

    G_Mb_lin = smooth_2d(Mb)
    G_Ms_lin = smooth_2d(Ms)

    # normalized absolute diff vs REAL energy (SpecSegDiff mask)
    signed = G_Ms_lin - G_Mb_lin
    norm_s = np.abs(signed) / (G_Mb_lin + eps)
    flat = norm_s[np.isfinite(norm_s)].ravel()
    thr = np.quantile(flat, tau) if flat.size else 0.0
    mask_s = (norm_s > thr).astype(np.uint8, copy=False)

    G_Ms_db = to_db(G_Ms_lin)
    return G_Ms_db, mask_s

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    bona_index = build_bona_index_original(ASV19_ORIG_ROOT)

    index_rows = []
    for spk, vocoder, spoof_stem, sp_path, bona_id, bf_path in tqdm(
        iter_pairs(PROTOCOL_TXT, bona_index), desc="pairs"
    ):
        try:
            ys = load_mono_resample(sp_path, target_sr)  # spoof
            yb = load_mono_resample(bf_path, target_sr)  # bona

            G_Ms_db, mask_s = compute_mask_and_specs(yb, ys, tau=default_tau)

            pair_dir = out_root / bona_id / vocoder
            if args.overwrite and pair_dir.exists() and out_root in pair_dir.parents:
                shutil.rmtree(pair_dir)
            pair_dir.mkdir(parents=True, exist_ok=True)

            # minimal artifacts needed for training
            np.save(pair_dir / f"Ms_smooth__{spoof_stem}.npy", G_Ms_db.astype(np.float32))
            np.save(pair_dir / f"mask95_smoothed__{spoof_stem}.npy", mask_s.astype(np.uint8))

            index_rows.append({
                "bona_id": bona_id,
                "vocoder": vocoder,
                "spoof_stem": spoof_stem,
                "frames": int(G_Ms_db.shape[1]),
                "freq_bins": int(G_Ms_db.shape[0]),
                "out_dir": str(pair_dir),
            })
        except Exception as e:
            print(f"[error] {spoof_stem} vs {bona_id}: {e}")

    if index_rows:
        with open(out_root / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)

if __name__ == "__main__":
    main()