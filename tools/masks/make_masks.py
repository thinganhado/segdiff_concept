# tools/make_masks_min.py
import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Set
import shutil
import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ----------- fixed inputs ----------- #
VOC_V4_AUDIO_ROOT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/wav")
ASV19_PER_AUDIO_ROOT = Path("/home/opc/datasets/asvspoof2019/LA/per_audio_folders")
PROTOCOL_TXT = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/protocol.txt")
TRAIN_LST = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/scp/train.lst")
DEV_LST   = Path("/home/opc/datasets/project09/project09-voc.v4/voc.v4/scp/dev.lst")
OUT_DIR = Path("./data/voc.v4")  # write Monu-style here by default

# ----------- STFT & smoothing ----------- #
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

default_tau = 0.95
eps = 1e-8

# ----------- helpers ----------- #
def load_mono_resample(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32, copy=False)

def mag_spec(y: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center))

def smooth_2d(S: np.ndarray) -> np.ndarray:
    """Exact separable Gaussian, time then frequency, sizes [3, 11], variances [3, 5]."""
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

def build_bona_index(root: Path) -> Dict[str, Path]:
    if not root.exists():
        print(f"[error] ASV19_PER_AUDIO_ROOT not found: {root}")
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
        f"[info] Indexed bona folders at {root}, total {len(idx)} items "
        f"[T {sum(k.startswith('LA_T_') for k in idx)}, "
        f"D {sum(k.startswith('LA_D_') for k in idx)}, "
        f"E {sum(k.startswith('LA_E_') for k in idx)}]"
    )
    return idx

def find_spoof_audio(spoof_stem: str) -> Optional[Path]:
    for ext in (".wav", ".flac"):
        p = VOC_V4_AUDIO_ROOT / f"{spoof_stem}{ext}"
        if p.exists():
            return p
    hits = list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.wav")) or list(VOC_V4_AUDIO_ROOT.rglob(f"{spoof_stem}.flac"))
    return hits[0] if hits else None

def read_id_list(p: Path) -> Set[str]:
    """Return a set of stems from a simple list file. Accepts paths or names with/without extensions."""
    out: Set[str] = set()
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]            # first field
            stem = Path(token).stem           # drop extension if any
            out.add(stem)
    return out

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
                folder = ASV19_PER_AUDIO_ROOT / bona_id
                if folder.exists():
                    print(f"[skip] bona folder exists but audio missing: {folder}")
                else:
                    print(f"[skip] bona folder not found under {ASV19_PER_AUDIO_ROOT}: {bona_id}")
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

    signed = G_Ms_lin - G_Mb_lin
    norm_s = np.abs(signed) / (G_Mb_lin + eps)
    flat = norm_s[np.isfinite(norm_s)].ravel()
    thr = np.quantile(flat, tau) if flat.size else 0.0
    mask_s = (norm_s > thr).astype(np.uint8, copy=False)

    G_Ms_db = to_db(G_Ms_lin)
    return G_Ms_db, mask_s

# -------------- main -------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    train_img = out_root / "Training" / "img"
    train_msk = out_root / "Training" / "mask"
    test_img  = out_root / "Test" / "img"
    test_msk  = out_root / "Test" / "mask"

    # clean/create
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    for d in (train_img, train_msk, test_img, test_msk):
        d.mkdir(parents=True, exist_ok=True)

    # read splits
    train_ids = read_id_list(TRAIN_LST)
    dev_ids   = read_id_list(DEV_LST)   # used as "Test"
    if not train_ids and not dev_ids:
        print(f"[warn] empty split lists: {TRAIN_LST} {DEV_LST}")

    bona_index = build_bona_index(ASV19_PER_AUDIO_ROOT)

    written = 0
    for _, _, spoof_stem, sp_path, bona_id, bf_path in tqdm(iter_pairs(PROTOCOL_TXT, bona_index), desc="pairs"):
        try:
            # decide split by spoof_stem membership
            if spoof_stem in train_ids:
                tgt_img, tgt_msk = train_img, train_msk
            elif spoof_stem in dev_ids:
                tgt_img, tgt_msk = test_img, test_msk
            else:
                # not listed in either split → skip to avoid contaminating sets
                continue

            ys = load_mono_resample(sp_path, target_sr)  # spoof
            yb = load_mono_resample(bf_path, target_sr)  # bona

            G_Ms_db, mask_s = compute_mask_and_specs(yb, ys, tau=default_tau)

            # filenames mirror what the loader expects
            np.save(tgt_img / f"Ms_smooth__{spoof_stem}.npy", G_Ms_db.astype(np.float32))
            np.save(tgt_msk / f"mask95_smoothed__{spoof_stem}.npy", mask_s.astype(np.uint8))
            written += 1
        except Exception as e:
            print(f"[error] {spoof_stem} vs {bona_id}: {e}")

    # optional small index for bookkeeping
    idx_path = out_root / "index.csv"
    with open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "img_path", "mask_path"])
        for p in train_img.glob("Ms_smooth__*.npy"):
            stem = p.stem.split("Ms_smooth__")[-1]
            w.writerow(["Training", stem, str(p), str(train_msk / f"mask95_smoothed__{stem}.npy")])
        for p in test_img.glob("Ms_smooth__*.npy"):
            stem = p.stem.split("Ms_smooth__")[-1]
            w.writerow(["Test", stem, str(p), str(test_msk / f"mask95_smoothed__{stem}.npy")])

    print(f"[done] wrote {written} pairs into {out_root}")

if __name__ == "__main__":
    main()