# detect_concepts.py
# Logic-only PASS concept detection operating on Mb/Ms/add/miss/norm and crops_manifest.json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import json
import numpy as np

EPS = 1e-12

# ---- array I/O + cropping shim (UPDATED to match your filenames) ------------
from pathlib import Path
import numpy as np

def find_pair_dir(bona_id: str, vocoder: str, roots) -> Path:
    roots = [Path(r) for r in roots]
    for r in roots:
        cand = (r / bona_id / vocoder)
        if cand.exists():
            return cand
        base = (r / bona_id)
        if base.exists():
            for p in base.iterdir():
                if p.is_dir() and p.name.lower() == vocoder.lower():
                    return p
    raise FileNotFoundError(f"pair dir not found for {bona_id}/{vocoder} under {roots}")

def _try_load(pair_dir: Path, names) -> np.ndarray:
    for name in names:
        p = pair_dir / name
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"none of {names} exist under {pair_dir}")

def load_pair_arrays(pair_dir: Path, spoof_stem: str) -> dict:
    """
    Robust loader for your layout (see screenshot). Accepts:
      Mb: Mb_smooth.npy  | Mb_db*.npy (any variant)
      Ms: Ms_smooth__{stem}.npy | Ms_db*.npy | Ms_smooth.npy
      add: mask95_add_smoothed__{stem}.npy | add*.npy
      miss: mask95_miss_smoothed__{stem}.npy | miss*.npy
      norm: norm_s__{stem}.npy | norm*.npy
    """
    stem = spoof_stem

    cands = {
        # wide net for Mb (baseline)
        "Mb_db": [
            f"Mb_db__{stem}.npy", f"Mb_db_{stem}.npy", "Mb_db.npy",
            f"Mb_smooth__{stem}.npy", "Mb_smooth.npy", "Mb.npy"
        ],
        # synthetic
        "Ms_db": [
            f"Ms_db__{stem}.npy", f"Ms_db_{stem}.npy", "Ms_db.npy",
            f"Ms_smooth__{stem}.npy", "Ms_smooth.npy", "Ms.npy"
        ],
        # masks (add/miss)
        "add": [
            f"mask95_add_smoothed__{stem}.npy",
            f"mask95_add__{stem}.npy",
            f"add__{stem}.npy", "mask_add.npy", "add.npy"
        ],
        "miss": [
            f"mask95_miss_smoothed__{stem}.npy",
            f"mask95_miss__{stem}.npy",
            f"miss__{stem}.npy", "mask_miss.npy", "miss.npy"
        ],
        # norm
        "norm": [
            f"norm_s__{stem}.npy", f"norm__{stem}.npy",
            "norm_s.npy", "norm.npy"
        ],
    }

    out = {}
    # Mb is required
    out["Mb_db"] = _try_load(pair_dir, cands["Mb_db"])
    F, T = out["Mb_db"].shape

    # required Ms, optional others
    out["Ms_db"] = _try_load(pair_dir, cands["Ms_db"])
    for k in ("add", "miss", "norm"):
        try:
            out[k] = _try_load(pair_dir, cands[k])
        except FileNotFoundError:
            out[k] = np.zeros((F, T), dtype=np.float32)

    # coerce to 2D float32
    for k, arr in list(out.items()):
        arr = np.array(arr)
        if arr.ndim != 2:
            arr = arr.squeeze()
        out[k] = arr.astype(np.float32, copy=False)

    return out

def crop_arrays(arrs_full: dict, box_idx) -> dict:
    t0, t1, f0, f1 = [int(v) for v in box_idx]
    return {k: v[f0:f1, t0:t1] for k, v in arrs_full.items()}
# -----------------------------------------------------------------------------

# ---- thresholds (as agreed + additions) ----
VOWEL_INTERIOR_MIN_OVL = 0.50
FOG_ADD_MIN = 0.08
FOG_MISS_MIN = 0.08
FOG_ADD_MISS_SUM_MIN = 0.26
FOG_H_DELTA_MIN = 0.08
FOG_K_DELTA_MAX = -0.30
FOG_WIDTH_DELTA = 2.0
FOG_FLATNESS_DELTA = 0.05
FOG_EDGE_DROP_RATIO = 0.10
FOG_PVR_DROP_DB = 2.5  # NEW: Formant peak-to-valley contrast drop (Mb->Ms)

F1_RANGE_DEFAULT = (250.0, 1000.0)
F2_RANGE_DEFAULT = (1000.0, 3000.0)
F3_RANGE_DEFAULT = (1800.0, 3500.0)  # NEW

FRIC_CENTROID_DROP_HZ = 200.0
FRIC_KURT_DROP = 0.20
EDGE_DROP_RATIO = 0.10
FRIC_SLOPE_MORE_NEG_DB_PER_KHZ = 3.0   # NEW: Ms high-band slope more negative than Mb
FRIC_TIME_EDGE_DROP_RATIO = 0.10       # NEW: temporal edge drop (onsets softened)

ATTEN_PSR_DROP_DB_BASE = 1.0
ATTEN_PSR_DROP_FRAC = 0.15
ATTEN_MISS_MIN = 0.10
ATTEN_MISS_MINUS_ADD = 0.02
ATTEN_TILT_MORE_NEG_DB_PER_KHZ = 1.0   # NEW: Ms broadband spectral tilt more negative than Mb
ATTEN_DROP_Q = 0.80                     # NEW: quantile over per-frame PSR drops

BCR_MIN = 2.0
BCR_WINDOW_MS = 24.0
BCR_STRIPE_COVERAGE_MIN = 0.60
BCR_HEIGHT_MIN_FRAC = 0.60
STEP_DCENTROID_HZ = 120.0
STEP_DENERGY_DB = 1.5
SMOOTH_JUMP_MIN = 4.0  # NEW: boundary jump vs intra-segment smoothness

PSEUDO_MIN_DUR_MS = 40.0
PSEUDO_MAX_HEIGHT_HZ = 400.0
PSEUDO_ADD_FRAC_MIN = 0.12
PSEUDO_RANGE_RATIO_MAX = 0.50  # NEW: Ms F2 range <= 50% of Mb -> "too steady" track

# --- Coarticulation (upgraded) ---
COART_F2_BAND = (1000.0, 3500.0)
COART_WIN_MS = 60.0
COART_MIN_SLOPE_MB = 50.0            # original boundary-window slope gate
COART_SLOPE_RATIO_MAX = 0.6
COART_RANGE_RATIO_MAX = 0.7

# New: V-to-V window cues (late-in-V1 vs early-in-V2)
COART_V1_LATE_MS   = 30.0
COART_V2_EARLY_MS  = 30.0
COART_MIN_DELTA_VV_HZ   = 150.0      # Mb must show at least this V-to-V separation
COART_DELTA_RATIO_MAX   = 0.6        # Ms ΔF2 must be <= this fraction of Mb ΔF2

# New: band sanity for F2
COART_MIN_F2_ROWS = 8                # require enough vertical aperture in F2

NOISE_MIN_ADD_MISS_SUM = 0.15
NOISE_GLOBAL_PCT = 0.80

# ---- hyperneatness / hyperflat prosody thresholds ----
NEAT_JITTER_RATIO_MAX = 0.60       # Ms track jitter must be <= 60% of Mb
NEAT_CURVE_RATIO_MAX  = 0.60       # Ms track curvature must be <= 60% of Mb
NEAT_EDGE_RATIO_MAX   = 0.85       # Ms edge-gradient <= 85% of Mb
NEAT_RESID_RATIO_MAX  = 0.75       # NEW: Ms residual-roughness <= 75% of Mb

FLAT_MIN_DUR_MS = 200.0            # minimum crop duration to evaluate prosody
FLAT_ENERGY_CV_ABS_MAX = 0.10      # absolute flatness gate on energy envelope
FLAT_ENERGY_CV_RATIO_MAX = 0.60    # or relative to Mb
LOWF0_BAND = (70.0, 400.0)         # rough F0 + low-harmonic band
FLAT_LOW_CENTROID_STD_HZ_MAX = 25.0
FLAT_LOW_CENTROID_STD_RATIO_MAX = 0.60
FLAT_INFLECT_RATIO_MAX = 0.60      # NEW: Ms inflections <= 60% of Mb

# =============================================================================
# Phone-tier helpers (TextGrid) + class gates
# =============================================================================
TEXTGRID_ROOT = Path("/home/opc/aligned_textgrids")

VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
FRICATIVES = {"F","V","TH","DH","S","Z","SH","ZH"}
APPROX = {"L","R","W","Y","J"}
SONORANTS = VOWELS | {"M","N","NG"} | {"L","R","W","Y","J"}

def _core(lbl: str) -> str:
    return re.sub(r"\d$", "", (lbl or "").upper())

def _is_vowel(lbl: str) -> bool: return _core(lbl) in VOWELS
def _is_fric(lbl: str)  -> bool: return _core(lbl) in FRICATIVES
def _is_appr(lbl: str)  -> bool: return _core(lbl) in APPROX
def _is_sonor(lbl: str) -> bool: return _core(lbl) in SONORANTS

def crop_overlap_fraction(entry: Dict, pred) -> float:
    spans = entry.get("phone_spans") or []
    if not spans: return 1.0  # fail-open if no phones
    t0, t1 = (entry.get("box_tfr") or entry.get("bbox_sec_hz"))[:2]
    dur = max(1e-9, float(t1) - float(t0))
    ovl = 0.0
    for s in spans:
        if pred(s.get("label","")):
            ovl += float(s.get("overlap_sec", 0.0))
    return ovl / dur

def crop_is_class_interior(entry: Dict, pred, min_ovl=0.60) -> bool:
    return crop_overlap_fraction(entry, pred) >= float(min_ovl)

def crop_has_boundary_near_mid(entry: Dict, tol_ms: float = 25.0) -> bool:
    spans = entry.get("phone_spans") or []
    if not spans: return True
    t0, t1 = (entry.get("box_tfr") or entry.get("bbox_sec_hz"))[:2]
    mid = 0.5*(float(t0)+float(t1)); tol = tol_ms/1000.0
    for s in spans:
        if abs(float(s["start"])-mid) <= tol or abs(float(s["end"])-mid) <= tol:
            return True
    return False

def crop_voiced_fraction(entry: Dict) -> float:
    # sonorants + voiced obstruents
    voiced_obs = {"B","D","G","V","DH","Z","ZH"}
    return crop_overlap_fraction(entry, lambda x: (_is_sonor(x) or _core(x) in voiced_obs))

def _load_textgrid_phones(bona_id: str) -> List[Dict]:
    """Lightweight parser for Praat long TextGrid; looks for tier named 'phones/phone/phoneme'."""
    tg_path = TEXTGRID_ROOT / bona_id / f"{bona_id}.TextGrid"
    if not tg_path.exists():
        return []
    txt = tg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i, n = 0, len(txt)
    phones: List[Dict] = []
    while i < n:
        if 'class = "IntervalTier"' in txt[i]:
            # find name
            name = ""
            for j in range(i, min(i+10, n)):
                if txt[j].strip().startswith('name ='):
                    name = txt[j].split('=',1)[1].strip().strip('"').lower()
                    break
            if name in {"phones","phone","phoneme"}:
                j = i
                # advance to intervals
                while j < n and 'intervals [' not in txt[j]:
                    j += 1
                # read intervals
                while j < n and 'item [' not in txt[j]:
                    if 'intervals [' in txt[j]:
                        try:
                            xmin = float(txt[j+1].split('=')[1])
                            xmax = float(txt[j+2].split('=')[1])
                            lab  = txt[j+3].split('=',1)[1].strip().strip('"')
                            phones.append({"start": xmin, "end": xmax, "label": lab})
                            j += 4
                            continue
                        except Exception:
                            pass
                    j += 1
                break
        i += 1
    return phones

def _attach_phone_spans(crop: Dict, phones: List[Dict]):
    if "phone_spans" in crop or not phones:
        return
    t0, t1 = (crop.get("bbox_sec_hz") or crop.get("box_tfr"))[:2]
    spans = []
    for ph in phones:
        s, e = float(ph["start"]), float(ph["end"])
        ovl = max(0.0, min(t1, e) - max(t0, s))
        if ovl > 0.0:
            spans.append({"label": ph["label"], "start": s, "end": e, "overlap_sec": ovl})
    crop["phone_spans"] = spans

# =============================================================================

# ---- utils ----
def hz_per_bin(sr: int, n_freq_bins: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def db_to_amp(db_img: np.ndarray) -> np.ndarray:
    return np.power(10.0, db_img / 20.0)

def entropy_bits(x: np.ndarray, axis: int = 0) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True) + EPS
    p = x / s
    h = -(p * (np.log(p + EPS) / np.log(2.0))).sum(axis=axis)
    return h

def kurtosis_along_freq(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sig2 = ((x - mu) ** 2).mean(axis=0) + EPS
    k = ((x - mu) ** 4).mean(axis=0) / (sig2 ** 2)
    return k

def spectral_centroid(f_axis: np.ndarray, x: np.ndarray) -> np.ndarray:
    num = (f_axis[:, None] * x).sum(axis=0)
    den = x.sum(axis=0) + EPS
    return num / den

def vertical_edge_strength(x_db_crop: np.ndarray) -> float:
    g = np.abs(np.diff(x_db_crop, axis=0))
    return float(np.mean(g))

def temporal_edge_strength(x_db_crop: np.ndarray) -> float:
    gx = np.abs(np.diff(x_db_crop, axis=1))
    return float(np.mean(gx))

def ridge_width_bins(x: np.ndarray, sr: int) -> float:
    F, T = x.shape
    fbin_hz = hz_per_bin(sr, F)
    hwin = max(1, int(round(200.0 / fbin_hz)))
    widths = []
    for t in range(T):
        col = x[:, t]
        idx = int(np.argmax(col))
        lo = max(0, idx - hwin); hi = min(F, idx + hwin + 1)
        mx = np.max(col[lo:hi]) + EPS
        widths.append(np.sum(col[lo:hi] >= 0.7 * mx))
    return float(np.mean(widths))

def spectral_flatness_mean(X_amp: np.ndarray) -> float:
    gm = np.exp(np.mean(np.log(X_amp + EPS), axis=0))
    am = np.mean(X_amp, axis=0) + EPS
    sf = gm / am
    return float(np.mean(sf))

def edge_grad_mean(x_db: np.ndarray) -> float:
    gy = np.abs(np.diff(x_db, axis=0))
    gx = np.abs(np.diff(x_db, axis=1))
    return float(np.mean(gy)) + float(np.mean(gx))

def low_energy(M_db: np.ndarray, thr_lin: float = 1e-4) -> bool:
    return float(np.mean(db_to_amp(M_db))) < thr_lin

def crop_is_vowel_interior(entry: Dict) -> bool:
    spans = entry.get("phone_spans") or []
    if not spans:
        return True  # no phones available => don't block
    dur = float(entry["box_tfr"][1]) - float(entry["box_tfr"][0]) + 1e-9
    best = max(spans, key=lambda s: float(s.get("overlap_sec", 0.0)))
    ovl = float(best.get("overlap_sec", 0.0)) / dur
    lab = (best.get("label") or "").upper()
    is_vowel = bool(re.sub(r"\d$", "", lab) in VOWELS)
    return is_vowel and ovl >= VOWEL_INTERIOR_MIN_OVL

def band_intersects(band_lo_hz: float, band_hi_hz: float, lo: float, hi: float) -> bool:
    return not (hi < band_lo_hz or lo > band_hi_hz)

# ---- band helpers ----
def band_psr_db(X_db: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> float:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 3:
        return 0.0
    X = db_to_amp(X_db[idx, :])
    peak = np.max(X, axis=0) + EPS
    med = np.median(X, axis=0) + EPS
    psr = 20.0 * np.log10(peak / med)
    return float(np.mean(psr))

def band_psr_db_ts(X_db: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> np.ndarray:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 3: return np.zeros((X_db.shape[1],), dtype=float)
    X = db_to_amp(X_db[idx, :])
    peak = np.max(X, axis=0) + EPS
    med  = np.median(X, axis=0) + EPS
    return 20.0 * np.log10(peak / med)

def band_peak_valley_ratio_db(X_amp: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> float:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 6: return 0.0
    sub = X_amp[idx, :]
    peak = np.max(sub, axis=0) + EPS
    valley = np.quantile(sub, 0.25, axis=0) + EPS
    pvr_db = 20.0 * np.log10(peak / valley)
    return float(np.mean(pvr_db))

def band_log_slope_db_per_khz(X_amp: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> float:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 8: return 0.0
    y = 20.0 * np.log10(np.mean(X_amp[idx, :], axis=1) + EPS)
    x = f_axis[idx] / 1000.0
    m, _ = np.polyfit(x, y, 1)
    return float(m)

def broadband_tilt_db_per_khz(X_amp: np.ndarray, f_axis: np.ndarray, lo=300.0, hi=None) -> float:
    if hi is None: hi = float(f_axis[-1])
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 8: return 0.0
    y = 20.0 * np.log10(np.mean(X_amp[idx, :], axis=1) + EPS)
    x = f_axis[idx] / 1000.0
    m, _ = np.polyfit(x, y, 1)
    return float(m)

def _sec_to_local_col(entry: Dict, sr: int, hop: int, s_abs: float) -> int:
    """Map an absolute-time (s) to a column index inside the cropped patch."""
    t0 = float((entry.get("box_tfr") or entry["bbox_sec_hz"])[0])
    col = int(round((s_abs - t0) * sr / hop))
    T = int(entry.get("_local_T", 0))  # we’ll set this in the test
    return max(0, min(T-1, col))

def _v_neighbors(entry: Dict, mid_s: float) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (V1, V2) spans around mid_s from entry['phone_spans'] (best vowel left/right)."""
    spans = sorted(entry.get("phone_spans") or [], key=lambda s: float(s["start"]))
    V1, V2 = None, None
    for s in spans:
        st, en, lab = float(s["start"]), float(s["end"]), s.get("label","")
        if en <= mid_s and _is_vowel(lab):
            if (V1 is None) or (en > float(V1["end"])):
                V1 = s
        if st >= mid_s and _is_vowel(lab):
            if (V2 is None) or (st < float(V2["start"] if V2 else 1e9)):
                V2 = s
    return V1, V2

def _window_end(start: float, end: float, dur_ms: float) -> Tuple[float,float]:
    win0 = max(start, end - dur_ms/1000.0);  win1 = end
    return win0, win1

def _window_start(start: float, end: float, dur_ms: float) -> Tuple[float,float]:
    win0 = start;  win1 = min(end, start + dur_ms/1000.0)
    return win0, win1

def _mean_in_cols(x: np.ndarray, c0: int, c1: int) -> float:
    c0, c1 = int(c0), int(c1)
    if c1 <= c0 + 1: return 0.0
    return float(np.mean(x[c0:c1]))

# ---- extra helpers for hyperneatness / hyperflat prosody ----
def band_proxy_track(X_amp: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> np.ndarray:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 8:
        return np.zeros((X_amp.shape[1],), dtype=float)
    sub = X_amp[idx, :]
    arg = np.argmax(sub, axis=0)
    return f_axis[idx[arg]]

def track_stats(track: np.ndarray) -> Tuple[float, float]:
    if track.size < 3:
        return 0.0, 0.0
    d1 = np.abs(np.diff(track))
    d2 = np.abs(np.diff(track, n=2))
    jitter = float(np.mean(d1)) if d1.size else 0.0
    curve  = float(np.mean(d2)) if d2.size else 0.0
    return jitter, curve

def spectral_centroid_band_ts(X_amp: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> np.ndarray:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 3:
        return np.zeros((X_amp.shape[1],), dtype=float)
    sub = X_amp[idx, :]
    num = (f_axis[idx, None] * sub).sum(axis=0)
    den = sub.sum(axis=0) + EPS
    return num / den

def roughness_residual_ratio(X_db: np.ndarray, k: int = 5) -> float:
    """Simple vertical smoother; compare residual power to total power."""
    F, T = X_db.shape
    X = db_to_amp(X_db)
    pad = np.pad(X, ((k, k), (0, 0)), mode="edge")
    smooth = np.zeros_like(X)
    for f in range(F):
        smooth[f, :] = np.mean(pad[f:f+2*k+1, :], axis=0)
    resid = np.maximum(X - smooth, 0.0)
    return float(np.mean(resid)) / (float(np.mean(X)) + EPS)

def centroid_std_in_window(A_amp: np.ndarray, f_axis: np.ndarray, t0: int, t1: int) -> float:
    if t1 - t0 < 4: return 0.0
    c = spectral_centroid(f_axis, A_amp[:, t0:t1])
    return float(np.std(c))

def count_inflections(x: np.ndarray) -> int:
    if x.size < 5: return 0
    dx = np.diff(x)
    return int(np.sum((dx[1:] * dx[:-1]) < 0))

def best_lag(frA: np.ndarray, frB: np.ndarray, max_lag: int = 6) -> int:
    if frA.size < 6 or frB.size < 6: return 0
    lags = range(-max_lag, max_lag+1)
    def corr_at(l):
        if l >= 0:
            a, b = frA[l:], frB[:frB.size-l]
        else:
            a, b = frA[:frA.size+l], frB[-l:]
        if a.size < 4: return 0.0
        a = (a - a.mean()); b = (b - b.mean())
        den = (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
        return float(np.dot(a, b) / den)
    scores = [corr_at(l) for l in lags]
    return int(list(lags)[int(np.argmax(scores))])

# ---- tests ----
def test_formant_attenuation(entry: Dict, sr: int,
                             A_db: np.ndarray, B_db: np.ndarray,
                             add_crop: np.ndarray, miss_crop: np.ndarray) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    # HARD phone gate: vowel interior
    if not crop_is_class_interior(entry, _is_vowel, VOWEL_INTERIOR_MIN_OVL):
        return {"decision": "absent", "reason": "not_vowel_interior"}
    F = A_db.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)

    band_lo_hz, band_hi_hz = float(entry["box_tfr"][2]), float(entry["box_tfr"][3])
    F1_band, F2_band, F3_band = F1_RANGE_DEFAULT, F2_RANGE_DEFAULT, F3_RANGE_DEFAULT
    band_ok = band_intersects(band_lo_hz, band_hi_hz, *F1_band) or band_intersects(band_lo_hz, band_hi_hz, *F2_band)
    if not band_ok:
        return {"decision": "absent", "reason": "band_not_vowel_like"}

    psrA_F1 = band_psr_db(A_db, f_axis, F1_band)
    psrB_F1 = band_psr_db(B_db, f_axis, F1_band)
    psrA_F2 = band_psr_db(A_db, f_axis, F2_band)
    psrB_F2 = band_psr_db(B_db, f_axis, F2_band)
    psrA_F3 = band_psr_db(A_db, f_axis, F3_band)
    psrB_F3 = band_psr_db(B_db, f_axis, F3_band)

    drop_F1 = psrA_F1 - psrB_F1
    drop_F2 = psrA_F2 - psrB_F2
    drop_F3 = psrA_F3 - psrB_F3
    thr_F1 = max(ATTEN_PSR_DROP_DB_BASE, ATTEN_PSR_DROP_FRAC * max(1e-6, psrA_F1))
    thr_F2 = max(ATTEN_PSR_DROP_DB_BASE, ATTEN_PSR_DROP_FRAC * max(1e-6, psrA_F2))
    thr_F3 = max(ATTEN_PSR_DROP_DB_BASE, ATTEN_PSR_DROP_FRAC * max(1e-6, psrA_F3))

    # quantile over per-frame PSR drops
    dF1_ts = band_psr_db_ts(A_db, f_axis, F1_band) - band_psr_db_ts(B_db, f_axis, F1_band)
    dF2_ts = band_psr_db_ts(A_db, f_axis, F2_band) - band_psr_db_ts(B_db, f_axis, F2_band)
    q_drop = float(np.quantile(np.concatenate([dF1_ts, dF2_ts]), ATTEN_DROP_Q)) if (dF1_ts.size + dF2_ts.size) > 0 else 0.0

    # broadband tilt
    A = db_to_amp(A_db); B = db_to_amp(B_db)
    tiltA = broadband_tilt_db_per_khz(A, f_axis, lo=300.0)
    tiltB = broadband_tilt_db_per_khz(B, f_axis, lo=300.0)
    tilt_more_neg = (tiltB - tiltA) <= -ATTEN_TILT_MORE_NEG_DB_PER_KHZ

    add_frac  = float(np.mean(add_crop  > 0))
    miss_frac = float(np.mean(miss_crop > 0))
    gates = {
        "miss_min_ok": miss_frac >= ATTEN_MISS_MIN,
        "miss_minus_add_ok": (miss_frac - add_frac) >= ATTEN_MISS_MINUS_ADD,
    }
    present_F1 = (drop_F1 >= thr_F1)
    present_F2 = (drop_F2 >= thr_F2)
    present_extra = ((drop_F3 >= thr_F3) or (q_drop >= ATTEN_PSR_DROP_DB_BASE)) and tilt_more_neg

    decision_any = (present_F1 or present_F2 or present_extra) and all(gates.values())

    return {
        "psrA_F1_dB": psrA_F1, "psrB_F1_dB": psrB_F1, "drop_F1_dB": drop_F1, "thr_F1_dB": thr_F1,
        "psrA_F2_dB": psrA_F2, "psrB_F2_dB": psrB_F2, "drop_F2_dB": drop_F2, "thr_F2_dB": thr_F2,
        "psrA_F3_dB": psrA_F3, "psrB_F3_dB": psrB_F3, "drop_F3_dB": drop_F3, "thr_F3_dB": thr_F3,
        "q_drop_dB": q_drop, "tiltA_db_per_khz": tiltA, "tiltB_db_per_khz": tiltB,
        "add_frac": add_frac, "miss_frac": miss_frac,
        "gates": gates,
        "decision": "present" if decision_any else "absent"
    }

def test_fogging_vowel(entry: Dict, sr: int,
                       A_db: np.ndarray, B_db: np.ndarray,
                       add_crop: np.ndarray, miss_crop: np.ndarray) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    interior_ok = crop_is_vowel_interior(entry)
    if not interior_ok:
        return {"decision": "absent", "reason": "not_vowel_interior"}
    band_lo_hz, band_hi_hz = float(entry["box_tfr"][2]), float(entry["box_tfr"][3])
    band_ok = band_intersects(band_lo_hz, band_hi_hz, *F1_RANGE_DEFAULT) or \
              band_intersects(band_lo_hz, band_hi_hz, *F2_RANGE_DEFAULT)

    add_frac  = float(np.mean(add_crop  > 0))
    miss_frac = float(np.mean(miss_crop > 0))
    dir_ok = ((add_frac >= FOG_ADD_MIN) or (miss_frac >= FOG_MISS_MIN)) and \
             ((add_frac + miss_frac) >= FOG_ADD_MISS_SUM_MIN)

    A = db_to_amp(A_db); B = db_to_amp(B_db)
    F = A_db.shape[0]; f_axis = np.linspace(0.0, sr/2.0, F)

    H_A = float(np.mean(entropy_bits(A, axis=0)))
    H_B = float(np.mean(entropy_bits(B, axis=0)))
    K_A = float(np.mean(kurtosis_along_freq(A)))
    K_B = float(np.mean(kurtosis_along_freq(B)))
    W_A = ridge_width_bins(A, sr); W_B = ridge_width_bins(B, sr)
    SF_A = spectral_flatness_mean(A); SF_B = spectral_flatness_mean(B)
    EG_A = edge_grad_mean(A_db);     EG_B = edge_grad_mean(B_db)

    # NEW: formant peak-to-valley contrast
    pvrA1 = band_peak_valley_ratio_db(A, f_axis, F1_RANGE_DEFAULT)
    pvrB1 = band_peak_valley_ratio_db(B, f_axis, F1_RANGE_DEFAULT)
    pvrA2 = band_peak_valley_ratio_db(A, f_axis, F2_RANGE_DEFAULT)
    pvrB2 = band_peak_valley_ratio_db(B, f_axis, F2_RANGE_DEFAULT)
    pvr_drop = max(pvrA1 - pvrB1, pvrA2 - pvrB2)

    votes = 0
    votes += 1 if (H_B - H_A) > FOG_H_DELTA_MIN else 0
    votes += 1 if (K_B - K_A) < FOG_K_DELTA_MAX else 0
    votes += 1 if (W_B - W_A) > FOG_WIDTH_DELTA else 0
    votes += 1 if (SF_B - SF_A) > FOG_FLATNESS_DELTA else 0
    votes += 1 if ((EG_A - EG_B) / (EG_A + EPS)) > FOG_EDGE_DROP_RATIO else 0
    votes += 1 if pvr_drop >= FOG_PVR_DROP_DB else 0  # NEW vote

    atten_like = (miss_frac - add_frac) >= 0.10
    decision = "present" if (band_ok and dir_ok and (votes >= 2) and not atten_like) else "absent"
    return {
        "H_delta": H_B - H_A,
        "K_delta": K_B - K_A,
        "Width_delta_bins": W_B - W_A,
        "SF_delta": SF_B - SF_A,
        "edge_drop_ratio": (EG_A - EG_B) / (EG_A + EPS),
        "PVR_drop_dB": pvr_drop,  # NEW
        "AddFrac": add_frac, "MissFrac": miss_frac,
        "gates": {"interior_ok": interior_ok, "band_ok": band_ok, "dir_ok": dir_ok, "atten_like": atten_like},
        "decision": decision
    }

def test_fogging_fricative(A_db: np.ndarray, B_db: np.ndarray, sr: int, entry: Optional[Dict] = None) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    if entry is not None and not crop_is_class_interior(entry, _is_fric, 0.60):
        return {"decision": "absent", "reason": "not_fricative_interior"}
    A = db_to_amp(A_db); B = db_to_amp(B_db)
    F, _ = A.shape
    f_axis = np.linspace(0.0, sr/2.0, F)
    cA = float(np.mean(spectral_centroid(f_axis, A)))
    cB = float(np.mean(spectral_centroid(f_axis, B)))
    kA = float(np.mean(kurtosis_along_freq(A)))
    kB = float(np.mean(kurtosis_along_freq(B)))
    eA = vertical_edge_strength(A_db)
    eB = vertical_edge_strength(B_db)

    # NEW: high-band slope & temporal sharpness
    hb = (3500.0, max(3500.0, sr/2.0 - 50.0))
    sA = band_log_slope_db_per_khz(A, f_axis, hb)
    sB = band_log_slope_db_per_khz(B, f_axis, hb)
    gxA = temporal_edge_strength(A_db); gxB = temporal_edge_strength(B_db)
    t_edge_drop = (gxA - gxB) / (gxA + EPS)

    slope_vote = 1 if (sB - sA) <= -FRIC_SLOPE_MORE_NEG_DB_PER_KHZ else 0
    time_vote  = 1 if t_edge_drop >= FRIC_TIME_EDGE_DROP_RATIO else 0

    decision = ( ((cA - cB) >= FRIC_CENTROID_DROP_HZ)
               + ((kA - kB) >= FRIC_KURT_DROP)
               + ((eA - eB) / (eA + EPS) >= EDGE_DROP_RATIO)
               + slope_vote + time_vote ) >= 2

    return {
        "centroid_A_Hz": cA, "centroid_B_Hz": cB, "centroid_drop_Hz": cA - cB,
        "kurt_drop": kA - kB,
        "edge_drop_ratio": (eA - eB) / (eA + EPS),
        "hi_band_slope_A": sA, "hi_band_slope_B": sB,
        "time_edge_drop": t_edge_drop,
        "decision": "present" if decision else "absent"
    }

def boundary_step_test(A_amp: np.ndarray, sr: int, tmid: int, halfw: int) -> Dict:
    T = A_amp.shape[1]
    L0 = max(0, tmid - 2*halfw); L1 = max(0, tmid - halfw)
    R0 = min(T, tmid + halfw);   R1 = min(T, tmid + 2*halfw)
    if L1 - L0 < 2 or R1 - R0 < 2:
        return {"dCentroid": 0.0, "dEnergy_db": 0.0}
    F = A_amp.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)
    cL = float(np.mean(spectral_centroid(f_axis, A_amp[:, L0:L1])))
    cR = float(np.mean(spectral_centroid(f_axis, A_amp[:, R0:R1])))
    eL = float(np.mean(20.0 * np.log10(np.sum(A_amp[:, L0:L1], axis=0) + EPS)))
    eR = float(np.mean(20.0 * np.log10(np.sum(A_amp[:, R0:R1], axis=0) + EPS)))
    return {"dCentroid": abs(cR - cL), "dEnergy_db": abs(eR - eL)}

def test_concatenatedness_addmiss(add_crop: np.ndarray, miss_crop: np.ndarray,
                                  A_db: np.ndarray, sr: int, hop: int,
                                  entry: Optional[Dict] = None) -> Dict:
    """
    Concatenation boundary via add/miss masks (no norm_s):
      • Build a union mask U = (add | miss)
      • Boundary Contrast Ratio (BCR): center-vs-sides coverage of U
      • Stripe height: how much of the freq axis is active in the center window
      • Step evidence from Mb: centroid/energy jump and smooth_vs_jump
    """
    if entry is not None and not crop_has_boundary_near_mid(entry, tol_ms=25.0):
        return {"BCR": 0.0, "center_coverage": 0.0, "height_frac": 0.0,
                "dCentroid": 0.0, "dEnergy_db": 0.0, "smooth_vs_jump": 0.0,
                "decision": "absent", "reason": "no_boundary_near_center"}

    if add_crop.size == 0 or miss_crop.size == 0:
        return {"BCR": 0.0, "center_coverage": 0.0, "height_frac": 0.0,
                "dCentroid": 0.0, "dEnergy_db": 0.0, "smooth_vs_jump": 0.0,
                "decision": "absent"}

    U = ((add_crop > 0) | (miss_crop > 0)).astype(np.float32)
    if U.shape[1] < 8:
        return {"BCR": 0.0, "center_coverage": 0.0, "height_frac": 0.0,
                "dCentroid": 0.0, "dEnergy_db": 0.0, "smooth_vs_jump": 0.0,
                "decision": "absent"}

    T = U.shape[1]
    mid = T // 2
    w = max(2, int(round((BCR_WINDOW_MS / 1000.0) * (sr / hop))))

    center = U[:, max(0, mid - w):min(T, mid + w)]
    left  = U[:, max(0, mid - 3*w):max(0, mid - 2*w)]
    right = U[:, min(T, mid + 2*w):min(T, mid + 3*w)]

    B  = float(np.sum(center))
    I1 = float(np.sum(left))
    I2 = float(np.sum(right))
    denom = 0.5 * (I1 + I2) + 1e-9
    BCR = B / denom if denom > 0 else 0.0

    total = float(np.sum(U)) + 1e-9
    center_coverage = B / total

    # stripe height along frequency
    stripe = np.mean(center, axis=1)
    height_frac = float(np.mean(stripe >= 0.75 * np.max(stripe))) if np.max(stripe) > 0 else 0.0

    # optional polarity mix info (not a hard gate, just returned for debugging)
    c_add  = float(np.mean(add_crop[:, max(0, mid - w):min(T, mid + w)]  > 0))
    c_miss = float(np.mean(miss_crop[:, max(0, mid - w):min(T, mid + w)] > 0))
    mix_frac = min(c_add, c_miss)

    # Mb step evidence (same as before)
    A_amp = db_to_amp(A_db)
    d = boundary_step_test(A_amp, sr=sr, tmid=mid, halfw=max(2, w // 2))
    step_ok = (d["dCentroid"] >= STEP_DCENTROID_HZ) or (d["dEnergy_db"] >= STEP_DENERGY_DB)

    # boundary jump vs intra-segment smoothness
    F = A_db.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)
    L0 = max(0, mid - 2*w); L1 = max(0, mid - w)
    R0 = min(T, mid + w);   R1 = min(T, mid + 2*w)
    std_intra = 0.5 * (centroid_std_in_window(A_amp, f_axis, L0, L1) +
                       centroid_std_in_window(A_amp, f_axis, R0, R1))
    smooth_vs_jump = d["dCentroid"] / (std_intra + 1e-6)
    step_ok = step_ok or (smooth_vs_jump >= SMOOTH_JUMP_MIN)

    decision = "present" if (BCR >= BCR_MIN and
                             center_coverage >= BCR_STRIPE_COVERAGE_MIN and
                             height_frac >= BCR_HEIGHT_MIN_FRAC and
                             step_ok) else "absent"

    return {
        "BCR": BCR,
        "center_coverage": center_coverage,
        "height_frac": height_frac,
        "dCentroid": d["dCentroid"],
        "dEnergy_db": d["dEnergy_db"],
        "smooth_vs_jump": smooth_vs_jump,
        "center_add_frac": c_add,
        "center_miss_frac": c_miss,
        "mix_frac": mix_frac,
        "decision": decision
    }

def test_pseudo_formant(add_crop: np.ndarray, fbin_hz: float, tbin_s: float, entry: Optional[Dict] = None) -> Dict:
    if entry is not None and not crop_is_class_interior(entry, _is_appr, 0.60):
        return {"has_component": False, "decision": "absent", "reason": "not_approximant"}
    lab = (add_crop > 0).astype(np.uint8)
    if lab.sum() == 0:
        return {"has_component": False, "decision": "absent"}
    visited = np.zeros_like(lab, dtype=bool)
    F, T = lab.shape
    found = False
    best = {"dur_ms": 0.0, "height_hz": 0.0}
    for f in range(F):
        for t in range(T):
            if lab[f, t] == 0 or visited[f, t]:
                continue
            stack = [(f, t)]
            fmin = fmax = f
            tmin = tmax = t
            visited[f, t] = True
            while stack:
                fi, ti = stack.pop()
                for df, dt in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nf, nt = fi + df, ti + dt
                    if 0 <= nf < F and 0 <= nt < T and not visited[nf, nt] and lab[nf, nt] == 1:
                        visited[nf, nt] = True
                        stack.append((nf, nt))
                        fmin = min(fmin, nf); fmax = max(fmax, nf)
                        tmin = min(tmin, nt); tmax = max(tmax, nt)
            height_hz = (max(1, (fmax - fmin + 1)) - 1) * fbin_hz
            dur_ms = (max(1, (tmax - tmin + 1)) - 1) * tbin_s * 1000.0
            if dur_ms >= PSEUDO_MIN_DUR_MS and height_hz <= PSEUDO_MAX_HEIGHT_HZ:
                found = True
                if dur_ms > best["dur_ms"]:
                    best = {"dur_ms": dur_ms, "height_hz": height_hz}
    add_frac = float(np.mean(add_crop > 0))
    decision = found and (add_frac >= PSEUDO_ADD_FRAC_MIN)
    best.update({"add_frac": add_frac, "has_component": found, "decision": "present" if decision else "absent"})
    return best

def f2_proxy_track(X_amp: np.ndarray, f_axis: np.ndarray, band: Tuple[float,float]) -> np.ndarray:
    lo, hi = band
    idx = np.where((f_axis >= lo) & (f_axis <= hi))[0]
    if idx.size < 8: return np.zeros((X_amp.shape[1],), dtype=float)
    sub = X_amp[idx, :]
    arg = np.argmax(sub, axis=0)
    return f_axis[idx[arg]]

def coart_window_indices(T: int, mid: int, win_ms: float, sr: int, hop: int) -> Tuple[int,int]:
    w = int(round((win_ms/1000.0) * (sr / hop)))
    t0 = max(0, mid - w); t1 = min(T, mid + w)
    return t0, t1

def test_coarticulatory_deficit(Mb_db: np.ndarray, Ms_db: np.ndarray, sr: int, hop: int, entry: Optional[Dict] = None) -> Dict:
    # Energy & boundary gates
    if low_energy(Mb_db):
        return {"decision": "absent", "reason": "low_energy"}
    if entry is not None and not crop_has_boundary_near_mid(entry, tol_ms=COART_WIN_MS/2):
        return {"decision": "absent", "reason": "no_boundary_in_window"}

    # Build true f-axis from the crop Hz span (fixes the original band bug)
    F, T = Mb_db.shape
    if entry is not None and ("box_tfr" in entry or "bbox_sec_hz" in entry):
        t0s, t1s, f0_hz, f1_hz = (entry.get("box_tfr") or entry["bbox_sec_hz"])
        f_axis = np.linspace(float(f0_hz), float(f1_hz), F)
    else:
        f_axis = np.linspace(0.0, sr/2.0, F)

    # Require enough F2 rows inside the crop
    f2_lo, f2_hi = COART_F2_BAND
    f2_rows = np.where((f_axis >= f2_lo) & (f_axis <= f2_hi))[0]
    if f2_rows.size < COART_MIN_F2_ROWS:
        return {
            "decision": "absent", "reason": "no_f2_in_crop",
            "f2_rows_in_crop": int(f2_rows.size),
            "crop_f_lo_hz": float(f_axis[0]), "crop_f_hi_hz": float(f_axis[-1])
        }

    # Time window around the phonetic boundary
    A = db_to_amp(Mb_db); B = db_to_amp(Ms_db)
    mid_col = T // 2
    def _coart_cols(win_ms: float) -> Tuple[int,int]:
        w = int(round((win_ms/1000.0) * (sr / hop)))
        return max(0, mid_col - w), min(T, mid_col + w)

    c0, c1 = _coart_cols(COART_WIN_MS)
    if c1 - c0 < 6:
        return {"decision": "absent", "reason": "short_window"}

    # Store local T so time→col mapping works for V-to-V windows
    if entry is not None:
        entry["_local_T"] = T

    # ---------- (1) Boundary-window slopes/ranges (your original cue, fixed axis) ----------
    f2A_bw = band_proxy_track(A[:, c0:c1], f_axis, COART_F2_BAND)
    f2B_bw = band_proxy_track(B[:, c0:c1], f_axis, COART_F2_BAND)
    sA = float(np.mean(np.abs(np.diff(f2A_bw)))) if f2A_bw.size > 1 else 0.0
    sB = float(np.mean(np.abs(np.diff(f2B_bw)))) if f2B_bw.size > 1 else 0.0
    rA = float(np.max(f2A_bw) - np.min(f2A_bw)) if f2A_bw.size else 0.0
    rB = float(np.max(f2B_bw) - np.min(f2B_bw)) if f2B_bw.size else 0.0
    lag = best_lag(f2A_bw, f2B_bw, max_lag=4)

    gate_bw = {"Mb_slope_min_ok": sA >= COART_MIN_SLOPE_MB}
    present_bw = gate_bw["Mb_slope_min_ok"] \
                 and (sB <= COART_SLOPE_RATIO_MAX * sA) \
                 and (rB <= COART_RANGE_RATIO_MAX * rA) \
                 and (abs(lag) >= 1)

    # ---------- (2) V-to-V cue: ΔF2 late-in-V1 vs early-in-V2 ----------
    has_vv = False
    deltaA = deltaB = 0.0
    if entry is not None and entry.get("phone_spans"):
        mid_s = 0.5*(float(t0s) + float(t1s))
        V1, V2 = _v_neighbors(entry, mid_s)
        if V1 is not None and V2 is not None:
            has_vv = True
            # build F2 proxy tracks for the full crop once
            f2A_full = band_proxy_track(A, f_axis, COART_F2_BAND)
            f2B_full = band_proxy_track(B, f_axis, COART_F2_BAND)

            # late in V1
            v1_0, v1_1 = _window_end(float(V1["start"]), float(V1["end"]), COART_V1_LATE_MS)
            v1c0 = _sec_to_local_col(entry, sr, hop, v1_0); v1c1 = _sec_to_local_col(entry, sr, hop, v1_1)
            # early in V2
            v2_0, v2_1 = _window_start(float(V2["start"]), float(V2["end"]), COART_V2_EARLY_MS)
            v2c0 = _sec_to_local_col(entry, sr, hop, v2_0); v2c1 = _sec_to_local_col(entry, sr, hop, v2_1)

            F2A_v1 = _mean_in_cols(f2A_full, v1c0, v1c1);  F2A_v2 = _mean_in_cols(f2A_full, v2c0, v2c1)
            F2B_v1 = _mean_in_cols(f2B_full, v1c0, v1c1);  F2B_v2 = _mean_in_cols(f2B_full, v2c0, v2c1)

            deltaA = abs(F2A_v2 - F2A_v1)  # Mb separation across V1→V2
            deltaB = abs(F2B_v2 - F2B_v1)  # Ms separation (should be smaller)
    gate_vv = {"Mb_delta_min_ok": (deltaA >= COART_MIN_DELTA_VV_HZ) if has_vv else False}
    present_vv = has_vv and gate_vv["Mb_delta_min_ok"] and (deltaB <= COART_DELTA_RATIO_MAX * deltaA)

    # ---------- Final decision: need boundary cue, OR (preferably) boundary + V-to-V if available ----------
    present = bool(present_bw and (present_vv or not has_vv))

    return {
        # boundary-window summary
        "Mb_slope_Hz_per_fr": sA, "Ms_slope_Hz_per_fr": sB,
        "Mb_range_Hz": rA, "Ms_range_Hz": rB,
        "lag_frames": lag, "gates": {**gate_bw, **gate_vv, "has_vv": has_vv},
        "f2_rows_in_crop": int(f2_rows.size),
        "crop_f_lo_hz": float(f_axis[0]), "crop_f_hi_hz": float(f_axis[-1]),
        # V-to-V summary
        "F2_delta_vv_Mb_Hz": float(deltaA), "F2_delta_vv_Ms_Hz": float(deltaB),
        "delta_ratio_Ms_over_Mb": float(deltaB / (deltaA + EPS)) if deltaA > 0 else 1.0,
        "decision": "present" if present else "absent"
    }

def test_hyperneatness(entry: Dict, sr: int, Mb_db: np.ndarray, Ms_db: np.ndarray) -> Dict:
    if low_energy(Mb_db):
        return {"decision": "absent", "reason": "low_energy"}
    # HARD phone gate: sonorant interior
    if not crop_is_class_interior(entry, _is_sonor, 0.60):
        return {"decision": "absent", "reason": "not_sonorant_interior"}
    F = Mb_db.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)

    ok_band = True
    if "box_tfr" in entry:
        blo, bhi = float(entry["box_tfr"][2]), float(entry["box_tfr"][3])
        ok_band = band_intersects(blo, bhi, *F1_RANGE_DEFAULT) or band_intersects(blo, bhi, *F2_RANGE_DEFAULT)
    if not ok_band:
        return {"decision": "absent", "reason": "band_not_vowel_like"}

    A = db_to_amp(Mb_db); B = db_to_amp(Ms_db)

    # proxy tracks in F1/F2
    f1A = band_proxy_track(A, f_axis, F1_RANGE_DEFAULT); f1B = band_proxy_track(B, f_axis, F1_RANGE_DEFAULT)
    f2A = band_proxy_track(A, f_axis, F2_RANGE_DEFAULT); f2B = band_proxy_track(B, f_axis, F2_RANGE_DEFAULT)

    j1A, c1A = track_stats(f1A); j1B, c1B = track_stats(f1B)
    j2A, c2A = track_stats(f2A); j2B, c2B = track_stats(f2B)

    eA = edge_grad_mean(Mb_db); eB = edge_grad_mean(Ms_db)
    edge_ratio = float(eB / (eA + EPS))

    # NEW: residual roughness proxy
    resA = roughness_residual_ratio(Mb_db, k=5)
    resB = roughness_residual_ratio(Ms_db, k=5)
    res_ratio = float(resB / (resA + EPS))

    jit_ratio_f1 = float(j1B / (j1A + EPS)) if j1A > 0 else 1.0
    cur_ratio_f1 = float(c1B / (c1A + EPS)) if c1A > 0 else 1.0
    jit_ratio_f2 = float(j2B / (j2A + EPS)) if j2A > 0 else 1.0
    cur_ratio_f2 = float(c2B / (c2A + EPS)) if c2A > 0 else 1.0

    jit_ratio = min(jit_ratio_f1, jit_ratio_f2)
    cur_ratio = min(cur_ratio_f1, cur_ratio_f2)

    present = (jit_ratio <= NEAT_JITTER_RATIO_MAX and
               cur_ratio <= NEAT_CURVE_RATIO_MAX and
               edge_ratio <= NEAT_EDGE_RATIO_MAX and
               res_ratio <= NEAT_RESID_RATIO_MAX)

    return {
        "F1_jitter_A": j1A, "F1_jitter_B": j1B, "F1_curv_A": c1A, "F1_curv_B": c1B,
        "F2_jitter_A": j2A, "F2_jitter_B": j2B, "F2_curv_A": c2A, "F2_curv_B": c2B,
        "jitter_ratio_min": jit_ratio, "curv_ratio_min": cur_ratio, "edge_ratio": edge_ratio,
        "residual_ratio": res_ratio,
        "decision": "present" if present else "absent"
    }

def test_hyperflat_prosody(Mb_db: np.ndarray, Ms_db: np.ndarray, sr: int, hop: int, entry: Optional[Dict] = None) -> Dict:
    if low_energy(Ms_db):
        return {"decision": "absent", "reason": "low_energy"}
    if entry is not None and crop_voiced_fraction(entry) < 0.60:
        return {"decision": "absent", "reason": "unvoiced_region"}
    T = Ms_db.shape[1]
    dur_ms = 1000.0 * (T * hop / float(sr))
    if dur_ms < FLAT_MIN_DUR_MS:
        return {"decision": "absent", "reason": "short_crop"}

    A = db_to_amp(Mb_db); B = db_to_amp(Ms_db)
    F = B.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)

    # amplitude envelope (broadband)
    eA = np.sum(A, axis=0); eB = np.sum(B, axis=0)
    cvA = float(np.std(eA) / (np.mean(eA) + EPS))
    cvB = float(np.std(eB) / (np.mean(eB) + EPS))

    # low-frequency "pitch region" centroid variance
    cA = spectral_centroid_band_ts(A, f_axis, LOWF0_BAND)
    cB = spectral_centroid_band_ts(B, f_axis, LOWF0_BAND)
    cstdA = float(np.std(cA)); cstdB = float(np.std(cB))

    # NEW: inflection count (turns in contour)
    infA = count_inflections(cA); infB = count_inflections(cB)
    inf_ok = (infB <= FLAT_INFLECT_RATIO_MAX * max(1, infA))

    cv_ok   = (cvB <= FLAT_ENERGY_CV_ABS_MAX) or (cvB <= FLAT_ENERGY_CV_RATIO_MAX * cvA)
    cstd_ok = (cstdB <= FLAT_LOW_CENTROID_STD_HZ_MAX) or (cstdB <= FLAT_LOW_CENTROID_STD_RATIO_MAX * cstdA)

    present = bool(cv_ok and cstd_ok and inf_ok)

    return {
        "dur_ms": dur_ms,
        "energy_cv_A": cvA, "energy_cv_B": cvB,
        "lowband_centroid_std_A_Hz": cstdA, "lowband_centroid_std_B_Hz": cstdB,
        "inflections_A": infA, "inflections_B": infB, "inflection_ok": inf_ok,
        "cv_ok": cv_ok, "centroid_ok": cstd_ok,
        "decision": "present" if present else "absent"
    }

# ---- confidence calculator ----
def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _pos_margin(val: float, thr: float) -> float:
    """For criteria of the form val >= thr."""
    denom = abs(thr) + EPS
    return _clip01((val - thr) / denom)

def _neg_margin(val: float, thr: float) -> float:
    """For criteria of the form val <= thr."""
    denom = abs(thr) + EPS
    return _clip01((thr - val) / denom)

def compute_confidence(concept: str, res: Dict) -> float:
    """
    Turn satisfied-test strength into a confidence in [0,1], without changing decisions.
    Uses margins vs thresholds and (when applicable) the count of satisfied cues.
    """
    c = concept
    r = res

    if c == "formant_attenuation":
        mF1 = _pos_margin(r.get("drop_F1_dB", 0.0), r.get("thr_F1_dB", 1.0))
        mF2 = _pos_margin(r.get("drop_F2_dB", 0.0), r.get("thr_F2_dB", 1.0))
        mF3 = _pos_margin(r.get("drop_F3_dB", 0.0), r.get("thr_F3_dB", 1.0)) if "thr_F3_dB" in r else 0.0
        mQ  = _pos_margin(r.get("q_drop_dB", 0.0), ATTEN_PSR_DROP_DB_BASE)
        # tilt margin: Ms more negative than Mb by threshold
        mTilt = _pos_margin((r.get("tiltA_db_per_khz", 0.0) - r.get("tiltB_db_per_khz", 0.0)), ATTEN_TILT_MORE_NEG_DB_PER_KHZ)
        mm = _pos_margin(r.get("miss_frac", 0.0), ATTEN_MISS_MIN)
        mma = _pos_margin(r.get("miss_frac", 0.0) - r.get("add_frac", 0.0), ATTEN_MISS_MINUS_ADD)
        gate_strength = min(1.0, (mm + mma) / 2.0)
        peak_strength = max(mF1, mF2, mF3, mQ)
        return _clip01(0.6 * peak_strength + 0.2 * mTilt + 0.2 * gate_strength)

    if c == "fogging_vowel":
        mH  = _pos_margin(r.get("H_delta", 0.0), FOG_H_DELTA_MIN)
        mK  = _neg_margin(r.get("K_delta", 0.0), FOG_K_DELTA_MAX)
        mW  = _pos_margin(r.get("Width_delta_bins", 0.0), FOG_WIDTH_DELTA)
        mSF = _pos_margin(r.get("SF_delta", 0.0), FOG_FLATNESS_DELTA)
        mE  = _pos_margin(r.get("edge_drop_ratio", 0.0), FOG_EDGE_DROP_RATIO)
        mPVR = _pos_margin(r.get("PVR_drop_dB", 0.0), FOG_PVR_DROP_DB)
        vote_margins = [mH, mK, mW, mSF, mE, mPVR]
        vote_strength = float(np.mean([v for v in vote_margins if v > 0])) if any(vote_margins) else 0.0
        addf = r.get("AddFrac", 0.0); missf = r.get("MissFrac", 0.0)
        m_dir_any = max(_pos_margin(addf, FOG_ADD_MIN), _pos_margin(missf, FOG_MISS_MIN))
        m_dir_sum = _pos_margin(addf + missf, FOG_ADD_MISS_SUM_MIN)
        dir_strength = min(1.0, 0.5 * m_dir_any + 0.5 * m_dir_sum)
        return _clip01(0.7 * vote_strength + 0.3 * dir_strength)

    if c == "fogging_fricative":
        mC = _pos_margin(r.get("centroid_drop_Hz", 0.0), FRIC_CENTROID_DROP_HZ)
        mK = _pos_margin(r.get("kurt_drop", 0.0), FRIC_KURT_DROP)
        mE = _pos_margin(r.get("edge_drop_ratio", 0.0), EDGE_DROP_RATIO)
        # NEW margins
        mS = _pos_margin((r.get("hi_band_slope_A", 0.0) - r.get("hi_band_slope_B", 0.0)), FRIC_SLOPE_MORE_NEG_DB_PER_KHZ)
        mTx = _pos_margin(r.get("time_edge_drop", 0.0), FRIC_TIME_EDGE_DROP_RATIO)
        mets = [mC, mK, mE, mS, mTx]
        num_met = int(np.sum([m > 0 for m in mets]))
        avg_margin = float(np.mean([m for m in mets if m > 0])) if num_met else 0.0
        return _clip01(0.6 * avg_margin + 0.4 * (num_met / 5.0))

    if c == "pseudo_formant":
        dur = r.get("dur_ms", 0.0)
        ht  = r.get("height_hz", 0.0)
        add = r.get("add_frac", 0.0)
        m_dur = _pos_margin(dur, PSEUDO_MIN_DUR_MS)
        m_ht  = _neg_margin(ht, PSEUDO_MAX_HEIGHT_HZ)
        m_add = _pos_margin(add, PSEUDO_ADD_FRAC_MIN)
        # optional: steady-track margin if available
        fr = r.get("f2_range_ratio", None)
        m_steady = _neg_margin(fr, PSEUDO_RANGE_RATIO_MAX) if fr is not None else 0.0
        return _clip01(float(np.mean([m_dur, m_ht, m_add, max(0.0, m_steady)])))

    if c == "concatenatedness":
        mBCR   = _pos_margin(r.get("BCR", 0.0), BCR_MIN)
        mCov   = _pos_margin(r.get("center_coverage", 0.0), BCR_STRIPE_COVERAGE_MIN)
        mHfrac = _pos_margin(r.get("height_frac", 0.0), BCR_HEIGHT_MIN_FRAC)
        mStepC = _pos_margin(r.get("dCentroid", 0.0), STEP_DCENTROID_HZ)
        mStepE = _pos_margin(r.get("dEnergy_db", 0.0), STEP_DENERGY_DB)
        mStep  = max(mStepC, mStepE)
        mSmooth = _pos_margin(r.get("smooth_vs_jump", 0.0), SMOOTH_JUMP_MIN) if "smooth_vs_jump" in r else 0.0
        return _clip01(float(np.mean([mBCR, mCov, mHfrac, max(mStep, mSmooth)])))

    if c == "coarticulatory_deficit":
    sA = r.get("Mb_slope_Hz_per_fr", 0.0)
    sB = r.get("Ms_slope_Hz_per_fr", 0.0)
    rA = r.get("Mb_range_Hz", 0.0)
    rB = r.get("Ms_range_Hz", 0.0)
    m_gate   = _pos_margin(sA, COART_MIN_SLOPE_MB)
    m_sratio = _neg_margin(sB, COART_SLOPE_RATIO_MAX * max(sA, EPS))
    m_rratio = _neg_margin(rB, COART_RANGE_RATIO_MAX * max(rA, EPS))
    m_lag    = _pos_margin(abs(r.get("lag_frames", 0)), 1.0)

    # V-to-V margins if available
    if r.get("gates", {}).get("has_vv", False):
        dA = r.get("F2_delta_vv_Mb_Hz", 0.0)
        dB = r.get("F2_delta_vv_Ms_Hz", 0.0)
        m_dgate  = _pos_margin(dA, COART_MIN_DELTA_VV_HZ)
        m_dratio = _neg_margin(dB, COART_DELTA_RATIO_MAX * max(dA, EPS))
        m_band   = 1.0 if r.get("f2_rows_in_crop", 0) >= COART_MIN_F2_ROWS else 0.0
        return _clip01(0.20*m_gate + 0.20*m_sratio + 0.15*m_rratio + 0.15*m_lag + 0.15*m_dgate + 0.15*m_dratio + 0.0*m_band)
    else:
        m_band = 1.0 if r.get("f2_rows_in_crop", 0) >= COART_MIN_F2_ROWS else 0.0
        return _clip01(0.30*m_gate + 0.30*m_sratio + 0.25*m_rratio + 0.15*m_lag + 0.0*m_band)

    if c == "hyperneatness":
        m_jit = _neg_margin(r.get("jitter_ratio_min", 1.0), NEAT_JITTER_RATIO_MAX)
        m_cur = _neg_margin(r.get("curv_ratio_min", 1.0),   NEAT_CURVE_RATIO_MAX)
        m_edge = _neg_margin(r.get("edge_ratio", 1.0),      NEAT_EDGE_RATIO_MAX)
        m_res  = _neg_margin(r.get("residual_ratio", 1.0),  NEAT_RESID_RATIO_MAX) if "residual_ratio" in r else 0.0
        return _clip01(float(np.mean([m_jit, m_cur, m_edge, m_res])))

    if c == "hyperflat_prosody":
        cvA = r.get("energy_cv_A", 0.0); cvB = r.get("energy_cv_B", 0.0)
        cA  = r.get("lowband_centroid_std_A_Hz", 0.0)
        cB  = r.get("lowband_centroid_std_B_Hz", 0.0)
        m_cv_abs = _neg_margin(cvB, FLAT_ENERGY_CV_ABS_MAX)
        m_cv_rel = _neg_margin(cvB, FLAT_ENERGY_CV_RATIO_MAX * max(cvA, EPS))
        m_cv = max(m_cv_abs, m_cv_rel)
        m_cstd_abs = _neg_margin(cB, FLAT_LOW_CENTROID_STD_HZ_MAX)
        m_cstd_rel = _neg_margin(cB, FLAT_LOW_CENTROID_STD_RATIO_MAX * max(cA, EPS))
        m_cstd = max(m_cstd_abs, m_cstd_rel)
        dur_ms = r.get("dur_ms", 0.0)
        m_dur = _clip01((dur_ms - FLAT_MIN_DUR_MS) / max(FLAT_MIN_DUR_MS, 1.0))
        # NEW: inflection margin via ratio
        infA = r.get("inflections_A", 0); infB = r.get("inflections_B", 0)
        inf_ratio = (infB / max(1.0, float(infA))) if infA or infB else 1.0
        m_inf = _neg_margin(inf_ratio, FLAT_INFLECT_RATIO_MAX)
        return _clip01(0.35 * m_cv + 0.35 * m_cstd + 0.20 * m_inf + 0.10 * m_dur)

    return 0.0

# ---- routing ----
def route_and_test(entry: Dict, sr: int, hop: int,
                   arrs_full: Dict[str, np.ndarray],
                   global_norm_p80: float) -> Dict:
    box_idx = entry.get("bbox_idx") or entry.get("box_index")
    if not box_idx:
        raise ValueError("Crop entry missing bbox_idx/box_index")

    sub = crop_arrays(arrs_full, box_idx)
    Mb = sub["Mb_db"]; Ms = sub["Ms_db"]
    add = sub["add"];   miss = sub["miss"];   norm = sub["norm"]
    F, _ = Mb.shape
    fbin_hz = hz_per_bin(sr, F); tbin_s = hop / sr

    if "box_tfr" not in entry and "bbox_sec_hz" in entry:
        t0, t1, f0, f1 = entry["bbox_sec_hz"]
        entry["box_tfr"] = [t0, t1, f0, f1]

    tests = {
        "fogging_vowel":          test_fogging_vowel(entry, sr, Mb, Ms, add, miss),
        "formant_attenuation":    test_formant_attenuation(entry, sr, Mb, Ms, add, miss),
        "fogging_fricative":      test_fogging_fricative(Mb, Ms, sr, entry),
        "pseudo_formant":         test_pseudo_formant(add, fbin_hz, tbin_s, entry),
        "concatenatedness":       test_concatenatedness_addmiss(add, miss, Mb, sr, hop, entry),
        "coarticulatory_deficit": test_coarticulatory_deficit(Mb, Ms, sr, hop, entry),
        "hyperneatness":          test_hyperneatness(entry, sr, Mb, Ms),
        "hyperflat_prosody":      test_hyperflat_prosody(Mb, Ms, sr, hop, entry),
    }

    # Optional tightening for pseudo-formant: require steady-track in Ms (low F2 range vs Mb)
    if tests["pseudo_formant"]["decision"] == "present":
        A_amp = db_to_amp(Mb); B_amp = db_to_amp(Ms)
        f_axis = np.linspace(0.0, sr/2.0, F)
        f2A = f2_proxy_track(A_amp, f_axis, F2_RANGE_DEFAULT)
        f2B = f2_proxy_track(B_amp, f_axis, F2_RANGE_DEFAULT)
        rA = float(np.max(f2A) - np.min(f2A)) if f2A.size else 0.0
        rB = float(np.max(f2B) - np.min(f2B)) if f2B.size else 0.0
        ratio = float(rB / (rA + EPS)) if rA > 0 else 1.0
        tests["pseudo_formant"]["f2_range_ratio"] = ratio
        tests["pseudo_formant"]["steady_track_flag"] = bool(ratio <= PSEUDO_RANGE_RATIO_MAX)
        if not tests["pseudo_formant"]["steady_track_flag"]:
            tests["pseudo_formant"]["decision"] = "absent"
            tests["pseudo_formant"]["reason"] = "no_steady_track"

    # Attach confidences
    for cname, tres in tests.items():
        if tres.get("decision") == "present":
            conf = compute_confidence(cname, tres)
            tres["confidence"] = float(conf)
            tres["confidence_pct"] = float(100.0 * conf)

    add_miss_sum = float(np.mean((add > 0) | (miss > 0)))
    crop_norm_p90 = float(np.quantile(norm[np.isfinite(norm)], 0.90)) if norm.size else 0.0
    decisions = [k for k, v in tests.items() if v.get("decision") == "present"]

    if len(decisions) == 0 and (add_miss_sum < NOISE_MIN_ADD_MISS_SUM or entry.get("non_speech", False)) and (crop_norm_p90 < global_norm_p80):
        label = "noise_or_uncertain"
    elif len(decisions) == 0:
        label = "no_concept_detected"
    else:
        label = ",".join(sorted(decisions))

    return {
        "crop_id": entry.get("crop_id"),
        "bbox_idx": box_idx,
        "bbox_sec_hz": entry.get("bbox_sec_hz") or entry.get("box_tfr"),
        "derived": {
            "mask_coverage": add_miss_sum,
            "crop_norm_p90": crop_norm_p90,
            "global_norm_p80": global_norm_p80
        },
        "tests": tests,
        "final_label": label
    }

# ---- top-level API ----
def analyze_manifest(manifest_path: Path, array_roots: List[Path]) -> List[Dict]:
    """
    Pure function: loads manifest, arrays, runs tests, returns list of per-crop results.
    No file writes.
    """
    with open(manifest_path, "r") as f:
        j = json.load(f)
    if isinstance(j, dict) and "crops" in j:
        meta = j
        crops = j["crops"]
    else:
        # legacy list of crops (supported)
        crops = j
        parts = manifest_path.parts
        meta = {
            "bona_id": parts[-3],
            "vocoder": parts[-2],
            "spoof_stem": f"{parts[-2]}_{parts[-3]}",
            "stft": {"sr": 16000, "hop": 256}
        }

    pair_dir = find_pair_dir(meta["bona_id"], meta["vocoder"], [manifest_path.parents[2]])  # try sibling by default
    try:
        # If arrays are elsewhere, caller should re-route using their own roots & find_pair_dir
        pair_dir = find_pair_dir(meta["bona_id"], meta["vocoder"], [pair_dir.parent])
    except Exception:
        pass  # fall through; caller can supply locate path

    results = []
    return results  # overridden by analyze_manifest_with_roots below

def analyze_manifest_with_roots(manifest_path: Path, array_roots: List[Path]) -> List[Dict]:
    with open(manifest_path, "r") as f:
        j = json.load(f)
    if isinstance(j, dict) and "crops" in j:
        meta = j
        crops = j["crops"]
    else:
        crops = j
        parts = manifest_path.parts
        meta = {
            "bona_id": parts[-3],
            "vocoder": parts[-2],
            "spoof_stem": f"{parts[-2]}_{parts[-3]}",
            "stft": {"sr": 16000, "hop": 256}
        }

    # arrays location
    pair_dir = find_pair_dir(meta["bona_id"], meta["vocoder"], array_roots)
    arrs_full = load_pair_arrays(pair_dir, meta["spoof_stem"])
    norm_full = arrs_full["norm"]
    global_norm_p80 = float(np.quantile(norm_full[np.isfinite(norm_full)], NOISE_GLOBAL_PCT)) if norm_full.size else 0.0

    sr = int(meta["stft"]["sr"])
    hop = int(meta["stft"]["hop"])

    # attach phone spans from TextGrid (fail-open if missing)
    phones = _load_textgrid_phones(meta["bona_id"])
    for c in crops:
        if "box_tfr" not in c and "bbox_sec_hz" in c:
            t0, t1, f0, f1 = c["bbox_sec_hz"]
            c["box_tfr"] = [t0, t1, f0, f1]
        _attach_phone_spans(c, phones)

    return [route_and_test(entry=c, sr=sr, hop=hop, arrs_full=arrs_full, global_norm_p80=global_norm_p80) for c in crops]