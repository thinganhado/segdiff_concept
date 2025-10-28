#!/usr/bin/env python3
# Detect PASS-style concepts on crops listed in ./crops/<BONA>/<VOCODER>/crops_manifest.json
import argparse, json, re, glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------- STFT mapping (must match creation) ----------
EPS = 1e-12

# ---------- thresholds (your tuned defaults + a few new ones) ----------
# Fogging (vowels)
VOWEL_INTERIOR_MIN_OVL = 0.60  # only used if phone_spans present
FOG_ADD_MIN = 0.08
FOG_MISS_MIN = 0.08
FOG_ADD_MISS_SUM_MIN = 0.26
FOG_H_DELTA_MIN = 0.08
FOG_K_DELTA_MAX = -0.30
FOG_WIDTH_DELTA = 2.0
FOG_FLATNESS_DELTA = 0.05
FOG_EDGE_DROP_RATIO = 0.10

# Vowel bands
F1_RANGE_DEFAULT = (250.0, 1000.0)
F2_RANGE_DEFAULT = (1000.0, 3000.0)

# Fricatives (fogging)
FRIC_CENTROID_DROP_HZ = 200.0
FRIC_KURT_DROP = 0.20
EDGE_DROP_RATIO = 0.10

# Formant attenuation
ATTEN_PSR_DROP_DB_BASE = 1.5
ATTEN_PSR_DROP_FRAC = 0.15
ATTEN_MISS_MIN = 0.12
ATTEN_MISS_MINUS_ADD = 0.05

# Concatenatedness
BCR_MIN = 2.0
BCR_WINDOW_MS = 24.0
BCR_STRIPE_COVERAGE_MIN = 0.60
BCR_HEIGHT_MIN_FRAC = 0.60
STEP_DCENTROID_HZ = 120.0
STEP_DENERGY_DB = 1.5

# Pseudo-formant
PSEUDO_MIN_DUR_MS = 40.0
PSEUDO_MAX_HEIGHT_HZ = 400.0
PSEUDO_ADD_FRAC_MIN = 0.12

# Coartic deficit
COART_F2_BAND = (1000.0, 3500.0)
COART_WIN_MS = 60.0
COART_MIN_SLOPE_MB = 50.0
COART_SLOPE_RATIO_MAX = 0.6
COART_RANGE_RATIO_MAX = 0.7

# Noise routing
NOISE_MIN_ADD_MISS_SUM = 0.15
NOISE_GLOBAL_PCT = 0.80

# ---------- utils ----------
def hz_per_bin(sr: int, n_freq_bins: int) -> float:
    return (sr / 2.0) / max(1, n_freq_bins - 1)

def frames_to_sec(sr: int, hop: int, fr: int) -> float:
    return fr * hop / sr

def sec_to_frames(sr: int, hop: int, sec: float) -> int:
    return int(round(sec * sr / hop))

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

# ---------- phone helpers (optional; manifest may not carry phones) ----------
def crop_is_vowel_interior(entry: Dict) -> bool:
    spans = entry.get("phone_spans") or []
    if not spans:
        # If no phone info, don't block detection on this gate
        return True
    dur = float(entry["box_tfr"][1]) - float(entry["box_tfr"][0]) + 1e-9
    best = max(spans, key=lambda s: float(s.get("overlap_sec", 0.0)))
    ovl = float(best.get("overlap_sec", 0.0)) / dur
    lab = (best.get("label") or "").upper()
    is_vowel = bool(re.sub(r"\d$", "", lab) in {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"})
    return is_vowel and ovl >= VOWEL_INTERIOR_MIN_OVL

def band_intersects(band_lo_hz: float, band_hi_hz: float, lo: float, hi: float) -> bool:
    return not (hi < band_lo_hz or lo > band_hi_hz)

# ---------- file helpers ----------
def load_pair_arrays(pair_dir: Path, spoof_stem: str) -> Dict[str, np.ndarray]:
    paths = {
        "Mb_db": pair_dir / "Mb_smooth.npy",
        "Ms_db": pair_dir / f"Ms_smooth__{spoof_stem}.npy",
        "add":   pair_dir / f"mask95_add_smoothed__{spoof_stem}.npy",
        "miss":  pair_dir / f"mask95_miss_smoothed__{spoof_stem}.npy",
        "norm":  pair_dir / f"norm_s__{spoof_stem}.npy",
    }
    arrs = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing {k} at {p}")
        arrs[k] = np.load(p)
    return arrs

def crop_arrays(arrs: Dict[str, np.ndarray], box_idx: List[int]) -> Dict[str, np.ndarray]:
    t0, t1, f0, f1 = [int(v) for v in box_idx]
    out = {}
    for k, a in arrs.items():
        if a.ndim != 2:  # all are 2D here
            continue
        out[k] = a[f0:f1, t0:t1]
    return out

# ---------- tests ----------
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

def test_formant_attenuation(entry: Dict, sr: int,
                             A_db: np.ndarray, B_db: np.ndarray,
                             add_crop: np.ndarray, miss_crop: np.ndarray) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    F = A_db.shape[0]
    f_axis = np.linspace(0.0, sr/2.0, F)

    # choose bands: try vowel-dependent if provided, else defaults
    band_lo_hz, band_hi_hz = float(entry["box_tfr"][2]), float(entry["box_tfr"][3])
    F1_band, F2_band = F1_RANGE_DEFAULT, F2_RANGE_DEFAULT
    # gate to "vowel-like" band
    band_ok = band_intersects(band_lo_hz, band_hi_hz, *F1_band) or band_intersects(band_lo_hz, band_hi_hz, *F2_band)
    if not band_ok:
        return {"decision": "absent", "reason": "band_not_vowel_like"}

    psrA_F1 = band_psr_db(A_db, f_axis, F1_band)
    psrB_F1 = band_psr_db(B_db, f_axis, F1_band)
    psrA_F2 = band_psr_db(A_db, f_axis, F2_band)
    psrB_F2 = band_psr_db(B_db, f_axis, F2_band)

    drop_F1 = psrA_F1 - psrB_F1
    drop_F2 = psrA_F2 - psrB_F2
    thr_F1 = max(ATTEN_PSR_DROP_DB_BASE, ATTEN_PSR_DROP_FRAC * max(1e-6, psrA_F1))
    thr_F2 = max(ATTEN_PSR_DROP_DB_BASE, ATTEN_PSR_DROP_FRAC * max(1e-6, psrA_F2))

    add_frac  = float(np.mean(add_crop  > 0))
    miss_frac = float(np.mean(miss_crop > 0))
    gates = {
        "miss_min_ok": miss_frac >= ATTEN_MISS_MIN,
        "miss_minus_add_ok": (miss_frac - add_frac) >= ATTEN_MISS_MINUS_ADD,
    }
    present_F1 = (drop_F1 >= thr_F1) and all(gates.values())
    present_F2 = (drop_F2 >= thr_F2) and all(gates.values())

    return {
        "psrA_F1_dB": psrA_F1, "psrB_F1_dB": psrB_F1, "drop_F1_dB": drop_F1, "thr_F1_dB": thr_F1,
        "psrA_F2_dB": psrA_F2, "psrB_F2_dB": psrB_F2, "drop_F2_dB": drop_F2, "thr_F2_dB": thr_F2,
        "add_frac": add_frac, "miss_frac": miss_frac,
        "gates": gates,
        "decision": "present" if (present_F1 or present_F2) else "absent"
    }

def test_fogging_vowel(entry: Dict, sr: int,
                       A_db: np.ndarray, B_db: np.ndarray,
                       add_crop: np.ndarray, miss_crop: np.ndarray) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    # Gates
    interior_ok = crop_is_vowel_interior(entry)  # True if no phones in entry
    band_lo_hz, band_hi_hz = float(entry["box_tfr"][2]), float(entry["box_tfr"][3])
    band_ok = band_intersects(band_lo_hz, band_hi_hz, *F1_RANGE_DEFAULT) or \
              band_intersects(band_lo_hz, band_hi_hz, *F2_RANGE_DEFAULT)

    add_frac  = float(np.mean(add_crop  > 0))
    miss_frac = float(np.mean(miss_crop > 0))
    dir_ok = ((add_frac >= FOG_ADD_MIN) or (miss_frac >= FOG_MISS_MIN)) and \
             ((add_frac + miss_frac) >= FOG_ADD_MISS_SUM_MIN)

    # Stats
    A = db_to_amp(A_db); B = db_to_amp(B_db)
    H_A = float(np.mean(entropy_bits(A, axis=0)))
    H_B = float(np.mean(entropy_bits(B, axis=0)))
    K_A = float(np.mean(kurtosis_along_freq(A)))
    K_B = float(np.mean(kurtosis_along_freq(B)))
    W_A = ridge_width_bins(A, sr); W_B = ridge_width_bins(B, sr)
    SF_A = spectral_flatness_mean(A); SF_B = spectral_flatness_mean(B)
    EG_A = edge_grad_mean(A_db);     EG_B = edge_grad_mean(B_db)

    votes = 0
    votes += 1 if (H_B - H_A) > FOG_H_DELTA_MIN else 0
    votes += 1 if (K_B - K_A) < FOG_K_DELTA_MAX else 0
    votes += 1 if (W_B - W_A) > FOG_WIDTH_DELTA else 0
    votes += 1 if (SF_B - SF_A) > FOG_FLATNESS_DELTA else 0
    votes += 1 if ((EG_A - EG_B) / (EG_A + EPS)) > FOG_EDGE_DROP_RATIO else 0

    atten_like = (miss_frac - add_frac) >= 0.10
    decision = "present" if (interior_ok and band_ok and dir_ok and (votes >= 2) and not atten_like) else "absent"
    return {
        "H_delta": H_B - H_A,
        "K_delta": K_B - K_A,
        "Width_delta_bins": W_B - W_A,
        "SF_delta": SF_B - SF_A,
        "edge_drop_ratio": (EG_A - EG_B) / (EG_A + EPS),
        "AddFrac": add_frac, "MissFrac": miss_frac,
        "gates": {"interior_ok": interior_ok, "band_ok": band_ok, "dir_ok": dir_ok, "atten_like": atten_like},
        "decision": decision
    }

def test_fogging_fricative(A_db: np.ndarray, B_db: np.ndarray, sr: int) -> Dict:
    if low_energy(A_db): return {"decision": "absent", "reason": "low_energy"}
    A = db_to_amp(A_db); B = db_to_amp(B_db)
    F, _ = A.shape
    f_axis = np.linspace(0.0, sr/2.0, F)
    cA = float(np.mean(spectral_centroid(f_axis, A)))
    cB = float(np.mean(spectral_centroid(f_axis, B)))
    kA = float(np.mean(kurtosis_along_freq(A)))
    kB = float(np.mean(kurtosis_along_freq(B)))
    eA = vertical_edge_strength(A_db)
    eB = vertical_edge_strength(B_db)
    decision = ((cA - cB) >= FRIC_CENTROID_DROP_HZ) + ((kA - kB) >= FRIC_KURT_DROP) + ((eA - eB) / (eA + EPS) >= EDGE_DROP_RATIO) >= 2
    return {
        "centroid_A_Hz": cA, "centroid_B_Hz": cB, "centroid_drop_Hz": cA - cB,
        "kurt_drop": kA - kB,
        "edge_drop_ratio": (eA - eB) / (eA + EPS),
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

def test_concatenatedness(norm_crop: np.ndarray, A_db: np.ndarray, sr: int, hop: int) -> Dict:
    if norm_crop.size == 0 or norm_crop.shape[1] < 8:
        return {"BCR": 0.0, "center_coverage": 0.0, "height_frac": 0.0, "dCentroid": 0.0, "dEnergy_db": 0.0, "decision": "absent"}
    T = norm_crop.shape[1]
    mid = T // 2
    w = max(2, int(round((BCR_WINDOW_MS/1000.0) * (1.0 / (hop/ (sr))))))  # frames in window

    center = norm_crop[:, max(0, mid - w):min(T, mid + w)]
    left   = norm_crop[:, max(0, mid - 3*w):max(0, mid - 2*w)]
    right  = norm_crop[:, min(T, mid + 2*w):min(T, mid + 3*w)]

    B  = float(np.sum(center))
    I1 = float(np.sum(left)); I2 = float(np.sum(right))
    denom = 0.5 * (I1 + I2) + 1e-9
    BCR = B / denom if denom > 0 else 0.0

    total = float(np.sum(norm_crop)) + 1e-9
    center_coverage = B / total
    stripe = np.mean(center, axis=1)
    height_frac = float(np.mean(stripe >= 0.75 * np.max(stripe))) if np.max(stripe) > 0 else 0.0

    d = boundary_step_test(db_to_amp(A_db), sr=sr, tmid=mid, halfw=max(2, w//2))
    step_ok = (d["dCentroid"] >= STEP_DCENTROID_HZ) or (d["dEnergy_db"] >= STEP_DENERGY_DB)

    decision = "present" if (BCR >= BCR_MIN and center_coverage >= BCR_STRIPE_COVERAGE_MIN and height_frac >= BCR_HEIGHT_MIN_FRAC and step_ok) else "absent"
    return {"BCR": BCR, "center_coverage": center_coverage, "height_frac": height_frac, "dCentroid": d["dCentroid"], "dEnergy_db": d["dEnergy_db"], "decision": decision}

def test_pseudo_formant(add_crop: np.ndarray, fbin_hz: float, tbin_s: float) -> Dict:
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
    w = sec_to_frames(sr, hop, win_ms / 1000.0)
    t0 = max(0, mid - w); t1 = min(T, mid + w)
    return t0, t1

def test_coarticulatory_deficit(Mb_db: np.ndarray, Ms_db: np.ndarray, sr: int, hop: int) -> Dict:
    if low_energy(Mb_db): return {"decision": "absent", "reason": "low_energy"}
    A = db_to_amp(Mb_db); B = db_to_amp(Ms_db)
    F, T = A.shape
    f_axis = np.linspace(0.0, sr/2.0, F)
    mid = T // 2
    t0, t1 = coart_window_indices(T, mid, COART_WIN_MS, sr, hop)
    if t1 - t0 < 6:
        return {"decision": "absent", "reason": "short_window"}

    f2A = f2_proxy_track(A[:, t0:t1], f_axis, COART_F2_BAND)
    f2B = f2_proxy_track(B[:, t0:t1], f_axis, COART_F2_BAND)

    sA = np.mean(np.abs(np.diff(f2A))) if f2A.size > 1 else 0.0
    sB = np.mean(np.abs(np.diff(f2B))) if f2B.size > 1 else 0.0
    rA = float(np.max(f2A) - np.min(f2A)) if f2A.size else 0.0
    rB = float(np.max(f2B) - np.min(f2B)) if f2B.size else 0.0

    gates = {"Mb_slope_min_ok": sA >= COART_MIN_SLOPE_MB}
    present = gates["Mb_slope_min_ok"] and (sB <= COART_SLOPE_RATIO_MAX * sA) and (rB <= COART_RANGE_RATIO_MAX * rA)

    return {"Mb_slope_Hz_per_fr": sA, "Ms_slope_Hz_per_fr": sB, "Mb_range_Hz": rA, "Ms_range_Hz": rB, "gates": gates, "decision": "present" if present else "absent"}

# ---------- routing ----------
def route_and_test(entry: Dict, sr: int, hop: int,
                   arrs_full: Dict[str, np.ndarray],
                   global_norm_p80: float) -> Dict:
    # support both new (bbox_idx) and legacy (box_index) keys
    box_idx = entry.get("bbox_idx") or entry.get("box_index")
    if not box_idx:
        raise ValueError("Crop entry missing bbox_idx/box_index")

    sub = crop_arrays(arrs_full, box_idx)
    Mb = sub["Mb_db"]; Ms = sub["Ms_db"]
    add = sub["add"];   miss = sub["miss"];   norm = sub["norm"]
    F, _ = Mb.shape
    fbin_hz = hz_per_bin(sr, F); tbin_s = hop / sr

    # synthetic 'band' fields for gates if not present
    if "box_tfr" not in entry and "bbox_sec_hz" in entry:
        t0, t1, f0, f1 = entry["bbox_sec_hz"]
        entry["box_tfr"] = [t0, t1, f0, f1]

    tests = {}
    # Vowel-like fogging + attenuation
    tests["fogging_vowel"] = test_fogging_vowel(entry, sr, Mb, Ms, add, miss)
    tests["formant_attenuation"] = test_formant_attenuation(entry, sr, Mb, Ms, add, miss)
    # Fricative fogging (always computed; you can gate by band if desired)
    tests["fogging_fricative"] = test_fogging_fricative(Mb, Ms, sr)
    # Pseudo-formant (approximants) – we can't phone-gate here; leave as generic add-component check
    tests["pseudo_formant"] = test_pseudo_formant(add, fbin_hz, tbin_s)
    # Concatenatedness on norm + boundary step on Mb
    tests["concatenatedness"] = test_concatenatedness(norm, Mb, sr, hop)
    # Coarticulatory deficit
    tests["coarticulatory_deficit"] = test_coarticulatory_deficit(Mb, Ms, sr, hop)

    # light routing like your original
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

# ---------- manifest + pair lookup ----------
def find_pair_dir(bona_id: str, vocoder: str, roots: List[Path]) -> Path:
    for r in roots:
        cand = r / bona_id / vocoder
        if (cand / "Mb_smooth.npy").exists():
            return cand
    raise FileNotFoundError(f"Could not find arrays folder for {bona_id}/{vocoder} under roots {roots}")

def analyze_manifest(manifest_path: Path, array_roots: List[Path], overwrite: bool) -> List[Dict]:
    with open(manifest_path, "r") as f:
        j = json.load(f)
    if isinstance(j, dict) and "crops" in j:
        meta = j
        crops = j["crops"]
    else:
        # legacy list of crops shape (not expected here, but supported)
        crops = j
        # try to parse from file path
        parts = manifest_path.parts
        meta = {
            "bona_id": parts[-3],  # .../crops/<BONA>/<VOCODER>/crops_manifest.json
            "vocoder": parts[-2],
            "spoof_stem": f"{parts[-2]}_{parts[-3]}",
            "stft": {"sr": 16000, "hop": 256}
        }

    out_dir = manifest_path.parent
    out_path = out_dir / "concept_results.json"
    if out_path.exists() and not overwrite:
        with open(out_path, "r") as f:
            return json.load(f)

    if not crops:
        with open(out_path, "w") as f:
            json.dump([], f, indent=2)
        return []

    pair_dir = find_pair_dir(meta["bona_id"], meta["vocoder"], array_roots)
    arrs_full = load_pair_arrays(pair_dir, meta["spoof_stem"])
    norm_full = arrs_full["norm"]
    global_norm_p80 = float(np.quantile(norm_full[np.isfinite(norm_full)], NOISE_GLOBAL_PCT)) if norm_full.size else 0.0

    sr = int(meta["stft"]["sr"])
    hop = int(meta["stft"]["hop"])

    results = [route_and_test(entry=c, sr=sr, hop=hop, arrs_full=arrs_full, global_norm_p80=global_norm_p80) for c in crops]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

def collect_paths(single_manifest: str = None, pattern: str = None) -> List[Path]:
    paths = []
    if single_manifest:
        paths.append(Path(single_manifest))
    if pattern:
        for p in glob.glob(pattern, recursive=True):
            if p.endswith("crops_manifest.json"):
                paths.append(Path(p))
    # dedupe
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Detect PASS concepts on crops listed in crops_manifest.json.")
    ap.add_argument("--manifest", type=str, default=None, help="Path to a single crops_manifest.json")
    ap.add_argument("--glob", type=str, default="./crops/**/crops_manifest.json", help="Glob for many manifests")
    ap.add_argument("--array-roots", type=str, default="./mask_outputs_vocv4",
                    help="Colon- or comma-separated roots that contain <BONA>/<VOCODER>/Mb_smooth.npy etc.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite concept_results.json if it exists")
    args = ap.parse_args()

    roots = [Path(s) for s in re.split(r"[:,]", args.array_roots) if s.strip()]

    paths = collect_paths(args.manifest, args.glob if not args.manifest else None)
    if not paths:
        print("No manifest path(s). Use --manifest or --glob")
        return

    for mp in paths:
        print(f"[info] analyzing {mp}")
        try:
            analyze_manifest(mp, array_roots=roots, overwrite=args.overwrite)
            print(f"[ok] wrote {mp.parent / 'concept_results.json'}")
        except Exception as e:
            print(f"[error] {mp}: {e}")

if __name__ == "__main__":
    main()