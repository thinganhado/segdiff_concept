#!/usr/bin/env python3
# Aggregate all K crops using concept_results.json (preferred) or suggested_crops.json (fallback).
# Outputs:
#   ./suggested_crops/<BONA>/<VOCODER>/coverage_all_k.json
#   ./suggested_crops/dataset_catalog_all_k.json
#   ./suggested_crops/_GLOBAL/reference_sheet.json  (NEW: best overall exemplar per concept)

import json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------- config ----------
CONCEPTS = [
    "fogging_vowel",
    "formant_attenuation",
    "fogging_fricative",
    "pseudo_formant",
    "concatenatedness",
    "coarticulatory_deficit",  # NEW
    "hyperneatness",           # NEW
    "hyperflat_prosody",       # NEW
]

DEFAULT_HI_THR = 0.70
DEFAULT_LO_THR = 0.50
TOPN_PAIR_DEFAULT = 10
TOPN_PER_CONCEPT_DEFAULT = 5
TOPN_DATASET_DEFAULT = 20

# ---------- io utils ----------
def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, payload: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(p)

def parse_k(dir_name: str) -> Optional[int]:
    m = re.match(r"^k(\d+)$", dir_name.strip().lower())
    return int(m.group(1)) if m else None

# ---------- real-confidence helpers (from concept_results) ----------
def clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def safe(v, d=0.0):
    try: return float(v)
    except Exception: return d

# --- Concept-specific confidence (fallback math if detect_concepts didn't attach "confidence") ---
def conf_formant_attenuation(td: Dict) -> float:
    d1 = safe(td.get("drop_F1_dB")); t1 = max(safe(td.get("thr_F1_dB"), 1e-6), 1e-6)
    d2 = safe(td.get("drop_F2_dB")); t2 = max(safe(td.get("thr_F2_dB"), 1e-6), 1e-6)
    c1 = clip01(d1 / t1)
    c2 = clip01(d2 / t2)
    conf = max(c1, c2)
    miss = safe(td.get("miss_frac")); add = safe(td.get("add_frac"))
    if (miss - add) < 0:
        conf *= 0.6
    return conf if (td.get("decision") == "present") else 0.0

def conf_fogging_vowel(td: Dict) -> float:
    H_margin = safe(td.get("H_delta")) - 0.08         # FOG_H_DELTA_MIN
    K_margin = -(safe(td.get("K_delta")) - (-0.30))   # drop desired
    W_margin = safe(td.get("Width_delta_bins")) - 2.0
    cH = clip01(H_margin / 0.10)
    cK = clip01(K_margin / 0.10)
    cW = clip01(W_margin / 2.0)
    gates = td.get("gates", {})
    dir_ok = bool(gates.get("dir_ok", False))
    interior_band_ok = bool(gates.get("interior_ok", False) and gates.get("band_ok", False))
    conf = (cH + cK + cW) / 3.0
    if not dir_ok: conf *= 0.5
    if not interior_band_ok: conf = 0.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_fogging_fricative(td: Dict) -> float:
    rC = clip01(safe(td.get("centroid_drop_Hz")) / 200.0)
    rK = clip01(safe(td.get("kurt_drop"))       / 0.20)
    rE = clip01(safe(td.get("edge_drop_ratio")) / 0.10)
    conf = (rC + rK + rE) / 3.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_concatenatedness(td: Dict) -> float:
    cB = clip01(safe(td.get("BCR")) / 2.0)
    cC = clip01(safe(td.get("center_coverage")) / 0.60)
    cH = clip01(safe(td.get("height_frac")) / 0.60)
    # We ignore the boundary step here (already part of present/absent), but real detect_concepts adds it.
    conf = (cB + cC + cH) / 3.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_pseudo_formant(td: Dict) -> float:
    d = safe(td.get("dur_ms")); h = safe(td.get("height_hz")); a = safe(td.get("add_frac"))
    cD = clip01(d / 40.0)
    cH = clip01((400.0 - min(h, 400.0)) / 400.0)
    cA = clip01(a / 0.12)
    conf = (cD + cH + cA) / 3.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_coarticulatory_deficit(td: Dict) -> float:
    # Gates & margins mirroring detect_concepts defaults
    sA = safe(td.get("Mb_slope_Hz_per_fr"))
    sB = safe(td.get("Ms_slope_Hz_per_fr"))
    rA = safe(td.get("Mb_range_Hz"))
    rB = safe(td.get("Ms_range_Hz"))
    # gate: sA >= 50
    m_gate = clip01((sA - 50.0) / 50.0)
    # ratio margins (smaller is better)
    s_ratio = sB / max(sA, 1e-9)
    r_ratio = rB / max(rA, 1e-9)
    m_sratio = clip01((0.60 - s_ratio) / 0.60)
    m_rratio = clip01((0.70 - r_ratio) / 0.70)
    conf = (m_gate + m_sratio + m_rratio) / 3.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_hyperneatness(td: Dict) -> float:
    # smaller ratios are better; thresholds 0.60, 0.60, 0.85
    jr = safe(td.get("jitter_ratio_min"), 1.0)
    cr = safe(td.get("curv_ratio_min"),   1.0)
    er = safe(td.get("edge_ratio"),       1.0)
    m_j = clip01((0.60 - jr) / 0.60)
    m_c = clip01((0.60 - cr) / 0.60)
    m_e = clip01((0.85 - er) / 0.85)
    conf = (m_j + m_c + m_e) / 3.0
    return conf if (td.get("decision") == "present") else 0.0

def conf_hyperflat_prosody(td: Dict) -> float:
    # energy CV abs<=0.10 or rel<=0.60*cvA; centroid std abs<=25 or rel<=0.60*cstdA; small dur bonus
    cvA = safe(td.get("energy_cv_A")); cvB = safe(td.get("energy_cv_B"))
    cA  = safe(td.get("lowband_centroid_std_A_Hz")); cB = safe(td.get("lowband_centroid_std_B_Hz"))
    dur = safe(td.get("dur_ms"))
    m_cv_abs = clip01((0.10 - cvB) / 0.10)
    m_cv_rel = clip01((0.60 * max(cvA, 1e-9) - cvB) / max(0.60 * max(cvA, 1e-9), 1e-6))
    m_cv = max(m_cv_abs, m_cv_rel)
    m_cs_abs = clip01((25.0 - cB) / 25.0)
    m_cs_rel = clip01((0.60 * max(cA, 1e-9) - cB) / max(0.60 * max(cA, 1e-9), 1e-6))
    m_cs = max(m_cs_abs, m_cs_rel)
    m_dur = clip01((dur - 200.0) / 200.0)  # soft bonus once > min dur (200ms)
    conf = 0.45 * m_cv + 0.45 * m_cs + 0.10 * m_dur
    return conf if (td.get("decision") == "present") else 0.0

CONF_FUN = {
    "formant_attenuation": conf_formant_attenuation,
    "fogging_vowel":       conf_fogging_vowel,
    "fogging_fricative":   conf_fogging_fricative,
    "concatenatedness":    conf_concatenatedness,
    "pseudo_formant":      conf_pseudo_formant,
    "coarticulatory_deficit": conf_coarticulatory_deficit,  # NEW
    "hyperneatness":          conf_hyperneatness,           # NEW
    "hyperflat_prosody":      conf_hyperflat_prosody,       # NEW
}

def test_confidence(name: str, td: Dict) -> float:
    fn = CONF_FUN.get(name)
    if not fn: return 0.0
    try:
        # Prefer detect_concepts-attached "confidence"
        return float(td.get("confidence")) if "confidence" in td else fn(td)
    except Exception:
        return 0.0

# ---------- proxy/fallback (from suggested_crops.json) ----------
def pick_metric_name(crop: Dict) -> Optional[str]:
    if "score_sum_norm" in crop and crop["score_sum_norm"] is not None:
        return "score_sum_norm"
    if "score_area" in crop and crop["score_area"] is not None:
        return "score_area"
    if "area_bins" in crop and crop["area_bins"] is not None:
        return "area_bins"
    return None

def minmax_conf(vals: List[float]) -> List[float]:
    if not vals: return []
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin + 1e-12:
        return [1.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]

# ---------- gather hits ----------
def gather_from_concept_results(crops_sweep_root: Path) -> Dict[Tuple[str,str], Dict]:
    """
    Returns {(bona,voc): {'stft':{}, 'hits': [ {concept, crop_id, bbox_idx, bbox_sec_hz, confidence, confidence_pct, K} ... ]}}
    """
    pairs: Dict[Tuple[str,str], Dict] = {}
    for kdir in sorted(p for p in crops_sweep_root.iterdir() if p.is_dir()):
        K = parse_k(kdir.name)
        if K is None: continue
        for bona_dir in sorted(p for p in kdir.iterdir() if p.is_dir()):
            bona = bona_dir.name
            for voc_dir in sorted(p for p in bona_dir.iterdir() if p.is_dir()):
                voc = voc_dir.name
                crops_dir = voc_dir / "crops"
                cr = crops_dir / "concept_results.json"
                if not cr.exists():
                    continue
                try:
                    results = load_json(cr)
                except Exception:
                    continue
                # Try to grab stft meta from crops_manifest.json
                stft = {}
                cm = crops_dir / "crops_manifest.json"
                if cm.exists():
                    try:
                        j = load_json(cm)
                        stft = j.get("stft") or {}
                    except Exception:
                        pass
                key = (bona, voc)
                if key not in pairs:
                    pairs[key] = {"stft": stft, "hits": []}
                for r in results:
                    label = r.get("final_label", "")
                    if label in ("no_concept_detected", "noise_or_uncertain"):
                        continue
                    tests = r.get("tests") or {}
                    for name in CONCEPTS:
                        td = tests.get(name)
                        if not td or td.get("decision") != "present":
                            continue
                        conf = test_confidence(name, td)
                        if conf <= 0:
                            continue
                        pairs[key]["hits"].append({
                            "concept": name,
                            "crop_id": r.get("crop_id"),
                            "bbox_idx": r.get("bbox_idx"),
                            "bbox_sec_hz": r.get("bbox_sec_hz"),
                            "confidence": float(conf),
                            "confidence_pct": round(float(conf) * 100.0, 1),
                            "K": K
                        })
    return pairs

def gather_from_suggested(suggested_root: Path) -> Dict[Tuple[str,str], Dict]:
    pairs: Dict[Tuple[str,str], Dict] = {}
    for spath in suggested_root.rglob("suggested_crops.json"):
        try:
            j = load_json(spath)
        except Exception:
            continue
        bona = j.get("bona_id"); voc = j.get("vocoder")
        stft = j.get("stft", {})
        crops = j.get("selected_crops") or []
        # Prefer per-crop concept_confidences if present, else proxy normalize a ranking score
        metric = None
        for c in crops:
            metric = pick_metric_name(c)
            if metric: break
        proxy_scores = [float(c.get(metric, 0.0)) if metric else 1.0 for c in crops]
        proxy_confs = minmax_conf(proxy_scores)
        hits = []
        for idx, c in enumerate(crops):
            # if concept_confidences present, use those; else use proxy_confs[idx]
            per_conf: Dict[str, float] = {}
            if "concept_confidences" in c and isinstance(c["concept_confidences"], dict):
                for nm, v in c["concept_confidences"].items():
                    try:
                        per_conf[nm] = float(v)
                    except Exception:
                        continue
            tests = c.get("present_tests") or list(per_conf.keys())
            for name in tests:
                if name not in CONCEPTS: 
                    continue
                conf = per_conf.get(name, proxy_confs[idx])
                hits.append({
                    "concept": name,
                    "crop_id": c.get("crop_id"),
                    "bbox_idx": c.get("bbox_idx"),
                    "bbox_sec_hz": c.get("bbox_sec_hz"),
                    "confidence": float(conf),
                    "confidence_pct": round(float(conf)*100.0, 1),
                    "K": c.get("K")  # may be None in suggested files
                })
        key = (bona, voc)
        if key not in pairs:
            pairs[key] = {"stft": stft, "hits": []}
        pairs[key]["hits"].extend(hits)
    return pairs

def merge_hits(primary: Dict[Tuple[str,str], Dict], fallback: Dict[Tuple[str,str], Dict]) -> Dict[Tuple[str,str], Dict]:
    out = dict(primary)
    for key, meta in fallback.items():
        if key not in out:
            out[key] = meta
        else:
            out[key]["hits"].extend(meta.get("hits", []))
            if not out[key].get("stft"):
                out[key]["stft"] = meta.get("stft", {})
    return out

def dedupe_by_bbox_concept(hits: List[Dict], score_key: str = "confidence") -> List[Dict]:
    best: Dict[Tuple[Tuple[int,int,int,int], str], Dict] = {}
    for h in hits:
        bb = tuple(int(x) for x in h["bbox_idx"])
        key = (bb, h["concept"])
        prev = best.get(key)
        if prev is None or float(h.get(score_key, 0.0)) > float(prev.get(score_key, 0.0)):
            best[key] = h
    return list(best.values())

# ---------- summaries ----------
def per_pair_summaries(pairs: Dict[Tuple[str,str], Dict],
                       hi_thr=DEFAULT_HI_THR, lo_thr=DEFAULT_LO_THR,
                       topn_pair_overall: int = TOPN_PAIR_DEFAULT,
                       topn_pair_per_concept: int = TOPN_PER_CONCEPT_DEFAULT) -> Dict[Tuple[str,str], Dict]:
    out: Dict[Tuple[str,str], Dict] = {}
    for (bona, voc), meta in pairs.items():
        hits_all = dedupe_by_bbox_concept(list(meta.get("hits", [])), "confidence")
        by_concept_counts = {c: 0 for c in CONCEPTS}
        for h in hits_all:
            by_concept_counts[h["concept"]] += 1
        top_overall = sorted(hits_all, key=lambda z: z["confidence"], reverse=True)[:topn_pair_overall]
        per_concept: Dict[str, Dict] = {}
        for c in CONCEPTS:
            ch = [h for h in hits_all if h["concept"] == c]
            ch_sorted = sorted(ch, key=lambda z: z["confidence"], reverse=True)
            max_conf = ch_sorted[0]["confidence"] if ch_sorted else 0.0
            num_hi = sum(1 for h in ch if h["confidence"] >= hi_thr)
            num_lo = sum(1 for h in ch if lo_thr <= h["confidence"] < hi_thr)
            per_concept[c] = {
                "max_conf_pct": round(max_conf * 100.0, 1),
                "num_hi": int(num_hi),
                "num_lo": int(num_lo),
                "topN": ch_sorted[:topn_pair_per_concept],
                "reference": (ch_sorted[0] if ch_sorted else None),  # explicit best-per-concept (per pair)
            }
        out[(bona, voc)] = {
            "bona_id": bona,
            "vocoder": voc,
            "stft": meta.get("stft", {}),
            "confidence": {
                "type": "real_or_proxy",
                "note": "Uses real PASS-derived confidence when concept_results.json is present; "
                        "falls back to concept_confidences in suggested_crops.json, then to normalized ranks.",
                "hi_thr": hi_thr, "lo_thr": lo_thr,
            },
            "counts": {"total_hits": len(hits_all), "by_concept": by_concept_counts},
            "top_overall": top_overall,
            "per_concept": per_concept,
        }
    return out

def dataset_catalog(pairs_out: Dict[Tuple[str,str], Dict],
                    hi_thr=DEFAULT_HI_THR, lo_thr=DEFAULT_LO_THR,
                    topn_dataset: int = TOPN_DATASET_DEFAULT) -> Tuple[Dict, Dict]:
    """
    Returns:
      cat: overall coverage + top exemplars
      ref_sheet: best single exemplar per concept across the dataset
    """
    exemplars: Dict[str, List[Dict]] = {c: [] for c in CONCEPTS}
    for (bona, voc), summ in pairs_out.items():
        for c in CONCEPTS:
            for h in summ["per_concept"][c]["topN"]:
                exemplars[c].append({
                    "bona_id": bona,
                    "vocoder": voc,
                    "concept": c,
                    "confidence_pct": h["confidence"] * 100.0,
                    "K": h.get("K"),
                    "crop_id": h.get("crop_id"),
                    "bbox_idx": h.get("bbox_idx"),
                    "bbox_sec_hz": h.get("bbox_sec_hz"),
                })
    cat = {"total_pairs": len(pairs_out), "concepts": {}, "missing_concepts": []}
    ref_sheet: Dict[str, Optional[Dict]] = {c: None for c in CONCEPTS}
    for c in CONCEPTS:
        xs = sorted(exemplars[c], key=lambda z: z["confidence_pct"], reverse=True)
        pair_keys_with_c = set((x["bona_id"], x["vocoder"]) for x in xs)
        pairs_any = len(pair_keys_with_c)
        # hi/lo counts by max per pair
        pair_max: Dict[Tuple[str,str], float] = {}
        for x in xs:
            key = (x["bona_id"], x["vocoder"])
            pair_max[key] = max(pair_max.get(key, 0.0), float(x["confidence_pct"]) / 100.0)
        pairs_hi = sum(1 for v in pair_max.values() if v >= hi_thr)
        pairs_lo = sum(1 for v in pair_max.values() if lo_thr <= v < hi_thr)

        top_exemplars = xs[:topn_dataset]
        best = xs[0] if xs else None
        ref_sheet[c] = best

        cat["concepts"][c] = {
            "pairs_any": pairs_any,
            "pairs_lo": pairs_lo,
            "pairs_hi": pairs_hi,
            "found": pairs_hi > 0,
            "top_exemplars": top_exemplars,
        }
        if pairs_hi == 0:
            cat["missing_concepts"].append(c)
    return cat, ref_sheet

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Catalog concepts across Ks (prefer concept_results, fallback to suggested).")
    ap.add_argument("--crops-sweep-root", type=str, default="./crops_sweep",
                    help="Root with k*/<BONA>/<VOCODER>/crops/concept_results.json")
    ap.add_argument("--suggested-root", type=str, default="./suggested_crops",
                    help="Fallback root with <BONA>/<VOCODER>/suggested_crops.json")
    ap.add_argument("--out-root", type=str, default="./suggested_crops",
                    help="Where to write coverage_all_k.json, dataset_catalog_all_k.json, and _GLOBAL/reference_sheet.json")
    ap.add_argument("--hi-thr", type=float, default=DEFAULT_HI_THR)
    ap.add_argument("--lo-thr", type=float, default=DEFAULT_LO_THR)
    ap.add_argument("--topn-pair", type=int, default=TOPN_PAIR_DEFAULT)
    ap.add_argument("--topn-pair-concept", type=int, default=TOPN_PER_CONCEPT_DEFAULT)
    ap.add_argument("--topn-dataset", type=int, default=TOPN_DATASET_DEFAULT)
    args = ap.parse_args()

    crops_sweep_root = Path(args.crops_sweep_root)
    suggested_root = Path(args.suggested_root)
    out_root = Path(args.out_root)

    # 1) Prefer real confidences
    real_pairs = gather_from_concept_results(crops_sweep_root)
    # 2) Fallback (only for pairs without concept_results)
    proxy_pairs = gather_from_suggested(suggested_root)
    pairs = merge_hits(real_pairs, proxy_pairs)

    # 3) per-pair coverage
    summaries = per_pair_summaries(
        pairs,
        hi_thr=args.hi_thr,
        lo_thr=args.lo_thr,
        topn_pair_overall=args.topn_pair,
        topn_pair_per_concept=args.topn_pair_concept,
    )
    for (bona, voc), payload in summaries.items():
        save_json(out_root / bona / voc / "coverage_all_k.json", payload)

    # 4) dataset catalog + global reference sheet
    cat, ref_sheet = dataset_catalog(
        summaries,
        hi_thr=args.hi_thr,
        lo_thr=args.lo_thr,
        topn_dataset=args.topn_dataset,
    )
    save_json(out_root / "dataset_catalog_all_k.json", cat)
    save_json(out_root / "_GLOBAL" / "reference_sheet.json", ref_sheet)

if __name__ == "__main__":
    main()