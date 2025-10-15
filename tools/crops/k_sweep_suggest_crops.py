#!/usr/bin/env python3
# Sweep K, build dynamic crops from norm_s or mask .npy (no fixed windows),
# run concept detection, keep only detected crops, dedupe across K, and write:
#   ./suggested_crops/<BONA>/<VOCODER>/suggested_crops.json
# Also write per-K concept summary:
#   ./suggested_crops/<BONA>/<VOCODER>/k_summary.json
#
# Enhancements:
# - carry per-concept confidences from detect_concepts
# - confidence-aware dedupe
# - diversity-first selection with caps per concept
# - per-concept reference (highest-confidence crop per concept)

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any

import extract_mask_crops as emc
import detect_concepts as dc

# ---- safe JSON for NumPy scalars/arrays ----
import json as _json
try:
    import numpy as _np
except Exception:
    _np = None

class _NumpyJSONEncoder(_json.JSONEncoder):
    def default(self, o):
        if _np is not None:
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, _np.generic):
                return o.item()
        return super().default(o)

def _dump_json_safe(fp, payload):
    _json.dump(payload, fp, indent=2, cls=_NumpyJSONEncoder)

# ---------- helpers ----------
def parse_k_range(s: str) -> List[int]:
    s = s.strip()
    if re.match(r"^\d+[-:.]{1,2}\d+$", s):
        lo, hi = re.split(r"[-:.]{1,2}", s)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in re.split(r"[,\s]+", s) if x]

def parse_globs(s: str) -> List[str]:
    return [g.strip() for g in re.split(r"[,\s]+", s or "") if g.strip()]

def present_tests(result: Dict) -> List[str]:
    return [k for k, v in (result.get("tests") or {}).items() if v.get("decision") == "present"]

def concept_confidences(result: Dict) -> Dict[str, float]:
    """Return {concept: confidence in [0,1]} for present concepts (0.0 if missing)."""
    out: Dict[str, float] = {}
    tests = result.get("tests") or {}
    for cname, tres in tests.items():
        if tres.get("decision") == "present":
            conf = tres.get("confidence")
            if conf is None:
                # tolerant to older detect_concepts: compute on the fly if available
                try:
                    conf = float(dc.compute_confidence(cname, tres))
                except Exception:
                    conf = 0.0
            out[cname] = float(conf)
    return out

def primary_concept_info(result: Dict) -> Tuple[Optional[str], float]:
    """Return (best_concept, best_confidence)."""
    cc = concept_confidences(result)
    if not cc:
        return None, 0.0
    best = max(cc.items(), key=lambda kv: kv[1])
    return best[0], float(best[1])

def bbox_key(bbox_idx: List[int]) -> Tuple[int,int,int,int]:
    t0, t1, f0, f1 = [int(v) for v in bbox_idx]
    return (t0, t1, f0, f1)

def choose_better(a: Dict, b: Dict) -> Dict:
    """
    Prefer:
    1) more present tests,
    2) higher best per-concept confidence,
    3) higher score_sum_norm (if both have it),
    4) higher score_area,
    5) higher rank (smaller number) as gentle tie-breaker.
    """
    def score_pkg(x: Dict) -> Tuple[int, float, float, float, float]:
        n_present = len(x.get("present_tests", []) or [])
        best_conf = float(x.get("primary_confidence", 0.0))
        sum_norm = x.get("score_sum_norm")
        sum_norm = float(sum_norm) if sum_norm is not None else -1.0
        area = float(x.get("score_area", 0.0))
        # lower rank value is better; invert to keep "higher is better" ordering
        inv_rank = -float(x.get("rank", 1e9)) if x.get("rank") is not None else -1e9
        return (n_present, best_conf, sum_norm, area, inv_rank)

    sa, sb = score_pkg(a), score_pkg(b)
    return b if sb > sa else a

def filter_detected_only(manifest_path: Path, results: List[Dict]) -> List[Dict]:
    with open(manifest_path, "r") as f:
        j = json.load(f)
    crops = j["crops"] if isinstance(j, dict) and "crops" in j else j
    by_key = {tuple(c["bbox_idx"]): c for c in crops}

    detected: List[Dict] = []
    for r in results:
        label = r.get("final_label", "")
        if label in ("no_concept_detected", "noise_or_uncertain"):
            continue
        meta = by_key.get(tuple(r["bbox_idx"]), {})

        # confidences
        conf_map = concept_confidences(r)
        p_concept, p_conf = primary_concept_info(r)

        detected.append({
            "crop_id": r.get("crop_id"),
            "bbox_idx": r.get("bbox_idx"),
            "bbox_sec_hz": r.get("bbox_sec_hz"),
            "rank": meta.get("rank"),
            "score_sum_norm": meta.get("score_sum_norm"),
            "score_area": meta.get("score_area", 0.0),
            "area_bins": meta.get("area_bins", 0),
            "present_tests": present_tests(r),
            "concept_confidences": conf_map,       # {concept: confidence [0,1]}
            "primary_concept": p_concept,          # best concept by confidence
            "primary_confidence": p_conf,          # scalar [0,1]
            "final_label": label
        })
    return detected

def dedupe_across_k(crops: List[Dict]) -> List[Dict]:
    best: Dict[Tuple[int,int,int,int], Dict] = {}
    for c in crops:
        key = bbox_key(c["bbox_idx"])
        best[key] = c if key not in best else choose_better(best[key], c)
    return list(best.values())

def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _dump_json_safe(f, payload)
    print(path)

def write_concept_results(manifest_path: Path, results: List[Dict]) -> Path:
    out_path = manifest_path.parent / "concept_results.json"  # same folder as crops_manifest.json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        _dump_json_safe(f, results)
    print(out_path)
    return out_path

def summarize_detected_per_k(detected: List[Dict]) -> Dict:
    by_concept: Dict[str, int] = {}
    conf_mean: Dict[str, float] = {}
    conf_max: Dict[str, float] = {}
    conf_accum: Dict[str, List[float]] = {}

    for item in detected:
        cmap = item.get("concept_confidences", {}) or {}
        for c, conf in cmap.items():
            by_concept[c] = by_concept.get(c, 0) + 1
            conf_accum.setdefault(c, []).append(float(conf))

    for c, arr in conf_accum.items():
        if arr:
            conf_mean[c] = float(sum(arr) / len(arr))
            conf_max[c] = float(max(arr))
        else:
            conf_mean[c] = 0.0
            conf_max[c] = 0.0

    return {"num_detected": len(detected), "by_concept": by_concept,
            "confidence_mean": conf_mean, "confidence_max": conf_max}

def rglob_many(root: Path, patterns: Iterable[str]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if p.suffix.lower() == ".npy" and p not in seen:
                out.append(p); seen.add(p)
    return sorted(out)

def run_detection(manifest_path: Path, array_roots: List[Path]) -> List[Dict]:
    # Use analyze_manifest_with_roots if available, else fallback to analyze_manifest.
    # Be tolerant to whether the function supports `overwrite` or not.
    if hasattr(dc, "analyze_manifest_with_roots"):
        try:
            return dc.analyze_manifest_with_roots(manifest_path, array_roots=array_roots, overwrite=True)
        except TypeError:
            # older signature without overwrite
            return dc.analyze_manifest_with_roots(manifest_path, array_roots=array_roots)
    # fallback to legacy function
    if hasattr(dc, "analyze_manifest"):
        try:
            return dc.analyze_manifest(manifest_path, overwrite=True)
        except TypeError:
            return dc.analyze_manifest(manifest_path)
    raise RuntimeError("detect_concepts has neither analyze_manifest_with_roots nor analyze_manifest")

# ---------- selection / diversity ----------
def select_diverse_high_conf(crops: List[Dict],
                             target: int,
                             per_concept_cap: int,
                             min_conf: float) -> List[Dict]:
    """
    Greedy round-robin across concepts by descending confidence, with:
      - per-concept cap,
      - min confidence threshold,
      - fill any remaining with next best overall above min_conf.
    Uses `primary_concept` / `primary_confidence`.
    """
    # filter by min confidence and valid primary concept
    eligible = [c for c in crops if (c.get("primary_concept") and c.get("primary_confidence", 0.0) >= min_conf)]
    if not eligible:
        return []

    # group by primary concept and sort each group by primary_confidence desc
    by_concept: Dict[str, List[Dict]] = {}
    for c in eligible:
        by_concept.setdefault(c["primary_concept"], []).append(c)
    for grp in by_concept.values():
        grp.sort(key=lambda x: (float(x.get("primary_confidence", 0.0)),
                                float(x.get("score_sum_norm") or -1.0),
                                float(x.get("score_area", 0.0))), reverse=True)

    selected: List[Dict] = []
    taken_per_concept: Dict[str, int] = {}

    # round-robin
    concepts = sorted(by_concept.keys(), key=lambda k: by_concept[k][0]["primary_confidence"], reverse=True)
    ptrs = {k: 0 for k in concepts}
    while len(selected) < target:
        progressed = False
        for c in concepts:
            if len(selected) >= target:
                break
            cap = taken_per_concept.get(c, 0)
            if cap >= per_concept_cap:
                continue
            idx = ptrs[c]
            while idx < len(by_concept[c]) and by_concept[c][idx] in selected:
                idx += 1
            if idx < len(by_concept[c]):
                selected.append(by_concept[c][idx])
                taken_per_concept[c] = cap + 1
                ptrs[c] = idx + 1
                progressed = True
        if not progressed:
            break

    # fill remainder (if any) with best-overall not yet selected
    if len(selected) < target:
        pool = sorted(
            [c for c in eligible if c not in selected],
            key=lambda x: (float(x.get("primary_confidence", 0.0)),
                           float(x.get("score_sum_norm") or -1.0),
                           float(x.get("score_area", 0.0))),
            reverse=True,
        )
        for c in pool:
            if len(selected) >= target:
                break
            selected.append(c)

    return selected[:target]

def best_reference_per_concept(crops: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Return {concept: crop_info} choosing the single highest-confidence crop
    (ties broken by score_sum_norm, score_area, then rank).
    """
    best: Dict[str, Dict] = {}
    for c in crops:
        cmap = c.get("concept_confidences", {}) or {}
        for concept, conf in cmap.items():
            cur = best.get(concept)
            if cur is None:
                best[concept] = c
            else:
                better = choose_better(
                    {**cur, "primary_confidence": cur.get("concept_confidences", {}).get(concept, 0.0)},
                    {**c,   "primary_confidence": conf}
                )
                best[concept] = better
    # shrink to minimal fields for ref sheet
    out: Dict[str, Dict[str, Any]] = {}
    for concept, c in best.items():
        out[concept] = {
            "crop_id": c.get("crop_id"),
            "bbox_idx": c.get("bbox_idx"),
            "bbox_sec_hz": c.get("bbox_sec_hz"),
            "rank": c.get("rank"),
            "primary_concept": c.get("primary_concept"),
            "primary_confidence": float(c.get("concept_confidences", {}).get(concept, c.get("primary_confidence", 0.0))),
            "present_tests": c.get("present_tests"),
            "concept_confidences": c.get("concept_confidences"),
            "score_sum_norm": c.get("score_sum_norm"),
            "score_area": c.get("score_area"),
            "area_bins": c.get("area_bins"),
        }
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Dynamic-crop K sweep + PASS concept detection + per-K summary.")
    # input crawling (can be norm_s__*.npy or mask95_smoothed__*.npy)
    ap.add_argument("--in-root", type=str, default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Root containing norm_s/mask npy files (recursive).")
    ap.add_argument("--glob", type=str, default="norm_s__*.npy,mask95_smoothed__*.npy",
                    help="Comma/space-separated glob(s) searched recursively under --in-root.")
    ap.add_argument("--k-range", type=str, default="1-10",
                    help="K sweep, e.g. '1-10' or '1,2,4,8'.")
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--open", dest="open_sz", type=int, default=3)
    ap.add_argument("--close", dest="close_sz", type=int, default=3)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--pad-t-ms", type=float, default=32.0,
                    help="Time padding added around each CC bbox (milliseconds).")
    ap.add_argument("--pad-f-hz", type=float, default=300.0,
                    help="Frequency padding added around each CC bbox (Hz).")
    ap.add_argument("--out-crops-root", type=str, default="./crops_sweep",
                    help="Where intermediate crops_manifest.json files go (per-K subfolders).")

    # arrays (Mb/Ms/add/miss/norm) lookup roots for detection
    ap.add_argument("--array-roots", type=str,
                    default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Colon/comma-separated roots containing Mb/Ms/masks per pair: <root>/<BONA>/<VOCODER>/...")

    # selection / diversity controls
    ap.add_argument("--target-per-pair", type=int, default=8,
                    help="Desired number of suggested crops per (bona,vocoder).")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="Minimum primary-concept confidence to keep a crop.")
    ap.add_argument("--per-concept-cap", type=int, default=3,
                    help="Max number of crops per concept in the suggested set.")

    # final suggestions
    ap.add_argument("--suggested-root", type=str, default="./suggested_crops",
                    help="Output root for suggested_crops/<BONA>/<VOCODER>/suggested_crops.json (and k_summary.json)")
    args = ap.parse_args()

    ks = parse_k_range(args.k_range)
    array_roots = [Path(s) for s in re.split(r"[:,]", args.array_roots) if s.strip()]

    in_root = Path(args.in_root)
    out_crops_root = Path(args.out_crops_root)
    suggested_root = Path(args.suggested_root)

    # collect candidate npy files
    patterns = parse_globs(args.glob)
    files = rglob_many(in_root, patterns)
    if not files:
        print(f"[warn] no files found under {in_root} with {patterns}")
        return

    for npy_path in files:
        # derive meta (bona/vocoder) from filename
        meta = emc.parse_from_filename(npy_path.name)
        bona, voc = meta["bona_id"], meta["vocoder"]

        all_detected: List[Dict] = []
        stft_meta_ref: Optional[Dict] = None
        per_k_summary: Dict[str, Dict] = {}

        for K in ks:
            out_root_k = out_crops_root / f"k{K}"
            out_root_k.mkdir(parents=True, exist_ok=True)

            # 1) dynamic crops for this K
            manifest_path = emc.process_file(
                npy_path=npy_path,
                out_root=out_root_k,
                k_per_pair=K,
                min_area=args.min_area,
                open_sz=args.open_sz,
                close_sz=args.close_sz,
                sr=args.sr,
                hop=args.hop,
                pad_t_ms=args.pad_t_ms,
                pad_f_hz=args.pad_f_hz,
            )
            if manifest_path is None:
                per_k_summary[str(K)] = {"num_detected": 0, "by_concept": {}, "confidence_mean": {}, "confidence_max": {}}
                continue

            # 2) detect concepts
            results = run_detection(manifest_path, array_roots=array_roots)

            # 2b) persist full per-crop diagnostics for downstream tools (confidence, catalog)
            write_concept_results(manifest_path, results)

            # 3) keep only crops with concepts (+ confidences)
            detected = filter_detected_only(manifest_path, results)
            all_detected.extend(detected)

            # 3b) per-K summary (counts + confidence stats)
            per_k_summary[str(K)] = summarize_detected_per_k(detected)

            if stft_meta_ref is None:
                with open(manifest_path, "r") as f:
                    stft_meta_ref = (json.load(f).get("stft") or {"sr": args.sr, "hop": args.hop})

        if not stft_meta_ref:
            continue

        # 4) dedupe across K (confidence-aware)
        deduped = dedupe_across_k(all_detected)

        # 5) select diverse, high-confidence crops per pair
        selected = select_diverse_high_conf(
            deduped,
            target=args.target_per_pair,
            per_concept_cap=args.per_concept_cap,
            min_conf=args.min_conf,
        )

        # 6) build per-concept reference (best crop per concept from all deduped, not only selected)
        per_concept_ref = best_reference_per_concept(deduped)

        # 7) write suggestions
        write_json(
            suggested_root / bona / voc / "suggested_crops.json",
            {
                "bona_id": bona,
                "vocoder": voc,
                "stft": stft_meta_ref,
                "num_selected": len(selected),
                "selection_params": {
                    "target_per_pair": args.target_per_pair,
                    "min_conf": args.min_conf,
                    "per_concept_cap": args.per_concept_cap
                },
                "selected_crops": selected,        # items include primary_concept/confidence and concept_confidences
                "per_concept_reference": per_concept_ref  # best crop per concept (for ref sheet)
            }
        )

        # 8) write per-K summary JSON
        write_json(
            suggested_root / bona / voc / "k_summary.json",
            {
                "bona_id": bona,
                "vocoder": voc,
                "stft": stft_meta_ref,
                "params": {
                    "k_values": ks,
                    "min_area": args.min_area,
                    "open": args.open_sz,
                    "close": args.close_sz,
                    "pad_t_ms": args.pad_t_ms,
                    "pad_f_hz": args.pad_f_hz,
                    "target_per_pair": args.target_per_pair,
                    "min_conf": args.min_conf,
                    "per_concept_cap": args.per_concept_cap
                },
                "per_k": per_k_summary,
                "overall_unique_after_dedupe": len(deduped)
            }
        )

if __name__ == "__main__":
    main()