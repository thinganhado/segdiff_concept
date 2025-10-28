#!/usr/bin/env python3
# K-sweep: cropping + concept detection ONLY.
# Writes (per K):
#   ./crops_sweep/k{K}/<BONA>/<VOCODER>/crops/crops_manifest.json
#   ./crops_sweep/k{K}/<BONA>/<VOCODER>/crops/concept_results.json
# And a per-pair summary (no selections/recommendations):
#   ./suggested_crops/<BONA>/<VOCODER>/k_summary.json

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
                try:
                    conf = float(dc.compute_confidence(cname, tres))
                except Exception:
                    conf = 0.0
            out[cname] = float(conf)
    return out

def primary_concept_info(result: Dict) -> Tuple[Optional[str], float]:
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
    Tie-breaker for same bbox across Ks (no selection; just for 'overall_unique_after_dedupe' stat).
    Prefer:
    1) more present tests,
    2) higher best per-concept confidence,
    3) boundary near mid (gentle),
    4) higher solidity (gentle),
    5) higher score_sum_norm (if both have it),
    6) higher score_area,
    7) higher rank (smaller number).
    """
    def score_pkg(x: Dict) -> Tuple[int, float, float, float, float, float, float]:
        n_present = len(x.get("present_tests", []) or [])
        best_conf = float(x.get("primary_confidence", 0.0))
        boundary = 1.0 if bool(x.get("boundary_near_mid", False)) else 0.0
        solidity = float(x.get("solidity", 0.0))
        sum_norm = x.get("score_sum_norm")
        sum_norm = float(sum_norm) if sum_norm is not None else -1.0
        area = float(x.get("score_area", 0.0))
        inv_rank = -float(x.get("rank", 1e9)) if x.get("rank") is not None else -1e9
        return (n_present, best_conf, boundary, solidity, sum_norm, area, inv_rank)
    return b if score_pkg(b) > score_pkg(a) else a

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

        conf_map = concept_confidences(r)
        p_concept, p_conf = primary_concept_info(r)

        detected.append({
            "crop_id": r.get("crop_id"),
            "bbox_idx": r.get("bbox_idx"),
            "bbox_sec_hz": r.get("bbox_sec_hz"),
            "rank": meta.get("rank"),
            "score_sum_norm": meta.get("score_sum_norm"),
            "score_sum_canon": meta.get("score_sum_canon"),
            "score_area": meta.get("score_area", 0.0),
            "area_bins": meta.get("area_bins", 0),
            "solidity": meta.get("solidity"),
            "orientation": meta.get("orientation"),
            "freq_band": meta.get("freq_band"),
            "boundary_near_mid": meta.get("boundary_near_mid"),
            "voiced_fraction": meta.get("voiced_fraction"),
            "vowel_fraction": meta.get("vowel_fraction"),
            "present_tests": present_tests(r),
            "concept_confidences": conf_map,
            "primary_concept": p_concept,
            "primary_confidence": p_conf,
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
    # Prefer the with_roots variant
    if hasattr(dc, "analyze_manifest_with_roots"):
        return dc.analyze_manifest_with_roots(manifest_path, array_roots=array_roots)
    # Fallback
    if hasattr(dc, "analyze_manifest"):
        return dc.analyze_manifest(manifest_path)
    raise RuntimeError("detect_concepts has neither analyze_manifest_with_roots nor analyze_manifest")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Dynamic-crop K sweep + PASS concept detection (detect-only).")
    # input crawling (can be norm_s__*.npy or mask95_smoothed__*.npy)
    ap.add_argument("--in-root", type=str, default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Root containing norm_s/mask npy files (recursive).")
    ap.add_argument("--glob", type=str, default="mask95_smoothed__*.npy",
                    help="Comma/space-separated glob(s) under --in-root (mask95_smoothed__*.npy).")
    ap.add_argument("--k-range", type=str, default="1-10",
                    help="K sweep, e.g. '1-10' or '1,2,4,8'.")

    # cropper geometry & legacy morph
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--min-solidity", type=float, default=0.60,
                    help="Minimum (area / bbox_area) after hysteresis CCs.")
    ap.add_argument("--open", dest="open_sz", type=int, default=0,
                    help="Legacy opening kernel size (rows=cols). Usually 0 with hysteresis.")
    ap.add_argument("--close", dest="close_sz", type=int, default=0,
                    help="Legacy closing kernel size (rows=cols). Usually 0 with hysteresis.")

    # STFT geometry & padding
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--pad-t-ms", type=float, default=32.0,
                    help="Time padding added around each CC bbox (milliseconds).")
    ap.add_argument("--pad-f-hz", type=float, default=300.0,
                    help="Frequency padding added around each CC bbox (Hz).")

    # canonical/hysteresis controls
    ap.add_argument("--norm-quantile", type=float, default=0.95,
                    help="Legacy single-threshold for float-like inputs (still recorded).")
    ap.add_argument("--hyst-hi", type=float, default=0.97, dest="q_hi",
                    help="Initial high quantile for hysteresis seeds.")
    ap.add_argument("--hyst-lo", type=float, default=0.90, dest="q_lo",
                    help="Initial low quantile for hysteresis growth.")
    ap.add_argument("--close-time", type=int, default=3,
                    help="Anisotropic closing kernel width in time (columns).")
    ap.add_argument("--close-freq", type=int, default=1,
                    help="Anisotropic closing kernel height in frequency (rows).")
    ap.add_argument("--auto-tune", dest="auto_tune", action="store_true", default=True,
                    help="Auto-tune hysteresis thresholds toward K.")
    ap.add_argument("--no-auto-tune", dest="auto_tune", action="store_false",
                    help="Disable auto-tuning; use provided --hyst-hi/--hyst-lo as-is.")
    ap.add_argument("--nms-iou", type=float, default=0.70,
                    help="IoU threshold for NMS over padded boxes (disable with <=0 or >=1).")

    # arrays (Mb/Ms/add/miss/norm) lookup roots for detection
    ap.add_argument("--array-roots", type=str,
                    default="/home/opc/difussionXAI/mask_outputs_crop_phonetier",
                    help="Colon/comma-separated roots containing Mb/Ms/masks per pair: <root>/<BONA>/<VOCODER>/...")

    # TextGrids: used by cropper for tags AND by detector for phone spans
    ap.add_argument("--textgrid-root", type=str, default="/home/opc/aligned_textgrids",
                    help="Root dir for TextGrids: <root>/<BONA>/<BONA>.TextGrid")

    # outputs
    ap.add_argument("--out-crops-root", type=str, default="./crops_sweep",
                    help="Where intermediate crops_manifest.json files go (per-K subfolders).")
    ap.add_argument("--summary-root", type=str, default="./suggested_crops",
                    help="Where per-pair k_summary.json is written (kept for compatibility).")
    args = ap.parse_args()

    ks = parse_k_range(args.k_range)
    array_roots = [Path(s) for s in re.split(r"[:,]", args.array_roots) if s.strip()]

    in_root = Path(args.in_root)
    out_crops_root = Path(args.out_crops_root)
    summary_root = Path(args.summary_root)
    tg_root = Path(args.textgrid_root).resolve() if args.textgrid_root else None

    # Ensure detector sees the same TextGrid root
    if tg_root:
        try:
            dc.TEXTGRID_ROOT = tg_root
        except Exception:
            print("[warn] detect_concepts.TEXTGRID_ROOT could not be set")
        if not tg_root.exists():
            print(f"[warn] TextGrid root does not exist: {tg_root}")

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
                min_solidity=args.min_solidity,
                open_sz=args.open_sz,
                close_sz=args.close_sz,
                sr=args.sr,
                hop=args.hop,
                pad_t_ms=args.pad_t_ms,
                pad_f_hz=args.pad_f_hz,
                nms_iou=args.nms_iou,
                q_hi_init=args.q_hi,
                q_lo_init=args.q_lo,
                close_time=args.close_time,
                close_freq=args.close_freq,
                auto_tune=args.auto_tune,
                textgrid_root=tg_root,
            )
            if manifest_path is None:
                per_k_summary[str(K)] = {"num_detected": 0, "by_concept": {}, "confidence_mean": {}, "confidence_max": {}}
                continue

            # 2) detect concepts
            results = run_detection(manifest_path, array_roots=array_roots)

            # 2b) persist full per-crop diagnostics for downstream tools
            write_concept_results(manifest_path, results)

            # 3) stats for summary
            detected = filter_detected_only(manifest_path, results)
            all_detected.extend(detected)
            per_k_summary[str(K)] = summarize_detected_per_k(detected)

            if stft_meta_ref is None:
                with open(manifest_path, "r") as f:
                    stft_meta_ref = (json.load(f).get("stft") or {"sr": args.sr, "hop": args.hop})

        if not stft_meta_ref:
            continue

        # 4) only for a summary stat (no output selection): how many unique after dedupe?
        overall_unique_after_dedupe = len(dedupe_across_k(all_detected))

        # 5) write per-pair summary JSON (kept tiny; no selections/recommendations)
        write_json(
            summary_root / bona / voc / "k_summary.json",
            {
                "bona_id": bona,
                "vocoder": voc,
                "stft": stft_meta_ref,
                "params": {
                    "k_values": ks,
                    "min_area": args.min_area,
                    "min_solidity": args.min_solidity,
                    "open": args.open_sz,
                    "close": args.close_sz,
                    "pad_t_ms": args.pad_t_ms,
                    "pad_f_hz": args.pad_f_hz,
                    "norm_quantile": args.norm_quantile,
                    "hyst_hi": args.q_hi,
                    "hyst_lo": args.q_lo,
                    "close_time": args.close_time,
                    "close_freq": args.close_freq,
                    "auto_tune": args.auto_tune,
                    "nms_iou": args.nms_iou
                },
                "per_k": per_k_summary,
                "overall_unique_after_dedupe": overall_unique_after_dedupe
            }
        )

if __name__ == "__main__":
    main()