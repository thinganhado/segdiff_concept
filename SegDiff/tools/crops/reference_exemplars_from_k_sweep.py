#!/usr/bin/env python3
# reference_exemplars_from_k_sweep.py
# Build per-concept exemplar lists by reading **any**
# crops_sweep/**/crops/concept_results.json, robust to extra path levels.

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Optional

# keep concept list identical to detector
CONCEPTS = [
    "fogging_vowel",
    "formant_attenuation",
    "fogging_fricative",
    "pseudo_formant",
    "concatenatedness",
    "coarticulatory_deficit",
    "hyperneatness",
    "hyperflat_prosody",
]

# Optional: if some results are missing 'confidence', compute it via detect_concepts
try:
    import detect_concepts as dc
    _HAVE_DC = True
except Exception:
    _HAVE_DC = False

def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, payload: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(p)

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _scan_parents_for_k(p: Path) -> int:
    for part in p.parts:
        m = re.fullmatch(r"k(\d+)", part)
        if m:
            return int(m.group(1))
    return -1

def _bona_voc_from_path(cres_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Expect one of:
      .../k{K}/<BONA>/<VOCODER>/concept_results.json
      .../k{K}/<BONA>/<VOCODER>/crops/concept_results.json
    Returns (BONA, VOCODER) or (None, None) if not matched.
    """
    parts = list(cres_path.parts)
    try:
        i = next(i for i, s in enumerate(parts) if re.fullmatch(r"k\d+", s))
    except StopIteration:
        return None, None
    # after k{K} we expect at least BONA/VOC[/crops]/concept_results.json
    tail = parts[i+1:]
    if len(tail) >= 3 and tail[-1] == "concept_results.json":
        if tail[-2] == "crops":
            # .../kK/BONA/VOC/crops/concept_results.json
            if len(tail) >= 4:
                return tail[-4], tail[-3]
        else:
            # .../kK/BONA/VOC/concept_results.json
            return tail[-3], tail[-2]
    return None, None

def iter_result_files(crops_root: Path) -> Iterable[Tuple[str, str, int, Path]]:
    """
    Yields (bona_id, vocoder, K, concept_results.json path) for every match under crops_root,
    robust to presence/absence of a 'crops' subfolder.
    """
    for cres in crops_root.rglob("concept_results.json"):
        K = _scan_parents_for_k(cres)
        if K < 0:
            continue
        bona, voc = _bona_voc_from_path(cres)
        if not bona or not voc:
            continue
        yield (bona, voc, K, cres)

def concept_confidence_from_test(concept: str, tres: Dict) -> float:
    conf = tres.get("confidence")
    if conf is not None:
        try:
            return float(conf)
        except Exception:
            return 0.0
    if _HAVE_DC:
        try:
            return float(dc.compute_confidence(concept, tres))
        except Exception:
            pass
    return 0.0

def collect_all(crops_root: Path, debug: bool = False) -> Dict[str, List[Dict]]:
    """
    Returns {concept: [items...]}, where each item has:
      bona_id, vocoder, concept, confidence, confidence_pct, crop_id, bbox_idx, bbox_sec_hz, K, source
    """
    by_concept: Dict[str, List[Dict]] = {c: [] for c in CONCEPTS}

    for bona, voc, K, res_path in iter_result_files(crops_root):
        results = load_json(res_path)
        if not isinstance(results, list):
            continue

        dbg_counts = {c: 0 for c in CONCEPTS}

        for r in results:
            tests = r.get("tests") or {}

            # primary path: test block says "present"
            for concept in CONCEPTS:
                tres = tests.get(concept)
                if not isinstance(tres, dict):
                    continue
                if _norm(tres.get("decision")) != "present":
                    continue
                conf = concept_confidence_from_test(concept, tres)
                item = {
                    "bona_id": bona,
                    "vocoder": voc,
                    "concept": concept,
                    "confidence": float(conf),
                    "confidence_pct": float(conf) * 100.0,
                    "crop_id": r.get("crop_id"),
                    "bbox_idx": r.get("bbox_idx"),
                    "bbox_sec_hz": r.get("bbox_sec_hz"),
                    "K": int(K),
                    "source": "concept_results",
                }
                by_concept[concept].append(item)
                dbg_counts[concept] += 1

            # fallback: honor final_label if provided
            fl = _norm(r.get("final_label"))
            if fl in CONCEPTS:
                tres = tests.get(fl, {})
                conf = concept_confidence_from_test(fl, tres) if isinstance(tres, dict) else 0.0
                item = {
                    "bona_id": bona,
                    "vocoder": voc,
                    "concept": fl,
                    "confidence": float(conf),
                    "confidence_pct": float(conf) * 100.0,
                    "crop_id": r.get("crop_id"),
                    "bbox_idx": r.get("bbox_idx"),
                    "bbox_sec_hz": r.get("bbox_sec_hz"),
                    "K": int(K),
                    "source": "concept_results(final_label)",
                }
                by_concept[fl].append(item)
                dbg_counts[fl] += 1

        if debug:
            total = sum(dbg_counts.values())
            print(f"[debug] {res_path}: collected {total} items -> " +
                  ", ".join(f"{k}:{v}" for k, v in dbg_counts.items() if v))

    return by_concept

def dedupe(items: List[Dict]) -> List[Dict]:
    """
    Dedupe by (bona,voc,concept,tuple(bbox_idx)) keeping the highest confidence.
    If bbox_idx missing, key by crop_id.
    """
    best: Dict[Tuple[str,str,str,Tuple[Any,...]], Dict] = {}
    for it in items:
        bb = tuple(it.get("bbox_idx") or [])
        key = (it["bona_id"], it["vocoder"], it["concept"], bb if len(bb)==4 else ("CROP", it.get("crop_id")))
        prev = best.get(key)
        if prev is None or float(it.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            best[key] = it
    return list(best.values())

def main():
    ap = argparse.ArgumentParser(description="Build per-concept exemplar lists from crops_sweep concept_results.json files.")
    ap.add_argument("--crops-root", type=str, default="./crops_sweep",
                    help="Root containing k{K}/<BONA>/<VOCODER>/.../crops/concept_results.json (any depth).")
    ap.add_argument("--out-root", type=str, default="./suggested_crops/_global/concept_from_k_sweep",
                    help="Where to write <concept>_95plus.json and <concept>_topN.json")
    ap.add_argument("--min-conf", type=float, default=0.95, help="Threshold for *_95plus lists (0..1)")
    ap.add_argument("--topn", type=int, default=10, help="Top-N for *_topN.json across the whole dataset")
    ap.add_argument("--one-per-pair", action="store_true", default=False,
                    help="If set, keep at most one crop per (BONA,VOCODER) per concept (max confidence).")
    ap.add_argument("--debug", action="store_true", default=False,
                    help="Print per-file collection counts.")
    args = ap.parse_args()

    crops_root = Path(args.crops_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    by_concept_all = collect_all(crops_root, debug=args.debug)

    summary = {
        "min_conf": args.min_conf,
        "topn": args.topn,
        "one_per_pair": bool(args.one_per_pair),
        "note": "collected from per-K concept_results.json under crops_sweep; robust glob + dedupe across Ks.",
        "concepts": {}
    }

    for concept in CONCEPTS:
        items = dedupe(by_concept_all.get(concept, []))

        if args.one_per_pair:
            best_per_pair: Dict[Tuple[str,str], Dict] = {}
            for it in items:
                key = (it["bona_id"], it["vocoder"])
                if key not in best_per_pair or it["confidence"] > best_per_pair[key]["confidence"]:
                    best_per_pair[key] = it
            items = list(best_per_pair.values())

        items_sorted = sorted(items, key=lambda x: x["confidence"], reverse=True)
        items_95 = [x for x in items_sorted if x["confidence"] >= args.min_conf]
        items_top = items_sorted[:args.topn]

        save_json(out_root / f"{concept}_95plus.json",
                  {"concept": concept, "min_conf": args.min_conf, "items": items_95})
        save_json(out_root / f"{concept}_top{args.topn}.json",
                  {"concept": concept, "topn": args.topn, "items": items_top})

        summary["concepts"][concept] = {
            "n_total_collected": len(items),
            "n_95plus": len(items_95),
            "top_best_conf": (items_top[0]["confidence"] if items_top else 0.0),
            "top_min_conf": (items_top[-1]["confidence"] if items_top else 0.0),
            "files": {
                "all_95plus": f"{concept}_95plus.json",
                "topn": f"{concept}_top{args.topn}.json",
            }
        }

    save_json(out_root / "_summary.json", summary)

if __name__ == "__main__":
    main()