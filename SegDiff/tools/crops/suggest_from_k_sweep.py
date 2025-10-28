#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set

# Optional: use detect_concepts for fallback confidence if needed
try:
    import detect_concepts as dc
    HAVE_DC = True
except Exception:
    dc = None
    HAVE_DC = False

# ---------------- IO helpers ----------------
def load_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, payload: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(p)

def rglob(root: Path, pat: str) -> List[Path]:
    return sorted(root.rglob(pat))

# ---------------- concept helpers ----------------
def present_tests(result: Dict) -> List[str]:
    return [k for k, v in (result.get("tests") or {}).items() if v.get("decision") == "present"]

def concept_confidences(result: Dict) -> Dict[str, float]:
    """
    Return {concept: confidence in [0,1]} for present concepts.
    Prefer 'confidence' stored by detector; fallback to dc.compute_confidence if available.
    """
    out: Dict[str, float] = {}
    tests = result.get("tests") or {}
    for cname, tres in tests.items():
        if tres.get("decision") == "present":
            conf = tres.get("confidence")
            if conf is None and HAVE_DC and hasattr(dc, "compute_confidence"):
                try:
                    conf = float(dc.compute_confidence(cname, tres))
                except Exception:
                    conf = 0.0
            try:
                out[cname] = float(conf) if conf is not None else 0.0
            except Exception:
                out[cname] = 0.0
    return out

def primary_concept_info(result: Dict) -> Tuple[Optional[str], float]:
    cmap = concept_confidences(result)
    if not cmap:
        return None, 0.0
    best = max(cmap.items(), key=lambda kv: kv[1])
    return best[0], float(best[1])

def bbox_key(bbox_idx: List[int]) -> Tuple[int,int,int,int]:
    t0, t1, f0, f1 = [int(v) for v in bbox_idx]
    return (t0, t1, f0, f1)

def choose_better(a: Dict, b: Dict) -> Dict:
    """
    Tie-breaker for same bbox across Ks.
    Prefer: more present tests, higher primary_conf, boundary_near_mid, higher solidity,
            higher score_sum_norm, higher score_area, better (smaller) rank.
    """
    def s(x: Dict):
        n_present = len(x.get("present_tests") or [])
        best_conf = float(x.get("primary_confidence", 0.0))
        boundary  = 1.0 if bool(x.get("boundary_near_mid", False)) else 0.0
        solidity  = float(x.get("solidity") or 0.0)
        sum_norm  = x.get("score_sum_norm")
        sum_norm  = float(sum_norm) if sum_norm is not None else -1.0
        area      = float(x.get("score_area") or 0.0)
        inv_rank  = -float(x.get("rank", 1e9)) if x.get("rank") is not None else -1e9
        return (n_present, best_conf, boundary, solidity, sum_norm, area, inv_rank)
    return b if s(b) > s(a) else a

def filter_detected_only(manifest_path: Path, results: List[Dict]) -> List[Dict]:
    mj = load_json(manifest_path)
    crops = mj["crops"] if isinstance(mj, dict) and "crops" in mj else mj
    by_key = {tuple(c["bbox_idx"]): c for c in crops}
    detected: List[Dict] = []
    for r in results:
        label = r.get("final_label", "")
        if label in ("no_concept_detected", "noise_or_uncertain"):
            continue
        meta = by_key.get(tuple(r["bbox_idx"]), {})
        cmap = concept_confidences(r)
        p_con, p_conf = primary_concept_info(r)
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
            "concept_confidences": cmap,
            "primary_concept": p_con,
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

def select_diverse_high_conf(crops: List[Dict], target:int, per_concept_cap:int, min_conf:float) -> List[Dict]:
    elig = [c for c in crops if (c.get("primary_concept") and float(c.get("primary_confidence",0.0)) >= min_conf)]
    if not elig: return []
    byc: Dict[str,List[Dict]] = {}
    for c in elig:
        byc.setdefault(c["primary_concept"], []).append(c)
    for v in byc.values():
        v.sort(key=lambda x: (float(x.get("primary_confidence",0.0)),
                              float(x.get("score_sum_norm") or -1.0),
                              float(x.get("score_area",0.0))), reverse=True)
    selected: List[Dict] = []
    taken: Dict[str,int] = {}
    concepts = sorted(byc.keys(), key=lambda k: byc[k][0]["primary_confidence"], reverse=True)
    ptr = {k:0 for k in concepts}
    while len(selected) < target:
        progressed = False
        for c in concepts:
            if len(selected) >= target: break
            if taken.get(c,0) >= per_concept_cap: continue
            i = ptr[c]
            while i < len(byc[c]) and byc[c][i] in selected:
                i += 1
            if i < len(byc[c]):
                selected.append(byc[c][i])
                taken[c] = taken.get(c,0) + 1
                ptr[c] = i + 1
                progressed = True
        if not progressed: break
    if len(selected) < target:
        pool = sorted([c for c in elig if c not in selected],
                      key=lambda x: (float(x.get("primary_confidence",0.0)),
                                     float(x.get("score_sum_norm") or -1.0),
                                     float(x.get("score_area",0.0))), reverse=True)
        for c in pool:
            if len(selected) >= target: break
            selected.append(c)
    return selected[:target]

def best_reference_per_concept(crops: List[Dict]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict] = {}
    for c in crops:
        for name, conf in (c.get("concept_confidences") or {}).items():
            cur = best.get(name)
            if cur is None:
                best[name] = c
            else:
                best[name] = choose_better(
                    {**cur, "primary_confidence": (cur.get("concept_confidences") or {}).get(name, 0.0)},
                    {**c,   "primary_confidence": conf}
                )
    out = {}
    for name, c in best.items():
        out[name] = {
            "crop_id": c.get("crop_id"),
            "bbox_idx": c.get("bbox_idx"),
            "bbox_sec_hz": c.get("bbox_sec_hz"),
            "rank": c.get("rank"),
            "primary_concept": c.get("primary_concept"),
            "primary_confidence": float((c.get("concept_confidences") or {}).get(name, c.get("primary_confidence",0.0))),
            "present_tests": c.get("present_tests"),
            "concept_confidences": c.get("concept_confidences"),
            "score_sum_norm": c.get("score_sum_norm"),
            "score_area": c.get("score_area"),
            "area_bins": c.get("area_bins"),
        }
    return out

def compute_pair_concepts_best_and_summary(deduped: List[Dict], pair_json_rel: str):
    counts: Dict[str,int] = {}
    maxc: Dict[str,float] = {}
    best_pick: Dict[str, Dict] = {}
    for crop in deduped:
        for c, conf in (crop.get("concept_confidences") or {}).items():
            counts[c] = counts.get(c,0) + 1
            if (c not in maxc) or (conf > maxc[c]):
                maxc[c] = float(conf)
                best_pick[c] = {
                    "concept": c,
                    "best_confidence": float(conf),
                    "crop_id": crop.get("crop_id"),
                    "bbox_idx": crop.get("bbox_idx"),
                    "bbox_sec_hz": crop.get("bbox_sec_hz"),
                    "path_hint": pair_json_rel
                }
    concepts_best = [best_pick[c] for c in sorted(best_pick)]
    concepts_present_summary = {c: {"count": counts.get(c,0), "max_conf": maxc.get(c,0.0)}
                                for c in sorted(counts)}
    return concepts_best, concepts_present_summary

def greedy_cover(pairs: List[Dict], required: List[str], max_pairs:int, conf_thr:float, min_multi:int):
    req = list(dict.fromkeys(required))
    uncovered: Set[str] = set(req)
    eligible = []
    for p in pairs:
        hi = [(n,c) for (n,c) in p.get("hi_concepts",[]) if c >= conf_thr]
        u = sorted({n for (n,_) in hi})
        if len(u) >= min_multi:
            sconf = sum(c for (_,c) in hi)
            eligible.append({**p, "hi_set": set(u), "hi_sumconf": float(sconf), "hi_count": len(u)})
    picked = []
    while uncovered and eligible and len(picked) < max_pairs:
        def score(px):
            return (len(px["hi_set"] & uncovered), px["hi_sumconf"], px["hi_count"], tuple(px.get("pair_key","zzz")))
        best = max(eligible, key=score)
        eligible.remove(best)
        if not (best["hi_set"] & uncovered):
            continue
        picked.append({
            "bona_id": best["bona_id"],
            "vocoder": best["vocoder"],
            "concepts": [{"name": n, "confidence": float(c)} for (n,c) in sorted(best["hi_concepts"]) if c >= conf_thr],
            "num_concepts": int(len([1 for (_,c) in best["hi_concepts"] if c >= conf_thr])),
            "pair_key": best.get("pair_key")
        })
        uncovered -= best["hi_set"]
    return picked, sorted(uncovered)

# ---------------- crawl K sweep ----------------
def collect_pairs_from_crops_sweep(crops_sweep_root: Path) -> Dict[Tuple[str,str], Dict]:
    """
    Walk k*/<BONA>/<VOCODER>/crops/{crops_manifest.json,concept_results.json}
    Aggregate detected crops across Ks per pair.
    """
    by_pair: Dict[Tuple[str,str], Dict] = {}
    for manifest in rglob(crops_sweep_root, "k*/**/crops/crops_manifest.json"):
        voc_dir = manifest.parent.parent  # .../<VOCODER>/crops
        bona_dir = voc_dir.parent         # .../<BONA>/<VOCODER>
        bona = bona_dir.parent.name if bona_dir.name == "crops" else bona_dir.name  # robust if layout differs
        bona = bona_dir.name              # standard layout
        voc  = voc_dir.name
        key = (bona, voc)
        # matching concept_results.json
        cr = manifest.parent / "concept_results.json"
        if not cr.exists():
            continue
        try:
            results = load_json(cr)
        except Exception:
            continue

        # stash stft meta once
        stft = {}
        try:
            mj = load_json(manifest)
            stft = (mj.get("stft") or {})
        except Exception:
            pass

        detected = filter_detected_only(manifest, results)
        meta = by_pair.setdefault(key, {"stft": stft, "detected_all_k": []})
        if not meta.get("stft"):
            meta["stft"] = stft
        meta["detected_all_k"].extend(detected)
    return by_pair

# ---------------- main ----------------
def parse_list(s: str) -> List[str]:
    return [t.strip() for t in re.split(r"[,\s]+", s or "") if t.strip()]

def main():
    ap = argparse.ArgumentParser(description="Build per-pair suggestions + global recommendations from K-sweep outputs.")
    ap.add_argument("--crops-sweep-root", type=str, default="./crops_sweep",
                    help="Root with k*/<BONA>/<VOCODER>/crops/(crops_manifest.json, concept_results.json)")
    ap.add_argument("--suggested-root", type=str, default="./suggested_crops",
                    help="Output root for per-pair suggested_crops.json and _global/*")
    # per-pair selection
    ap.add_argument("--target-per-pair", type=int, default=8)
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--per-concept-cap", type=int, default=3)
    # global recommendation
    ap.add_argument("--required-concepts", type=str,
                    default="formant_attenuation,fogging_vowel,fogging_fricative,pseudo_formant,concatenatedness,coarticulatory_deficit,hyperneatness,hyperflat_prosody")
    ap.add_argument("--high-conf-thr", type=float, default=0.95)
    ap.add_argument("--min-multi-concepts", type=int, default=2)
    ap.add_argument("--max-recommended", type=int, default=20)
    args = ap.parse_args()

    crops_sweep_root = Path(args.crops_sweep_root)
    suggested_root   = Path(args.suggested_root)
    global_root      = suggested_root / "_global"
    required = parse_list(args.required_concepts)

    by_pair = collect_pairs_from_crops_sweep(crops_sweep_root)
    if not by_pair:
        print(f"[warn] no pairs found under {crops_sweep_root}")
        return

    # per-pair build + collect candidates for global cover
    candidate_pairs = []
    coverage_counts: Dict[str,int] = {}
    coverage_pairs: Dict[str,int] = {}
    exemplars_best: Dict[str, Dict] = {}
    exemplars_100: Dict[str, List[Dict]] = {}

    for (bona, voc), meta in by_pair.items():
        deduped = dedupe_across_k(meta.get("detected_all_k", []))
        selected = select_diverse_high_conf(deduped, args.target_per_pair, args.per_concept_cap, args.min_conf)
        per_concept_ref = best_reference_per_concept(deduped)

        pair_key = f"{bona}/{voc}"
        pair_json_rel = str((Path("suggested_crops") / bona / voc / "suggested_crops.json").as_posix())
        concepts_best, concepts_present_summary = compute_pair_concepts_best_and_summary(deduped, pair_json_rel)

        # coverage tallies
        for c, stat in concepts_present_summary.items():
            coverage_counts[c] = coverage_counts.get(c,0) + int(stat.get("count",0))
            if stat.get("count",0) > 0:
                coverage_pairs[c] = coverage_pairs.get(c,0) + 1

        # exemplars
        for cb in concepts_best:
            c = cb["concept"]; conf = float(cb.get("best_confidence", 0.0))
            if conf >= 1.0:
                exemplars_100.setdefault(c, []).append({
                    "bona_id": bona, "vocoder": voc, "confidence": conf,
                    "crop_id": cb.get("crop_id"), "bbox_idx": cb.get("bbox_idx"),
                    "pair_json": pair_json_rel
                })
            cur = exemplars_best.get(c)
            cand = {"bona_id": bona, "vocoder": voc, "confidence": conf,
                    "crop_id": cb.get("crop_id"), "bbox_idx": cb.get("bbox_idx"),
                    "pair_json": pair_json_rel}
            if (cur is None) or (conf > float(cur.get("confidence", 0.0))):
                exemplars_best[c] = cand

        # candidate for cover
        hi_concepts = [(cb["concept"], float(cb.get("best_confidence",0.0))) for cb in concepts_best]
        candidate_pairs.append({"bona_id": bona, "vocoder": voc, "pair_key": pair_key, "hi_concepts": hi_concepts})

        # write per-pair suggestion JSON
        save_json(
            suggested_root / bona / voc / "suggested_crops.json",
            {
                "bona_id": bona,
                "vocoder": voc,
                "pair_key": pair_key,
                "pair_json_rel": pair_json_rel,
                "stft": meta.get("stft", {}),
                "num_selected": len(selected),
                "selection_params": {
                    "target_per_pair": args.target_per_pair,
                    "min_conf": args.min_conf,
                    "per_concept_cap": args.per_concept_cap
                },
                "selected_crops": selected,
                "per_concept_reference": per_concept_ref,
                "concepts_best": concepts_best,
                "concepts_present_summary": concepts_present_summary
            }
        )

    # global: greedy cover
    picked, uncovered = greedy_cover(candidate_pairs, required,
                                     max_pairs=args.max_recommended,
                                     conf_thr=args.high_conf_thr,
                                     min_multi=args.min_multi_concepts)

    coverage = {
        "by_concept": {
            c: {"total_crops": int(coverage_counts.get(c,0)),
                "pairs_with_concept": int(coverage_pairs.get(c,0))}
            for c in sorted(set(list(coverage_counts.keys()) + list(coverage_pairs.keys())))
        },
        "total_pairs": int(len(by_pair))
    }

    global_root.mkdir(parents=True, exist_ok=True)
    save_json(global_root / "global_summary.json", {
        "run_params": {
            "min_conf_pair_select": args.min_conf,
            "pair_cap_per_concept": args.per_concept_cap,
            "high_conf_thr": args.high_conf_thr,
            "min_multi_concepts": args.min_multi_concepts,
            "max_recommended": args.max_recommended
        },
        "concepts_list": required,
        "coverage": coverage,
        "exemplars_100": {c: sorted(v, key=lambda x: (-float(x.get("confidence",0.0)),
                                                      x.get("bona_id",""), x.get("vocoder","")))
                          for c, v in exemplars_100.items()},
        "exemplars_best": exemplars_best
    })
    # keep the same schema your viz expects: key is "picked"
    save_json(global_root / "recommended_20_pairs.json",
              {"picked": picked, "required_concepts": required, "uncovered": uncovered})

if __name__ == "__main__":
    main()