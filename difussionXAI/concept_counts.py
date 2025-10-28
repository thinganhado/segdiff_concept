import argparse, json, glob, sys
from pathlib import Path
from collections import Counter, defaultdict

SKIP_LABELS = {"noise_or_uncertain", "no_concept_detected"}

def parse_concepts_from_crop(crop: dict) -> set:
    """
    Returns a set of concept names detected for a single crop.
    Prefers final_label, falls back to tests where decision is present.
    """
    concepts = set()
    final_label = crop.get("final_label", "") or ""
    if final_label and final_label not in SKIP_LABELS:
        for part in final_label.split(","):
            c = part.strip()
            if c and c not in SKIP_LABELS:
                concepts.add(c)
    # fallback, or add any present tests not already captured
    tests = crop.get("tests", {}) or {}
    for name, res in tests.items():
        if isinstance(res, dict) and res.get("decision") == "present":
            if name not in SKIP_LABELS:
                concepts.add(name)
    return concepts

def load_concept_results(path: Path) -> list:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def main():
    ap = argparse.ArgumentParser(description="Count concepts across concept_results.json files")
    ap.add_argument("--glob", type=str, required=True,
                    help="Glob pattern to find concept_results.json files, for example /home/opc/**/concept_results.json")
    ap.add_argument("--out", type=str, default="concept_counts.csv",
                    help="Path to write the CSV summary, default concept_counts.csv in CWD")
    ap.add_argument("--by-pair-out", type=str, default=None,
                    help="Optional CSV path to write per pair counts")
    ap.add_argument("--print", action="store_true",
                    help="Also print the totals to stdout")
    args = ap.parse_args()

    files = [Path(p) for p in glob.glob(args.glob, recursive=True) if p.endswith("concept_results.json")]
    if not files:
        print("No files matched. Nothing to do.")
        sys.exit(0)

    total_files = 0
    total_crops = 0
    concept_counts = Counter()
    pair_counts = defaultdict(Counter)  # pair_id -> Counter

    for fp in files:
        if not fp.exists():
            continue  # skip files that do not exist yet
        rows = load_concept_results(fp)
        if not rows:
            continue  # empty or unreadable
        total_files += 1
        for crop in rows:
            total_crops += 1
            pair_id = crop.get("pair_id", "UNKNOWN_PAIR")
            concepts = parse_concepts_from_crop(crop)
            for c in concepts:
                concept_counts[c] += 1
                pair_counts[pair_id][c] += 1

    # write totals CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("concept,count\n")
        for concept, count in concept_counts.most_common():
            f.write(f"{concept},{count}\n")

    # optional, per pair
    if args.by_pair_out:
        by_pair_path = Path(args.by_pair_out)
        by_pair_path.parent.mkdir(parents=True, exist_ok=True)
        # collect all concept keys
        all_concepts = sorted(concept_counts.keys())
        with open(by_pair_path, "w") as f:
            f.write("pair_id," + ",".join(all_concepts) + "\n")
            for pair_id, ctr in sorted(pair_counts.items()):
                row = [pair_id] + [str(ctr.get(c, 0)) for c in all_concepts]
                f.write(",".join(row) + "\n")

    if args.print:
        print(f"Processed files, {total_files}")
        print(f"Processed crops, {total_crops}")
        print("Concept totals,")
        for concept, count in concept_counts.most_common():
            print(f"  {concept}: {count}")
        if args.by_pair_out:
            print(f"Wrote per pair counts to, {args.by_pair_out}")

if __name__ == "__main__":
    main()