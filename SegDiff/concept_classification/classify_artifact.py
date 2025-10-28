# classify_artifact.py
# Usage:
#   python classify_artifact.py /home/opc/SegDiff/ref/ref_sheet.png /home/opc/SegDiff/concepts_query
#   # or classify one file:
#   python classify_artifact.py /home/opc/SegDiff/ref/ref_sheet.png /home/opc/SegDiff/concepts_query/fogging/foo.png

import sys, os, json, base64, mimetypes, argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

LABELS = ["fogging", "formant attenuation", "imaging"]  # exactly your labels
EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULT_MODEL = "gpt-4.1-mini"

def parse_args():
    p = argparse.ArgumentParser(
        description="Classify spectrogram artifacts using a ref sheet + query image(s)."
    )
    p.add_argument("ref", help="Path to reference sheet image (e.g., ref_sheet.png)")
    p.add_argument("query", help="Path to a single query image OR a labeled root folder")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vision-capable model to use (default: {DEFAULT_MODEL})",
    )
    return p.parse_args()

# ----------------- helpers -----------------
def data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def system_prompt() -> str:
    return (
        "You are an expert in spectrogram artifact classification. "
        "You will receive TWO images:\n"
        "1) a REFERENCE SHEET that shows three labeled examples (left→right): "
        "'fogging', 'formant attenuation', 'imaging';\n"
        "2) a QUERY spectrogram to classify.\n\n"
        "Task: Choose exactly ONE label for the QUERY from "
        "['fogging','formant attenuation','imaging'] based on similarity to the reference examples.\n"
        "Output ONLY JSON with fields {label, confidence} where confidence is in [0,1]."
    )

def classify_one(ref_data_url: str, query_img: Path, model: str = "gpt-4.1-mini") -> dict:
    payload = dict(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": system_prompt()},
            {"role": "user","content":[
                {"type":"input_text","text":"REFERENCE SHEET (left→right: fogging, formant attenuation, imaging)."},
                {"type":"input_image","image_url": ref_data_url},
            ]},
            {"role":"user","content":[
                {"type":"input_text","text":"Classify this QUERY spectrogram into exactly one label."},
                {"type":"input_image","image_url": data_url(query_img)},
            ]},
        ],
    )
    try:
        # Preferred: Structured Outputs
        payload["response_format"] = {
            "type":"json_schema",
            "json_schema":{
                "name":"artifact_classification",
                "schema":{
                    "type":"object",
                    "properties":{
                        "label":{"type":"string","enum": LABELS},
                        "confidence":{"type":"number","minimum":0,"maximum":1},
                    },
                    "required":["label","confidence"],
                    "additionalProperties": False,
                },
            },
        }
        resp = client.responses.create(**payload)
        return json.loads(resp.output_text)
    except TypeError:
        # Older SDK path: no response_format; best-effort JSON parse
        resp = client.responses.create(**{k:v for k,v in payload.items() if k!="response_format"})
        txt = resp.output_text.strip()
        try:
            return json.loads(txt)
        except Exception:
            # last resort: wrap into our schema if the model emitted plain text
            return {"label": txt.splitlines()[0][:32], "confidence": 0.0}

def iter_query_images(root: Path) -> List[Tuple[str, Path]]:
    # Accept either: a single file, or a folder with subfolders named after labels
    if root.is_file():
        return [("unknown", root)]
    imgs: List[Tuple[str, Path]] = []
    for label in LABELS:
        sub = root / label  # folder names must match LABELS exactly (including space)
        if not sub.exists():
            continue
        for p in sorted(sub.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                imgs.append((label, p))   # gold label inferred from parent folder
    return imgs

# ----------------- metrics -----------------
def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b

def compute_metrics(rows: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Dict]:
    idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)
    cm = [[0] * K for _ in range(K)]  # cm[gold][pred]

    for r in rows:
        g, p = r.get("gold"), r.get("label")
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1

    per: Dict[str, Dict[str, float]] = {}
    TP_sum = FP_sum = FN_sum = 0
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(K))
    for i, lab in enumerate(labels):
        TP = cm[i][i]
        FP = sum(cm[r][i] for r in range(K) if r != i)
        FN = sum(cm[i][c] for c in range(K) if c != i)
        TP_sum += TP; FP_sum += FP; FN_sum += FN
        prec = _safe_div(TP, TP + FP)
        rec  = _safe_div(TP, TP + FN)
        f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(cm[i])}

    acc = _safe_div(correct, total)
    micro_p = _safe_div(TP_sum, TP_sum + FP_sum)
    micro_r = _safe_div(TP_sum, TP_sum + FN_sum)
    micro_f1 = 0.0 if (micro_p + micro_r) == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)

    macro_p = sum(per[lab]["precision"] for lab in labels) / K
    macro_r = sum(per[lab]["recall"] for lab in labels) / K
    macro_f1 = sum(per[lab]["f1"] for lab in labels) / K

    return {
        "overall": {
            "accuracy": acc,
            "precision_micro": micro_p,
            "recall_micro": micro_r,
            "f1_micro": micro_f1,
            "precision_macro": macro_p,
            "recall_macro": macro_r,
            "f1_macro": macro_f1,
            "support": total,
        },
        "per_class": per,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }

def print_metrics(m: Dict[str, Dict], labels: List[str]) -> None:
    o = m["overall"]
    print("\n=== OVERALL ===")
    print(f"Accuracy        : {o['accuracy']:.3f}")
    print(f"Precision (macro / micro): {o['precision_macro']:.3f} / {o['precision_micro']:.3f}")
    print(f"Recall    (macro / micro): {o['recall_macro']:.3f} / {o['recall_micro']:.3f}")
    print(f"F1-Score  (macro / micro): {o['f1_macro']:.3f} / {o['f1_micro']:.3f}")

    print("\n=== PER-CLASS ===")
    print("Class                    Precision  Recall  F1    Support")
    for lab in labels:
        r = m["per_class"][lab]
        print(f"{lab:<24}  {r['precision']:.3f}     {r['recall']:.3f}  {r['f1']:.3f}  {r['support']:>6}")

    print("\n=== CONFUSION MATRIX (gold rows × pred cols) ===")
    labs = m["confusion_matrix"]["labels"]
    mat  = m["confusion_matrix"]["matrix"]
    header = " " * 14 + "  ".join(f"{l[:12]:>12}" for l in labs)
    print(header)
    for i, row in enumerate(mat):
        print(f"{labs[i][:12]:>12}  " + "  ".join(f"{v:12d}" for v in row))

# ----------------- main -----------------
def main():
    args = parse_args()

    ref = Path(args.ref).expanduser()
    qroot = Path(args.query).expanduser()

    if not ref.exists():
        print(f"[error] reference sheet not found: {ref}", file=sys.stderr); sys.exit(2)
    if not qroot.exists():
        print(f"[error] query path not found: {qroot}", file=sys.stderr); sys.exit(2)
    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY is not set in the environment.", file=sys.stderr); sys.exit(3)

    ref_du = data_url(ref)  # pre-encode once

    rows: List[Dict[str, Any]] = []
    for gold, img in iter_query_images(qroot):
        out = classify_one(ref_du, img, model=args.model)
        rows.append({
            "image": str(img),
            "gold": gold,
            "label": out["label"],
            "confidence": out["confidence"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    # Compute metrics when we have folder-derived gold labels
    if rows and all(r["gold"] in LABELS for r in rows):
        M = compute_metrics(rows, LABELS)
        print_metrics(M, LABELS)
    else:
        print("\n[info] Gold labels not available for all items; skipped metrics.")

if __name__ == "__main__":
    main()