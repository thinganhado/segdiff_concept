# gpt4omini_metrics.py
# Run: python viz_metrics.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = "/home/opc/difussionXAI/LLM_outputs"
# -------------------- INPUTS (paste your numbers) --------------------
overall = {
    "accuracy": 0.714,
    "precision_macro": 0.704, "precision_micro": 0.714,
    "recall_macro": 0.665,    "recall_micro": 0.714,
    "f1_macro": 0.670,        "f1_micro": 0.714,
}

per_class = {
    "fogging":               {"precision": 0.737, "recall": 0.700, "f1": 0.718, "support": 20},
    "formant attenuation":   {"precision": 0.667, "recall": 0.400, "f1": 0.500, "support": 10},
    "imaging":               {"precision": 0.708, "recall": 0.895, "f1": 0.791, "support": 19},
}

labels = ["fogging", "formant attenuation", "imaging"]
confusion = np.array([
    [14, 1, 5],
    [ 4, 4, 2],
    [ 1, 1,17],
])  # rows = gold, cols = predicted

# -------------------- helpers --------------------
def _annotate_bars(ax, rects):
    for r in rects:
        height = r.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(r.get_x() + r.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

def _ensure_dir(d: str):
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)

# -------------------- 1) Overall (macro) --------------------
fig1 = plt.figure(figsize=(6, 4))
ax1 = plt.gca()
cats = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
vals = [
    overall["accuracy"],
    overall["precision_macro"],
    overall["recall_macro"],
    overall["f1_macro"],
]
x = np.arange(len(cats))
rects = ax1.bar(x, vals)
ax1.set_xticks(x, cats, rotation=15, ha="right")
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("Score")
ax1.set_title("Overall (Macro) Metrics")
_annotate_bars(ax1, rects)
plt.tight_layout()
if SAVE_DIR:
    _ensure_dir(SAVE_DIR)
    plt.savefig(f"{SAVE_DIR}/overall_macro.png", dpi=160)

# -------------------- 2) Overall (micro) --------------------
fig2 = plt.figure(figsize=(6, 4))
ax2 = plt.gca()
cats2 = ["Accuracy", "Precision (micro)", "Recall (micro)", "F1 (micro)"]
vals2 = [
    overall["accuracy"],
    overall["precision_micro"],
    overall["recall_micro"],
    overall["f1_micro"],
]
x2 = np.arange(len(cats2))
rects2 = ax2.bar(x2, vals2)
ax2.set_xticks(x2, cats2, rotation=15, ha="right")
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Score")
ax2.set_title("Overall (Micro) Metrics")
_annotate_bars(ax2, rects2)
plt.tight_layout()
if SAVE_DIR:
    plt.savefig(f"{SAVE_DIR}/overall_micro.png", dpi=160)

# -------------------- 3) Per-class grouped bars --------------------
fig3 = plt.figure(figsize=(7.5, 4.8))
ax3 = plt.gca()

pc_prec = [per_class[k]["precision"] for k in labels]
pc_rec  = [per_class[k]["recall"]    for k in labels]
pc_f1   = [per_class[k]["f1"]        for k in labels]
pc_sup  = [per_class[k]["support"]   for k in labels]

idx = np.arange(len(labels))
w = 0.25

r1 = ax3.bar(idx - w, pc_prec, width=w, label="Precision")
r2 = ax3.bar(idx,       pc_rec,  width=w, label="Recall")
r3 = ax3.bar(idx + w,   pc_f1,   width=w, label="F1")

ax3.set_xticks(idx, labels, rotation=15, ha="right")
ax3.set_ylim(0, 1.0)
ax3.set_ylabel("Score")
ax3.set_title("Per-class Metrics (support in parentheses)")
ax3.legend()

# add per-class supports under x-ticks
xtick_lbls = [f"{lab} ({sup})" for lab, sup in zip(labels, pc_sup)]
ax3.set_xticklabels(xtick_lbls, rotation=15, ha="right")

# annotate bars
_annotate_bars(ax3, r1); _annotate_bars(ax3, r2); _annotate_bars(ax3, r3)

plt.tight_layout()
if SAVE_DIR:
    plt.savefig(f"{SAVE_DIR}/per_class.png", dpi=160)

# -------------------- 4) Confusion matrix heatmap --------------------
fig4 = plt.figure(figsize=(5.8, 5.2))
ax4 = plt.gca()
im = ax4.imshow(confusion)
ax4.set_title("Confusion Matrix (gold rows × pred cols)")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Gold")
ax4.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
ax4.set_yticks(np.arange(len(labels)), labels)

# annotate cells with counts
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax4.text(j, i, str(confusion[i, j]),
                 ha="center", va="center")

plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
plt.tight_layout()
if SAVE_DIR:
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", dpi=160)

# Show all figures
plt.show()