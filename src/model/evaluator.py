
import os
import pickle

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# Labels
LABEL_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABEL_ORDER)}


def _to_int(arr):
    
    arr = np.asarray(arr)
    if arr.dtype.kind in ("U", "S", "O"):
        return np.array([LABEL_TO_INT[str(x)] for x in arr])
    return arr.astype(int)


# Conf Matrix
def evaluate_and_save_metrics(y_test, y_pred, unique_labels):
    unique_labels = sorted(unique_labels)

    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    report = classification_report(y_test, y_pred, labels=unique_labels)

    print("=" * 60)
    print("           CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"  Weighted F1-Score : {f1:.4f}")
    print("=" * 60)

    print("\n  Confusion Matrix:")
    print(f"  Labels: {unique_labels}")
    print(cm)

    os.makedirs("models", exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.title("Contract Risk — Confusion Matrix")
    plt.tight_layout()

    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()

    print("\n  ✓ Confusion matrix plot saved → models/confusion_matrix.png")

    return f1, cm


# Metrics
def compute_full_metrics(y_test, y_pred, y_proba=None, label_names=None):
    
    if label_names is None:
        label_names = LABEL_ORDER

    y_t = _to_int(y_test)
    y_p = _to_int(y_pred)
    num_classes = len(label_names)

    metrics = {
        "accuracy": round(accuracy_score(y_t, y_p), 4),
        "f1_weighted": round(f1_score(y_t, y_p, average="weighted"), 4),
        "f1_macro": round(f1_score(y_t, y_p, average="macro"), 4),
    }

    # Per-class precision and recall
    precisions = precision_score(y_t, y_p, average=None, labels=range(num_classes), zero_division=0)
    recalls = recall_score(y_t, y_p, average=None, labels=range(num_classes), zero_division=0)
    for i, name in enumerate(label_names):
        metrics[f"precision_{name}"] = round(float(precisions[i]), 4)
        metrics[f"recall_{name}"] = round(float(recalls[i]), 4)

    # ROC-AUC (requires probabilities)
    if y_proba is not None and y_proba.shape[1] == num_classes:
        try:
            y_bin = label_binarize(y_t, classes=range(num_classes))
            metrics["roc_auc_macro"] = round(
                roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"), 4
            )
        except Exception:
            metrics["roc_auc_macro"] = None
    else:
        metrics["roc_auc_macro"] = None

    return metrics


# ROC
def plot_roc_curves(
    y_test, y_proba, label_names=None, model_name="Model", save_path="models/roc_curve.png"
):
    
    if label_names is None:
        label_names = LABEL_ORDER

    y_t = _to_int(y_test)
    num_classes = len(label_names)
    y_bin = label_binarize(y_t, classes=range(num_classes))

    colors = ["#10b981", "#f59e0b", "#ef4444"]  # green, amber, red

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-class ROC
    fpr_all, tpr_all, auc_all = {}, {}, {}
    for i in range(num_classes):
        fpr_all[i], tpr_all[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_all[i] = auc(fpr_all[i], tpr_all[i])
        ax.plot(
            fpr_all[i], tpr_all[i],
            color=colors[i], lw=2,
            label=f"{label_names[i]} (AUC = {auc_all[i]:.3f})",
        )

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr, mean_tpr,
        color="#4f46e5", lw=2.5, linestyle="--",
        label=f"Macro Avg (AUC = {macro_auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} — Multi-Class ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  ✓ ROC curve saved → {save_path}")

    return {name: round(auc_all[i], 4) for i, name in enumerate(label_names)}, round(macro_auc, 4)


# Ablation
def generate_ablation_table(rf_metrics: dict, bert_metrics: dict, save_path="models/ablation_study.png"):
    
    rows = [
        ("Accuracy",               rf_metrics.get("accuracy"),         bert_metrics.get("accuracy")),
        ("Weighted F1",            rf_metrics.get("f1_weighted"),      bert_metrics.get("f1_weighted")),
        ("Macro F1",               rf_metrics.get("f1_macro"),         bert_metrics.get("f1_macro")),
        ("ROC-AUC (Macro)",        rf_metrics.get("roc_auc_macro"),    bert_metrics.get("roc_auc_macro")),
        ("", "", ""),  # spacer
        ("Precision — Low Risk",   rf_metrics.get("precision_Low Risk"),   bert_metrics.get("precision_Low Risk")),
        ("Recall — Low Risk",      rf_metrics.get("recall_Low Risk"),      bert_metrics.get("recall_Low Risk")),
        ("Precision — Medium Risk", rf_metrics.get("precision_Medium Risk"), bert_metrics.get("precision_Medium Risk")),
        ("Recall — Medium Risk",    rf_metrics.get("recall_Medium Risk"),    bert_metrics.get("recall_Medium Risk")),
        ("Precision — High Risk",   rf_metrics.get("precision_High Risk"),   bert_metrics.get("precision_High Risk")),
        ("Recall — High Risk",      rf_metrics.get("recall_High Risk"),      bert_metrics.get("recall_High Risk")),
    ]

    def _fmt(v):
        if v is None:
            return "N/A"
        if v == "":
            return ""
        return f"{v:.4f}"

    def _delta(rf_v, bert_v):
        if rf_v is None or bert_v is None or rf_v == "" or bert_v == "":
            return ""
        d = bert_v - rf_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    cell_text = []
    cell_colors = []
    for metric, rf_v, bert_v in rows:
        delta = _delta(rf_v, bert_v)
        row_colors = ["#f8f9fa", "#f8f9fa", "#f8f9fa", "#f8f9fa"]
        if delta and delta.startswith("+"):
            row_colors[3] = "#d1fae5"  # green tint
        elif delta and delta.startswith("-"):
            row_colors[3] = "#fecaca"  # red tint
        cell_text.append([metric, _fmt(rf_v), _fmt(bert_v), delta])
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_title(
        "Ablation Study: TF-IDF + Random Forest  vs  Legal-BERT",
        fontsize=14, fontweight="bold", pad=20,
    )

    col_labels = ["Metric", "Random Forest", "Legal-BERT", "Δ (Improvement)"]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#4f46e5"] * 4,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_facecolor("#4f46e5")

    # Bold metric names
    for i in range(1, len(cell_text) + 1):
        cell = table[i, 0]
        cell.set_text_props(fontweight="bold", ha="left")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Ablation study saved → {save_path}")


# Conf Matrix
def save_confusion_matrix(y_test, y_pred, label_names=None, model_name="Model", save_path="models/confusion_matrix.png"):
    
    if label_names is None:
        label_names = LABEL_ORDER
    y_t = _to_int(y_test)
    y_p = _to_int(y_pred)

    cm = confusion_matrix(y_t, y_p, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


# Save
def save_artifacts(vectorizer, model, vec_path="models/vectorizer.pkl", model_path="models/model.pkl"):
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizer saved  → {vec_path}")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved       → {model_path}")
