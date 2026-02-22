import os
import pickle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

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

def save_artifacts(vectorizer, model, vec_path="models/vectorizer.pkl", model_path="models/model.pkl"):
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizer saved  → {vec_path}")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved       → {model_path}")
