"""
ContraLegal-AI — Model Training Orchestrator

Trains both the Random Forest baseline and (optionally) Legal-BERT,
then runs the full comparative evaluation.

Usage:
    python -m src.model_trainer
"""

from src.model.data_loader import load_real_data
from src.model.trainer import train_model
from src.model.evaluator import (
    evaluate_and_save_metrics,
    save_artifacts,
    compute_full_metrics,
    plot_roc_curves,
    save_confusion_matrix,
    generate_ablation_table,
)

if __name__ == "__main__":
    print("\n🔧  ContraLegal-AI — Model Training Pipeline (V2)")
    print("─" * 60)

    # ---- Load 3-class dataset ----
    df = load_real_data()
    print(f"  Dataset loaded: {len(df):,} samples")
    print(f"  Label distribution:\n{df['risk_label'].value_counts().to_string()}\n")

    # ==================================================================
    # PHASE 1: Random Forest Baseline
    # ==================================================================
    print("━" * 60)
    print("  PHASE 1 — TF-IDF + Random Forest (Baseline)")
    print("━" * 60)

    vectorizer, model, y_test_rf, y_pred_rf, y_proba_rf, unique_labels = train_model(df)

    evaluate_and_save_metrics(y_test_rf, y_pred_rf, unique_labels)
    save_artifacts(vectorizer, model)

    # ROC for RF
    rf_metrics = compute_full_metrics(y_test_rf, y_pred_rf, y_proba_rf, unique_labels)
    plot_roc_curves(y_test_rf, y_proba_rf, unique_labels, "Random Forest", "models/roc_rf.png")
    save_confusion_matrix(y_test_rf, y_pred_rf, unique_labels, "Random Forest", "models/cm_rf.png")

    print(f"\n  RF Metrics: {rf_metrics}\n")

    # ==================================================================
    # PHASE 2: Legal-BERT (if PyTorch is available)
    # ==================================================================
    bert_metrics = None
    try:
        import torch
        print("━" * 60)
        print("  PHASE 2 — Legal-BERT Fine-Tuning")
        print("━" * 60)

        from src.model.bert_trainer import train_bert

        y_test_bert, y_pred_bert, y_proba_bert, bert_label_names = train_bert(df)

        bert_metrics = compute_full_metrics(y_test_bert, y_pred_bert, y_proba_bert, bert_label_names)
        plot_roc_curves(y_test_bert, y_proba_bert, bert_label_names, "Legal-BERT", "models/roc_bert.png")
        save_confusion_matrix(y_test_bert, y_pred_bert, bert_label_names, "Legal-BERT", "models/cm_bert.png")

        print(f"\n  BERT Metrics: {bert_metrics}\n")

    except ImportError:
        print("\n  ⚠ PyTorch not installed — skipping BERT training.")
        print("    Train on Google Colab using: notebooks/train_legal_bert_colab.py")
        print("    Then copy models/legal_bert/ to this project.\n")

    # ==================================================================
    # PHASE 3: Ablation Study Comparison
    # ==================================================================
    if bert_metrics is not None:
        print("━" * 60)
        print("  PHASE 3 — Ablation Study: RF vs Legal-BERT")
        print("━" * 60)
        generate_ablation_table(rf_metrics, bert_metrics, "models/ablation_study.png")
        print("\n  ✓ Full comparative evaluation complete!")
    else:
        print("\n  ℹ Ablation study skipped (no BERT model).")
        print("    After training BERT on Colab, re-run this script to generate the comparison.\n")

    print("\n✅  Training pipeline complete. Artifacts ready in models/\n")
