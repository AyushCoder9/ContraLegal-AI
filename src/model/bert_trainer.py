"""
ContraLegal-AI — Legal-BERT Fine-Tuning (3-Class Risk Classification)

Fine-tunes nlpaueb/legal-bert-base-uncased on a 3-class clause risk dataset.
Classes: Low Risk (0), Medium Risk (1), High Risk (2)

Works on CPU, MPS (Apple Silicon), and CUDA (GPU).
For best performance, use Google Colab with a free GPU runtime.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 256
OUTPUT_DIR = "models/legal_bert"

LABEL_MAP = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def set_seed(seed: int = SEED):
    """Pin every random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Weighted Trainer (handles class imbalance)
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Custom HF Trainer that applies class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train_bert(
    df: pd.DataFrame,
    text_col: str = "clause_text",
    label_col: str = "risk_label",
    output_dir: str = OUTPUT_DIR,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = MAX_LENGTH,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Fine-tune Legal-BERT for 3-class clause risk classification.

    Returns:
        y_test (np.ndarray): Integer test labels
        y_pred (np.ndarray): Integer predicted labels
        y_proba (np.ndarray): (n, 3) probability matrix
        label_names (list[str]): ["Low Risk", "Medium Risk", "High Risk"]
    """
    set_seed(SEED)

    print(f"\n{'=' * 60}")
    print("  LEGAL-BERT FINE-TUNING — 3-Class Risk Classification")
    print(f"{'=' * 60}")

    # --- Encode labels ---
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].map(LABEL_MAP).tolist()

    # --- Stratified Split: 70% train / 15% val / 15% test ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=SEED, stratify=labels,
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=SEED, stratify=y_trainval,
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    use_fp16 = device == "cuda"

    # --- Tokenize ---
    print(f"  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    train_dataset = ClauseDataset(train_enc, y_train)
    val_dataset   = ClauseDataset(val_enc, y_val)
    test_dataset  = ClauseDataset(test_enc, y_test)

    # --- Class weights (handles imbalance) ---
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"  Class weights: Low={cw[0]:.3f}  Med={cw[1]:.3f}  High={cw[2]:.3f}")

    # --- Load Model ---
    print(f"  Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    # --- Training Arguments ---
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=use_fp16,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_weighted": f1_score(labs, preds, average="weighted"),
            "f1_macro": f1_score(labs, preds, average="macro"),
        }

    # --- Train ---
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n  Training started...\n")
    trainer.train()

    # --- Evaluate on Test Set ---
    print(f"\n{'=' * 60}")
    print("  TEST SET EVALUATION")
    print(f"{'=' * 60}")

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_proba = torch.nn.functional.softmax(
        torch.tensor(test_logits, dtype=torch.float32), dim=-1
    ).numpy()
    test_preds = np.argmax(test_logits, axis=-1)

    y_test_arr = np.array(y_test)
    label_names = ["Low Risk", "Medium Risk", "High Risk"]

    print(classification_report(y_test_arr, test_preds, target_names=label_names))

    # --- Save Model + Tokenizer ---
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n  ✓ Model + tokenizer saved → {output_dir}/")

    # Clean up checkpoints to save space
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print("  ✓ Training checkpoints cleaned up")

    return y_test_arr, test_preds, test_proba, label_names
