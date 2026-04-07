"""
============================================================================
ContraLegal-AI — Legal-BERT Fine-Tuning (Google Colab Script)
============================================================================

HOW TO USE THIS ON GOOGLE COLAB:
1. Open https://colab.research.google.com
2. Click Runtime → Change runtime type → GPU (T4 is free)
3. Create a new notebook
4. Paste this ENTIRE file into ONE cell and run it
5. When prompted, upload your 'legal_docs_modified.csv' file
6. Training will run (~15-20 min on T4 GPU)
7. Download the 'legal_bert_model.zip' file when it's done

AFTER COLAB - HOW TO GET THE MODEL BACK:
1. The script will auto-download 'legal_bert_model.zip'
2. Unzip it into your project: models/legal_bert/
   Your folder should look like:
     models/
       legal_bert/
         config.json
         model.safetensors
         tokenizer.json
         tokenizer_config.json
         special_tokens_map.json
         vocab.txt
3. Run: python -m src.model_trainer
   This will generate the ROC curves + ablation study with your trained model.
4. Run: streamlit run app.py
   The app will auto-detect and use Legal-BERT!

============================================================================
"""

# ── Cell 1: Install Dependencies ──────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "accelerate", "scikit-learn", "pandas", "tqdm"])

# ── Cell 2: Upload CSV ───────────────────────────────────────────────────
import os, re, random, shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

print("📂 Upload your legal_docs_modified.csv file:")
try:
    from google.colab import files
    uploaded = files.upload()
    csv_filename = list(uploaded.keys())[0]
except ImportError:
    # Not on Colab — use a local path
    csv_filename = "data/raw/legal_docs_modified.csv"

df = pd.read_csv(csv_filename)
print(f"✅ Loaded {len(df):,} rows")
print(df.columns.tolist())

# ── Cell 3: 3-Class Data Labeling ────────────────────────────────────────
# Embedded keywords for the risk heuristic (same as project's keywords.json)
RISK_KEYWORDS = {
    "indemnity": 0.9, "indemnify": 0.9, "indemnification": 0.9, "indemnified": 0.8,
    "termination for convenience": 1.0, "termination for cause": 0.9,
    "constructive termination": 0.9, "notice of termination": 0.8, "terminate": 0.7,
    "limitation of liability": 0.95, "liability": 0.7,
    "liquidated damages": 0.9, "consequential damages": 0.9, "punitive damages": 0.85,
    "confidentiality": 0.7, "confidential": 0.6, "non-disclosure": 0.75,
    "trade secret": 0.85, "trade secrets": 0.85,
    "intellectual property": 0.75, "intellectual property rights": 0.8,
    "patent": 0.7, "copyright": 0.65, "proprietary": 0.6,
    "work for hire": 0.75, "sublicense": 0.7,
    "material breach": 0.95, "breach": 0.8, "default": 0.75,
    "dispute resolution": 0.65, "binding arbitration": 0.8, "arbitration": 0.7,
    "injunction": 0.85, "injunctive relief": 0.9, "specific performance": 0.8,
    "assignment": 0.65, "assign": 0.6, "change of control": 0.9, "successor": 0.6,
    "force majeure": 0.6, "governing law": 0.5, "jurisdiction": 0.55,
    "choice of law": 0.55, "class action waiver": 0.85,
    "statute of limitations": 0.75, "non-compete": 0.9, "non-solicitation": 0.8,
    "waiver": 0.65, "negligence": 0.8, "gross negligence": 0.95, "warranty": 0.7,
}

def determine_risk(row):
    status = row["clause_status"]
    text = str(row["clause_text"]).strip().lower()

    if status == 0:
        return "Low Risk"

    # Calculate keyword weight for risky clauses
    total_weight = 0.0
    for kw, w in RISK_KEYWORDS.items():
        if kw in text:
            total_weight += w

    if total_weight >= 1.5 or (len(text) > 400 and total_weight >= 1.0):
        return "High Risk"
    else:
        return "Medium Risk"

df = df.dropna(subset=["clause_text", "clause_status"])
df["risk_label"] = df.apply(determine_risk, axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n📊 3-Class Distribution:")
print(df["risk_label"].value_counts())

# ── Cell 4: Constants & Setup ────────────────────────────────────────────
SEED = 42
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 256
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
OUTPUT_DIR = "legal_bert"

LABEL_MAP = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── Cell 5: Prepare Data ────────────────────────────────────────────────
texts = df["clause_text"].astype(str).tolist()
labels = df["risk_label"].map(LABEL_MAP).tolist()

# 70/15/15 stratified split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=SEED, stratify=labels
)
val_frac = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_frac, random_state=SEED, stratify=y_trainval
)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Tokenize
print(f"\n📝 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

train_dataset = ClauseDataset(train_enc, y_train)
val_dataset   = ClauseDataset(val_enc, y_val)
test_dataset  = ClauseDataset(test_enc, y_test)

# Class weights
cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
class_weights = torch.tensor(cw, dtype=torch.float32)
print(f"  Class weights: Low={cw[0]:.3f}  Med={cw[1]:.3f}  High={cw[2]:.3f}")

# ── Cell 6: Model & Training ────────────────────────────────────────────
print(f"\n🤖 Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_MAP
)

class WeightedTrainer(Trainer):
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

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=(device == "cuda"),
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

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n🚀 Training started...\n")
trainer.train()

# ── Cell 7: Evaluate on Test Set ─────────────────────────────────────────
print(f"\n{'='*60}")
print("  TEST SET EVALUATION")
print(f"{'='*60}")

test_output = trainer.predict(test_dataset)
test_logits = test_output.predictions
test_proba = torch.nn.functional.softmax(
    torch.tensor(test_logits, dtype=torch.float32), dim=-1
).numpy()
test_preds = np.argmax(test_logits, axis=-1)
y_test_arr = np.array(y_test)

label_names = ["Low Risk", "Medium Risk", "High Risk"]
print(classification_report(y_test_arr, test_preds, target_names=label_names))

# ── Cell 8: Save & Download Model ───────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Clean up checkpoints
ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir, ignore_errors=True)

print(f"\n✅ Model saved to {OUTPUT_DIR}/")

# Also save the test metrics for the ablation study
import json
test_metrics = {
    "y_test": y_test_arr.tolist(),
    "y_pred": test_preds.tolist(),
    "y_proba": test_proba.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)
print("✅ Test metrics saved (for ablation study)")

# Zip and download
shutil.make_archive("legal_bert_model", "zip", ".", OUTPUT_DIR)
print("\n📦 Model zipped → legal_bert_model.zip")

try:
    from google.colab import files
    files.download("legal_bert_model.zip")
    print("\n🎉 Download started! After download:")
    print("   1. Unzip into your project: models/legal_bert/")
    print("   2. Run: python -m src.model_trainer")
    print("   3. Run: streamlit run app.py")
except ImportError:
    print("\n📁 Find your model at: legal_bert_model.zip")

print("\n" + "="*60)
print("  ALL DONE! 🎉")
print("="*60)
