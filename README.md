# ContraLegal-AI ⚖️ : Intelligent Contract Risk Analysis

**Milestone 1 Submission** • **Team:** Null Set (Ayush Kumar Singh, Isha Singh, Priyanka Gnana Karanam)

> **ContraLegal-AI** is a hybrid intelligence application designed to automate the extraction, classification, and thematic analysis of legal risk clauses within PDF contracts. It acts as an "AI Paralegal," scanning complex documents to immediately flag high-risk clauses (like unrestricted liability or hidden auto-renewals).

---

## 🚀 Key Features

- **10x Faster Review**: Upload a multi-page PDF contract and instantly review its risk profile.
- **Privacy-First Redaction**: Automatically detects and masks PII (Phone numbers, Emails) via regular expressions before any processing occurs.
- **Hybrid Intelligence Scoring**: Combines a statistical **Random Forest ML Engine** (97% F1-Score) with a deterministic **Legal Keyword Multiplier** rule-engine.
- **Thematic K-Means Clustering**: Unsupervised machine learning automatically groups similar paragraphs together (e.g., all paragraphs about "Liability" go into one bucket).
- **Export Analytics**: Export the entire color-coded risk dashboard directly to an Excel file.

---

## 🧠 System Architecture & Roles

ContraLegal-AI was architected via Separation of Concerns:

### 1. The Core ML Engine (Ayush Kumar Singh)

- **Role:** ML Architecture & Training
- **Implementation:** Transformed raw text into mathematical vectors using **TF-IDF**. Handled severe legal data imbalance using `class_weight="balanced"` to train a **Random Forest Classifier** on 21,144 annotated clauses, achieving 97% accuracy. Designed the unsupervised **K-Means Clustering** routing for thematic UI grouping.

### 2. The Data Ingestion Pipeline (Isha Singh)

- **Role:** ETL & NLP Normalization
- **Implementation:** Engineered the PDF geometric extraction pipeline using `PyMuPDF`. Implemented text normalization using the `spaCy` NLP engine (stripping punctuation and stop words) and built the complex Regex Privacy Masker to sanitize raw corporate data into machine-readable formats.

### 3. The Hybrid Engine & UI (Priyanka Gnana Karanam)

- **Role:** Deterministic Logic & Frontend Application
- **Implementation:** Bridged the gap between statistical probability and absolute legal logic by building a deterministic Legal Keyword Threat Multiplier. Integrated all the pipeline stages into a cohesive MVC routing structure and rendered the interactive elements (Dataframes, Visuals, Exporters) via the Streamlit dashboard.

---

## 📂 Project Structure

```text
ContraLegal-AI/
├── app.py                      # Main Streamlit Dashboard Application
├── src/                        # Core Source Code
│   ├── model_trainer.py        # Controller for retraining the ML models
│   ├── ui/                     # Presentation Layer (Streamlit components)
│   ├── data_pipeline/          # Data Layer (PDF parsing, text cleaning)
│   ├── inference/              # App Layer (Hybrid prediction math & engines)
│   └── model/                  # Training Layer (Loaders, Random Forest code)
├── data/                       # Datasets
│   ├── raw/                    # Raw inputs (.csv, .pdf)
│   └── processed/
├── models/                     # Saved outputs (.pkl Vectorizer and Model files)
├── report/                     # IEEE LaTeX Project Report files
└── requirements.txt            # Python dependencies (format explicitly optimized for uv)
```

---

## 💻 Local Installation & Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AyushCoder9/ContraLegal-AI.git
   cd ContraLegal-AI
   ```

2. **Install Python dependencies:**

   > _Note: Our `requirements.txt` specifically utilizes the strict PEP 508 direct wheel syntax (`.whl`) for spaCy to ensure fast, safe deployment on strictly-typed fast installers like `uv`._

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   _The application will automatically download the `en_core_web_sm` dictionary via `spacy.cli` cleanly on its first run if it detects a missing installation limit._

---

## 📈 Milestone 1 Results

Evaluated on a 20% hold-out test set (4,229 unseen clauses), the system achieved:

- **Precision:** 0.98 (High Risk)
- **Recall:** 0.97 (High Risk)
- **Weighted F1-Score:** 97.26%

Check `report/report.pdf` for our full IEEE Double-Column academic breakdown!
