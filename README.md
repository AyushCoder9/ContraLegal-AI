# ContraLegal-AI ⚖️ — The Intelligent AI Paralegal

**Team: Null Set** · Ayush Kumar Singh · Isha Singh · Priyanka Gnana Karanam

[![CI/CD Status](https://github.com/AyushCoder9/ContraLegal-AI/actions/workflows/python-app.yml/badge.svg)](https://github.com/AyushCoder9/ContraLegal-AI/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Legal-BERT](https://img.shields.io/badge/Model-Legal--BERT-purple)](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?logo=streamlit)](https://contralegal-ai.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **ContraLegal-AI** is a full-stack intelligent legal-tech platform that transforms raw PDF contracts into actionable risk intelligence. Powered by a fine-tuned **Legal-BERT** transformer, **Retrieval-Augmented Generation (RAG)**, and **Spatial NLP**, it automates the detection, explanation, and physical annotation of legal risks within complex documents.

---

## 🚀 Live Features

| Feature | Technology | Description |
| :--- | :--- | :--- |
| **3-Class Risk Classification** | Legal-BERT (Transformer) | Classify every clause as High / Medium / Low risk |
| **Hybrid Scoring** | BERT + Keyword Engine | Deterministic rules augment ML probabilities |
| **AI Clause Explainer** | LangChain + Gemini/Groq | Explains *why* a clause is risky in plain English |
| **AI Clause Rewriter** | LangChain + Gemini/Groq | Suggests a fairer, balanced alternative |
| **RAG Contract Chat** | FAISS + LangChain | Ask any question about your contract |
| **Spatial PDF Highlighting** | PyMuPDF | Download PDF with risks physically highlighted in Red/Amber |
| **Excel + CSV Export** | openpyxl | Full clause-level risk report |
| **K-Means Clustering** | scikit-learn | Unsupervised thematic grouping of clauses |
| **CI/CD Pipeline** | GitHub Actions | Automated smoke tests on every commit |

---

## 🧠 System Architecture & Team Roles

The project enforces strict **Separation of Concerns**, with each team member owning an independent, composable subsystem.

### 1. Deep Learning Engine · Ayush Kumar Singh
- Fine-tuned `nlpaueb/legal-bert-base-uncased` for **3-class risk classification** (Low / Medium / High)
- Designed the **keyword-density heuristic** to generate the 3-class training labels from the binary-annotated dataset
- Built the complete **evaluation framework**: multi-class ROC-AUC curves, branded confusion matrices, and the **Ablation Study** comparing RF vs. BERT
- Integrated Legal-BERT into the production predictor with an **automatic RF fallback** for high availability

### 2. Generative AI & RAG · Priyanka Gnana Karanam
- Engineered the **RAG pipeline**: `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → **FAISS** vector store → `ConversationalRetrievalChain`
- Built the **LLM Provider Factory** supporting Google Gemini 2.5, Groq Llama-3 70B, and GPT-4o-mini
- Designed optimized **prompt templates** for clause explanation, rewriting, and contract Q&A
- Implemented the **Streamlit AI Assistant panel** with chat history management

### 3. Spatial NLP & Deployment · Isha Singh
- Built the **PDF bounding box annotator** using `PyMuPDF` coordinate-mapping: High Risk → Red, Medium Risk → Amber
- Engineered **semantic chunking** via `RecursiveCharacterTextSplitter` for optimal RAG retrieval context
- Architected the **GitHub Actions CI/CD workflow** (`python-app.yml`) for automated environment testing
- Led **Streamlit Community Cloud deployment** and environment configuration

---

## 📈 Performance Results

The transition from a classical TF-IDF/Random Forest system to a Legal-BERT transformer produced measurable gains across all evaluation metrics.

| Metric | RF Baseline | Legal-BERT | Δ |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.44% | **97.01%** | +2.57% |
| **Weighted F1** | 0.9441 | **0.9702** | +0.0261 |
| **Macro F1** | 0.8901 | **0.9371** | +0.0470 |
| **ROC-AUC (Macro)** | 0.9870 | **0.9948** | +0.0078 |
| **High Risk Recall** | 73.96% | **85.94%** | **+11.98% 🔥** |

> The most important result: **+11.98% High Risk Recall** — Legal-BERT catches significantly more dangerous clauses that the Random Forest misclassifies.

---

## 📂 Project Structure

```text
ContraLegal-AI/
├── app.py                          # Streamlit dashboard (main entry point)
├── .github/
│   └── workflows/python-app.yml   # CI/CD automated smoke tests
├── src/
│   ├── model_trainer.py            # Training orchestrator (RF + BERT)
│   ├── data_pipeline/
│   │   ├── pipeline.py             # End-to-end document pipeline
│   │   ├── pdf_extractor.py        # PyMuPDF text extraction
│   │   └── clause_segment.py       # Semantic chunking
│   ├── inference/
│   │   ├── predictor.py            # Hybrid BERT/RF prediction engine
│   │   ├── llm_engine.py           # LangChain RAG, explainer, rewriter
│   │   ├── inference_engine.py     # Keyword scoring engine
│   │   └── keyword_engine.py       # Legal keyword library
│   ├── model/
│   │   ├── bert_trainer.py         # Legal-BERT fine-tuning (local)
│   │   ├── trainer.py              # Random Forest trainer
│   │   ├── evaluator.py            # ROC-AUC, confusion matrix, ablation
│   │   └── data_loader.py          # 3-class label heuristic
│   └── utils/
│       ├── pdf_annotator.py        # Spatial bounding box highlighting
│       └── dummy_contract.py       # Demo contract for live testing
├── models/
│   ├── legal_bert/                 # Fine-tuned BERT weights
│   ├── model.pkl / vectorizer.pkl  # RF baseline (fallback)
│   ├── ablation_study.png          # RF vs BERT comparison table
│   ├── roc_bert.png / roc_rf.png   # Multi-class ROC curves
│   └── cm_bert.png / cm_rf.png     # Confusion matrices
├── notebooks/
│   └── train_legal_bert_colab.py   # GPU training script for Google Colab
├── report/
│   ├── report.tex                  # IEEE-format LaTeX paper
│   └── references.bib
└── requirements.txt
```

---

## 💻 Local Setup

### Prerequisites
- Python 3.10+
- At least one API key: [Google AI Studio](https://aistudio.google.com/) (free) or [Groq](https://console.groq.com/) (free)

### Installation

```bash
git clone https://github.com/AyushCoder9/ContraLegal-AI.git
cd ContraLegal-AI

pip install -r requirements.txt
```

### Run (without training — uses pre-trained BERT from repo)

```bash
streamlit run app.py
```

### Train from scratch (optional)

```bash
python -m src.model_trainer
```

> **Note:** Legal-BERT was trained on Google Colab (free T4 GPU, ~15 min). Use `notebooks/train_legal_bert_colab.py` if you need to retrain. The script auto-downloads and zips the model for local deployment.

---

## 🔮 Future Roadmap

- **CUAD Full Fine-tuning**: Expand training corpus to the full 500-contract CUAD dataset for improved generalization
- **Multi-Document Comparison**: Detect discrepancies and obligation shifts between contract versions
- **OCR Integration**: Process scanned/image-based legal documents via Tesseract
- **Docker Containerization**: On-premise deployment for data sovereignty in private legal environments
- **Jurisdiction-Aware Analysis**: Adapt risk scoring based on governing law jurisdiction

---

## 🎓 Acknowledgments

This project was developed as a comprehensive University AI initiative. We thank the creators of the [CUAD dataset](https://www.atticusprojectai.org/cuad), [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased), and the [LangChain](https://langchain.com/) and [Streamlit](https://streamlit.io/) teams.

---

<div align="center">
  <b>Null Set © 2026</b><br/>
  <i>Automating the Future of Paralegalism.</i>
</div>
