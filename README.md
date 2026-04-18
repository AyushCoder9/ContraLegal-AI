# ContraLegal-AI ⚖️ : The Intelligent AI Paralegal

**Team: Null Set** (Ayush Kumar Singh, Isha Singh, Priyanka Gnana Karanam)

> **ContraLegal-AI** is a comprehensive Legal-Tech platform designed to transform raw PDF contracts into actionable intelligence. By integrating **Deep Learning**, **Retrieval-Augmented Generation (RAG)**, and **Spatial NLP**, the system automates the detection, explanation, and visual marking of legal risks within complex documents.

---

## 🚀 Key Features

* **Spatial Risk Highlighting**: Generates a downloadable version of the original PDF with high-risk clauses physically highlighted in red/yellow bounding boxes using coordinate-to-text mapping.
* **Interactive Contract Chat**: A RAG-powered chatbot allowing users to query the document in plain English (e.g., *"What is the governing law in this agreement?"*).
* **Transformer-Based Classification**: Utilizes a fine-tuned **Legal-BERT** model for state-of-the-art 3-class risk detection (High, Medium, and Low).
* **AI-Powered Explainer**: Automatically explains *why* a specific clause was flagged and provides legally-sound alternative phrasing for risk mitigation.
* **Automated CI/CD**: Professional-grade MLOps using **GitHub Actions** to ensure code integrity and environment stability across the collaborative team.

---

## 🧠 System Architecture & Roles

The project follows a **Separation of Concerns** architecture, bridging Deep Learning research with production-grade Software Engineering.

### 1. Deep Learning Engine (Ayush Kumar Singh)
* **Role:** Transformer Architect & Evaluator.
* **Implementation:** Fine-tuned `nlpaueb/legal-bert-base-uncased` for 3-class risk classification. Developed the mathematical evaluation framework for ROC-AUC curves and performed ablation studies comparing Transformer performance against classical ML baselines.

### 2. Generative AI & RAG Architect (Priyanka Gnana Karanam)
* **Role:** LLM Orchestration & Semantic Search.
* **Implementation:** Engineered the **Retrieval-Augmented Generation (RAG)** pipeline. Configured **FAISS** for vector storage and designed optimized prompt templates for clause explanation and rewriting using **LangChain**.

### 3. NLP Systems & Deployment (Isha Singh)
* **Role:** Systems Integration, Spatial NLP & MLOps.
* **Implementation:** Engineered the **Spatial Highlighting Pipeline** using `PyMuPDF` to map text fragments to physical document coordinates. Implemented **Semantic Chunking** via `RecursiveCharacterTextSplitter` to optimize RAG context and architected the **GitHub Actions** CI/CD workflow.

---

## 📂 Project Structure

```text
ContraLegal-AI/
├── app.py                     # Main Streamlit Dashboard with Chat Interface
├── .github/workflows/         # CI/CD Automated Testing (python-app.yml)
├── src/
│   ├── data_pipeline/         # Semantic Chunking & NLP Normalization
│   ├── inference/             # BERT Predictor & LangChain LLM Engine
│   ├── model/                 # Legal-BERT Training & Evaluation
│   └── utils/                 # PDF Annotator (Spatial Bounding Boxes)
├── models/                    # Fine-tuned BERT weights and Vectorizers
├── data/                      # Sample Legal Contracts & Training Datasets
└── requirements.txt           # Project dependencies (Torch, LangChain, FAISS)

```
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
## 📈 Performance & Results

The transition to a **Transformer-based architecture** provided a significant leap in both classification granularity and overall accuracy:

| Metric | Baseline (Random Forest) | Final System (Legal-BERT) |
| :--- | :--- | :--- |
| **Classification** | 2-Class (Binary) | **3-Class (High/Med/Low)** |
| **F1-Score** | 97.2% | **98.4%** |
| **Intelligence** | Statistical Probability | **Semantic Context & RAG** |

---

## 🔮 Future Roadmap

While the core pipeline is fully functional, the architecture is designed to scale for enterprise-level legal needs:
* **Multi-Document Comparison**: Analyzing discrepancies and version differences between contract iterations.
* **OCR Integration**: Utilizing `Tesseract` or `EasyOCR` to process scanned, image-based legal archives.
* **On-Premise Deployment**: Containerizing the application via **Docker** to ensure absolute data sovereignty in private legal environments.

## 🎓 Acknowledgments

This project was developed as a comprehensive University-level AI initiative. We extend our gratitude to our faculty mentors for their technical guidance in transformer fine-tuning and the nuances of Legal-NLP.

---

### ✅ Deployment Status
* **Build Status:** ![CI/CD Status](https://github.com/AyushCoder9/ContraLegal-AI/actions/workflows/python-app.yml/badge.svg)
* **Technical Integrity:** The repository utilizes **GitHub Actions** to perform automated smoke tests on every commit, ensuring that all NLP modules and dependencies are stable and production-ready.

---
**Null Set © 2026** | *Innovating the future of Automated Paralegalism.*


