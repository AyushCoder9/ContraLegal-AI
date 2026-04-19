<div align="center">

![ContraLegal AI Banner](https://readme-typing-svg.demolab.com?font=Montserrat&size=45&pause=1000&color=F97316&center=true&vCenter=true&width=1000&height=100&lines=CONTRALEGAL+AI;AI-POWERED+LEGAL+ANALYST;TRANSFORMER-BASED+RISK+ANALYSIS;RAG-POWERED+CONTRACT+INTELLIGENCE)

**Team Null Set**
Ayush Kumar Singh | Isha Singh | Priyanka Gnana Karanam
*Newton School Of Technology*

---

[![CI/CD Status](https://github.com/AyushCoder9/ContraLegal-AI/actions/workflows/python-app.yml/badge.svg)](https://github.com/AyushCoder9/ContraLegal-AI/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Model Architecture](https://img.shields.io/badge/Architecture-Legal--BERT-6D28D9?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
[![Vector DB](https://img.shields.io/badge/Vector--DB-FAISS-00599C?style=flat&logo=scipy&logoColor=white)](https://github.com/facebookresearch/faiss)
[![LLM Proxy](https://img.shields.io/badge/LLM-Gemini_|_Groq-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Spatial NLP](https://img.shields.io/badge/Spatial--NLP-PyMuPDF-FFD43B?style=flat)](https://github.com/pymupdf/PyMuPDF)
[![Deployment](https://img.shields.io/badge/Platform-Streamlit_Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://contralegal-ai.streamlit.app)

</div>

---

### Project Overview

ContraLegal AI is an autonomous legal intelligence platform engineered to transform unstructured contract data into actionable risk distributions. By synthesizing **Legal-BERT** transformer architectures with **Retrieval-Augmented Generation (RAG)**, the system provides granular multi-class risk scoring, automated redrafting, and spatial PDF highlights to eliminate manual bottlenecks in enterprise legal review.

---

### Core Engineering Capabilities

| Capability | Orchestration | Technical Specification |
| :--- | :--- | :--- |
| **Trimodal Classification** | Legal-BERT | High, Medium, and Low-intensity risk granularity |
| **Dynamic Scoring** | Hybrid Logic | Fusion of Transformer probabilities and deterministic keyword heuristics |
| **Explainable AI** | RAG + LangChain | Root-cause analysis of flagged clauses in professional nomenclature |
| **Strategic Redrafting** | Generative AI | Automated generation of balanced, legally-sound alternative phrasing |
| **Conversational Querying** | FAISS Vector Store | Real-time, document-grounded Q&A for complex legal inquiries |
| **Spatial Annotation** | PyMuPDF API | Physical coordinate-to-text mapping for in-situ PDF highlighting |
| **Relational Data Export** | openpyxl | Structured synthesis of risk distributions in Excel and CSV formats |
| **Thematic Clustering** | Scikit-Learn | Unsupervised K-Means grouping of obligation-specific clauses |

---

### Engineering Hierarchy & Contributions

The platform is built upon a high-concurrency architecture with a strict separation of concerns across research and deployment layers.

#### Deep Learning & Transformation | Ayush Kumar Singh
*   Fine-tuned the `nlpaueb/legal-bert-base-uncased` transformer using a weighted-trainer objective for imbalanced class distribution.
*   Engineered the 3-class quantitative heuristic for synthetic label generation spanning over 21,000 samples.
*   Developed the formal ablation study and multi-class ROC-AUC evaluation suite to validate transformer superiority over statistical baselines.

#### Generative AI & RAG Orchestration | Priyanka Gnana Karanam
*   Architected the retrieval-augmented generation pipeline utilizing **FAISS** for vectorized similarity search.
*   Engineered the LLM Provider Factory, enabling seamless interoperability between Google Gemini, Groq, and OpenAI.
*   Validated prompt-engineering strategies for deterministic clause synthesis and document-grounded conversational flows.

#### Spatial NLP & Deployment Systems | Isha Singh
*   Engineered the spatial highlighting engine using **PyMuPDF** to perform physical document marking via bounding-box coordinate tracking.
*   Implemented semantic document segmentation to optimize transformer context windows.
*   Architected the automated CI/CD infrastructure via GitHub Actions for continuous environment validation.

---

### Quantitative Performance Matrix

The integration of transformer architectures resulted in a fundamental shift in both classification precision and recall intensity.

| Metrical Indicator | Random Forest Baseline | Legal-BERT Transformer | Improvement (Δ) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.44% | **97.01%** | +2.57% |
| **Weighted F1** | 0.9441 | **0.9702** | +2.76% |
| **Macro F1** | 0.8901 | **0.9371** | +5.28% |
| **ROC-AUC (Macro)** | 0.9870 | **0.9948** | +0.79% |
| **High Risk Recall** | 73.96% | **85.94%** | **+11.98%** |

*Note: The 11.98% surge in High Risk Recall represents the most critical engineering milestone, ensuring safety in mission-critical legal review.*

---

### Global Repository Schema

```text
ContraLegal-AI/
├── app.py                          # Streamlit Production Environment
├── .github/workflows/              # Automated CI/CD (python-app.yml)
├── src/
│   ├── model_trainer.py            # Phase-integrated Training Orchestrator
│   ├── data_pipeline/              # Semantic Extraction & Normalization
│   ├── inference/
│   │   ├── predictor.py            # Trimodal Detection Engine (BERT/RF)
│   │   ├── llm_engine.py           # RAG Orchestrator & Conversational Layer
│   │   └── keyword_engine.py       # Deterministic Rule Definitions
│   ├── model/
│   │   ├── bert_trainer.py         # Transformer Fine-tuning Suite
│   │   └── evaluator.py            # Quantitative Performance Metrics
│   └── utils/
│       └── pdf_annotator.py        # Spatial Coordinate Highlighting
├── models/
│   ├── legal_bert/                 # Fine-tuned Weights (nlpaueb)
│   └── ablation_study.png          # Baseline vs. Transformer Visualization
├── notebooks/
│   └── train_legal_bert_colab.py   # GPU-accelerated Training Script
└── report/
    ├── report.pdf                  # Formally Published IEEE Paper
    └── report.tex                  # Scientific Manuscript Source
```

---

### Operational Deployment

#### Environment Initialization

```bash
git clone https://github.com/AyushCoder9/ContraLegal-AI.git
pip install -r requirements.txt
```

#### Application Execution

To initiate the production dashboard with the global pre-trained model:
```bash
streamlit run app.py
```

#### Analytical Training (Optional)

To execute the full analytical pipeline and regenerate performance artifacts:
```bash
python -m src.model_trainer
```

---

### Scientific Publication

The technical methodology, algorithmic decisions, and empirical evaluations are documented in the associated IEEE conference-format manuscript located in the `report/` directory.

---

<div align="center">
  <b>Null Set | 2026</b><br/>
  <i>Engineered for Legal Precision.</i>
</div>
