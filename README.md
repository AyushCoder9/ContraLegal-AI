# ContraLegal-AI

Project 7: Intelligent Contract Risk Analysis (Milestone 1)

## Team Pipeline Division

This repository is structured to support 3 distinct development lanes to avoid merge conflicts:

1. **Data Pipeline (`src/data_pipeline/`)**: PDF extraction, cleaning, PII masking, and clause segmentation.
2. **ML Pipeline (`src/ml_pipeline/`)**: Target vectorization (TF-IDF), model training (Logistic Regression/Random Forest), and evaluation.
3. **Application & UI (`src/app/`)**: Streamlit user interface, integration of saved `.pkl` models for risk prediction.

## Directory Structure

- `data/`: Raw and processed datasets (ignored by git).
- `models/`: Saved `scikit-learn` `.pkl` models (ignored by git).
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `src/`: Core python packages for the pipelines.

## Setup & Installation

Follow these steps to set up the project locally:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download required models**:
   After installing the dependencies, you can download the necessary spaCy models by running the loader script directly:
   ```bash
   python src/utils/spacy_loader.py
   ```
   Or
   explicitly import and use it wherever needed in your code:
   ```python
   # In your python scripts
   from src.utils.spacy_loader import load_spacy_model
   
   nlp = load_spacy_model()
   ```
