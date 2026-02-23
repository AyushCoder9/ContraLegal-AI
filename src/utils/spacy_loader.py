import spacy
import subprocess
import sys

MODEL_NAME = "en_core_web_sm"

def load_spacy_model():
    """
    Load spaCy model.
    If not installed, download automatically.
    """
    try:
        return spacy.load(MODEL_NAME)

    except OSError:
        print(f"spaCy model '{MODEL_NAME}' not found.")
        print("Please ensure it is installed via requirements.txt")
        # In Streamlit Cloud, subprocess downloads often fail due to permissions.
        # It must be installed via the URL in requirements.txt.
        raise RuntimeError(f"Model {MODEL_NAME} is missing. Add its URL to requirements.txt.")