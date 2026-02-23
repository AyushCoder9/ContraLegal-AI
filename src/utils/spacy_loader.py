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
        print(f"spaCy model '{MODEL_NAME}' not found. Downloading via spacy.cli...")
        from spacy.cli import download
        download(MODEL_NAME)
        return spacy.load(MODEL_NAME)