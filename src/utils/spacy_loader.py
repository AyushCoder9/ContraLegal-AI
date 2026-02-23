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
        import en_core_web_sm
        return en_core_web_sm.load()
    except Exception as e:
        print(f"Failed to load via module, attempting spacy.load: {e}")
        return spacy.load(MODEL_NAME)