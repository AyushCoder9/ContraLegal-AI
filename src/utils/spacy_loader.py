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
        # Streamlit Cloud installs the direct tar.gz as a module, so we must load it via the module itself.
        import en_core_web_sm
        return en_core_web_sm.load()
    except Exception:
        try:
            # Fallback for local development if the symlink exists
            return spacy.load(MODEL_NAME)
        except OSError:
            print(f"spaCy model '{MODEL_NAME}' not found.")
            print("Downloading model...")
            import subprocess
            import sys
            # Fallback for local systems missing the model
            subprocess.check_call([sys.executable, "-m", "spacy", "download", MODEL_NAME])
            return spacy.load(MODEL_NAME)