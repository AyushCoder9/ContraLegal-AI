from typing import List
from src.utils.spacy_loader import load_spacy_model

class ClauseSegmenter:

    def __init__(self):
        self.nlp = load_spacy_model()

    def segment(self, text: str) -> List[str]:
        if not text:
            return []

        doc= self.nlp(text)
        valid_clauses = [
            sent.text.strip() for sent in doc.sents 
            if len(sent.text.strip().split()) > 3
        ]

        return valid_clauses