from typing import List
import os

from src.data_pipeline.pdf_extractor import PDFExtractor
from src.data_pipeline.text_cleaner import TextCleaner
from src.data_pipeline.privacy_masker import PrivacyMasker
from src.data_pipeline.clause_segment import ClauseSegmenter

class DataPipeline:
    def __init__(self):
        self.extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.segmenter = ClauseSegmenter()
        self.masker = PrivacyMasker()

    def process_document(self, pdf_path: str) -> List[str]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

        print(f"[{os.path.basename(pdf_path)}] 1. Extracting text...")
        raw_text = self.extractor.extract_text(pdf_path)

        print(f"[{os.path.basename(pdf_path)}] 2. Cleaning text...")
        cleaned_text = self.cleaner.clean(raw_text)

        print(f"[{os.path.basename(pdf_path)}] 3. Segmenting clauses...")
        clauses = self.segmenter.segment(cleaned_text)

        print(f"[{os.path.basename(pdf_path)}] 4. Masking sensible entities...")
        masked_clauses = self.masker.mask_pii(clauses)

        print(f"[{os.path.basename(pdf_path)}] Success! Extracted {len(clauses)} clauses.")
        return masked_clauses

if __name__ == "__main__":
    pipeline = DataPipeline()
    sample_path = "data/raw/sample.pdf"
    
    if os.path.exists(sample_path):
        results = pipeline.process_document(sample_path)
        print("\n--- Sample Clauses ---")
        for i, clause in enumerate(results[:3]): 
            print(f"Clause {i+1}: {clause[:100]}...\n")
    else:
        print(f"No sample document found at {sample_path}. Add one to test the script manually.")