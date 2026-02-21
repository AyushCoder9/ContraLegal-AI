import fitz  # from PyMuPDF

class PDFExtractor:

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""