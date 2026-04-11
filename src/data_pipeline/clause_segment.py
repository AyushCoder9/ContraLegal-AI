from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ClauseSegmenter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def segment(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = self.splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 5]