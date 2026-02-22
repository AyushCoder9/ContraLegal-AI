import json
import os
import re
from typing import Set, List

class KeywordEngine:
    DEFAULT_KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), "keywords.json")

    def __init__(self, keywords_path: str | None = None):
        self.keywords_path = keywords_path or self.DEFAULT_KEYWORDS_PATH
        self._keywords = self._load_keywords()

    def _load_keywords(self) -> List[str]:
        if not os.path.exists(self.keywords_path):
            raise FileNotFoundError(f"Keyword file not found: {self.keywords_path}")
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("keywords.json must contain a JSON array of strings")
        return [str(k).strip() for k in data if k]

    @property
    def keywords(self) -> List[str]:
        return self._keywords

    def extract_keywords(self, text: str) -> Set[str]:
        if not text:
            return set()
        found: Set[str] = set()
        lowered = text.lower()
        for kw in self._keywords:
            if " " in kw:
                if kw.lower() in lowered:
                    found.add(kw)
            else:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lowered):
                    found.add(kw)
        return found
