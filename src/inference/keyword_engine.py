import json
import os
import re
from typing import Dict, List, Set, Tuple


class KeywordEngine:
    DEFAULT_KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), "keywords.json")

    def __init__(self, keywords_path: str | None = None):
        self.keywords_path = keywords_path or self.DEFAULT_KEYWORDS_PATH
        self._keyword_weights: Dict[str, float] = self._load_keywords()

    def _load_keywords(self) -> Dict[str, float]:
        if not os.path.exists(self.keywords_path):
            raise FileNotFoundError(f"Keyword file not found: {self.keywords_path}")
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("keywords.json must contain a JSON array")
        weights: Dict[str, float] = {}
        for entry in data:
            if isinstance(entry, dict):
                term = str(entry.get("term", "")).strip()
                weight = float(entry.get("weight", 0.5))
            else:
                term = str(entry).strip()
                weight = 0.5
            if term:
                weights[term] = weight
        return weights

    @property
    def keywords(self) -> List[str]:
        return list(self._keyword_weights.keys())

    @property
    def keyword_weights(self) -> Dict[str, float]:
        return self._keyword_weights

    @property
    def total_weight(self) -> float:
        return sum(self._keyword_weights.values())

    def _build_pattern(self, term: str) -> re.Pattern:
        escaped = re.escape(term.lower())
        if " " in term:
            return re.compile(escaped)
        return re.compile(r"\b" + escaped + r"\b")

    def extract_keywords(self, text: str) -> Set[str]:
        if not text:
            return set()
        lowered = text.lower()
        return {
            term
            for term in self._keyword_weights
            if self._build_pattern(term).search(lowered)
        }

    def extract_keywords_with_positions(self, text: str) -> List[Dict]:
        if not text:
            return []
        lowered = text.lower()
        matches: List[Dict] = []
        seen_terms: Set[str] = set()
        for term, weight in self._keyword_weights.items():
            if term in seen_terms:
                continue
            pattern = self._build_pattern(term)
            m = pattern.search(lowered)
            if m:
                matches.append(
                    {
                        "term": term,
                        "weight": weight,
                        "start": m.start(),
                        "end": m.end(),
                    }
                )
                seen_terms.add(term)
        matches.sort(key=lambda x: x["start"])
        return matches
