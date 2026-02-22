import os
from typing import Set, List

from .keyword_engine import KeywordEngine


class RiskScorer:

    # Weighting constants – can be tuned later
    ML_WEIGHT = 0.7
    RULE_WEIGHT = 0.3

    def __init__(self, keywords_path: str | None = None):
        self.keyword_engine = KeywordEngine(keywords_path)
        self.total_keywords = len(self.keyword_engine.keywords)
        if self.total_keywords == 0:
            raise ValueError("Keyword list is empty; cannot compute rule score.")

    def _rule_flag_score(self, matched_keywords: Set[str]) -> float:
        return len(matched_keywords) / self.total_keywords

    def compute_hybrid_score(self, ml_score: float, matched_keywords: Set[str]) -> float:
        rule_score = self._rule_flag_score(matched_keywords)
        final_score = ml_score * self.ML_WEIGHT + rule_score * self.RULE_WEIGHT
        return max(0.0, min(1.0, final_score))

    def score_clause(self, clause: str, ml_score: float) -> dict:
        matched = self.keyword_engine.extract_keywords(clause)
        final = self.compute_hybrid_score(ml_score, matched)
        return {
            "clause_text": clause,
            "ml_score": ml_score,
            "keyword_flags": list(matched),
            "final_score": final,
        }
