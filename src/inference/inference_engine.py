from typing import Set

from .keyword_engine import KeywordEngine


class RiskScorer:
    ML_WEIGHT = 0.7
    RULE_WEIGHT = 0.3

    def __init__(self, keywords_path: str | None = None):
        self.keyword_engine = KeywordEngine(keywords_path)
        self._total_weight = self.keyword_engine.total_weight
        if self._total_weight == 0:
            raise ValueError("Total keyword weight is zero; cannot compute rule score.")

    def _rule_flag_score(self, matched_keywords: Set[str]) -> float:
        weights = self.keyword_engine.keyword_weights
        matched_weight = sum(weights.get(kw, 0.0) for kw in matched_keywords)
        return min(1.0, matched_weight / self._total_weight)

    def compute_hybrid_score(self, ml_score: float, matched_keywords: Set[str]) -> float:
        rule_score = self._rule_flag_score(matched_keywords)
        final = ml_score * self.ML_WEIGHT + rule_score * self.RULE_WEIGHT
        return max(0.0, min(1.0, final))

    def score_clause(self, clause: str, ml_score: float) -> dict:
        kw_matches = self.keyword_engine.extract_keywords_with_positions(clause)
        matched_terms = {m["term"] for m in kw_matches}
        final = self.compute_hybrid_score(ml_score, matched_terms)
        return {
            "clause_text": clause,
            "ml_score": round(ml_score, 4),
            "keyword_flags": [m["term"] for m in kw_matches],
            "keyword_matches": kw_matches,
            "final_score": round(final, 4),
        }
