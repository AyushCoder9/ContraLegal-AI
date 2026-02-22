import json
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Literal
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .inference_engine import RiskScorer

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


def _get_ml_scores(sentences: List[str], vectorizer, model) -> List[float]:
    tfidf_matrix = vectorizer.transform(sentences)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(tfidf_matrix)[:, 1]
        return probs.tolist()
    preds = model.predict(tfidf_matrix)
    return [float(p) for p in preds]


def predict_hybrid(sentences: List[str], vectorizer, model) -> pd.DataFrame:
    scorer = RiskScorer()
    ml_scores = _get_ml_scores(sentences, vectorizer, model)
    results = [scorer.score_clause(clause, ml_score) for clause, ml_score in zip(sentences, ml_scores)]
    return pd.DataFrame(results)


def export_results(
    df: pd.DataFrame,
    output_path: str,
    fmt: Literal["json", "csv"] = "json",
) -> str:

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    export_df = df.copy()
    if "keyword_flags" in export_df.columns:
        export_df["keyword_flags"] = export_df["keyword_flags"].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else v
        )

    if fmt == "csv":
        export_df.to_csv(output_path, index=False)
    else:
        records = df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    return os.path.abspath(output_path)


def summarize_contract(clauses: List[str], top_n: int = 5) -> List[dict]:
    if len(clauses) <= top_n:
        return [{"clause_text": c, "importance_score": 1.0} for c in clauses]

    vec = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vec.fit_transform(clauses)
    scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    return [
        {"clause_text": clauses[i], "importance_score": round(float(scores[i]), 4)}
        for i in sorted(top_indices)
    ]


def generate_clause_clusters(sentences: List[str], vectorizer, num_clusters: int = 4) -> tuple[list[int], dict]:
    if len(sentences) < num_clusters:
        num_clusters = len(sentences)

    tfidf_matrix = vectorizer.transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    cluster_headings = {}
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :3]]
        heading = " / ".join(w.title() for w in top_words)
        cluster_headings[i] = heading

    return clusters.tolist(), cluster_headings
