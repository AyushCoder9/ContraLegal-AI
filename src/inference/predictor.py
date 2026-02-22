import os
import pickle
import pandas as pd
from sklearn.cluster import KMeans

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

def predict_risk(sentences: list[str], vectorizer, model) -> pd.DataFrame:
    tfidf_matrix = vectorizer.transform(sentences)
    predictions = model.predict(tfidf_matrix)
    
    results_data = {
        "clause_text": sentences, 
        "risk_label": predictions
    }
    return pd.DataFrame(results_data)

def generate_clause_clusters(sentences: list[str], vectorizer, num_clusters: int = 4) -> tuple[list[int], dict]:
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
