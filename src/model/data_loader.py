import os
import pandas as pd
from src.inference.keyword_engine import KeywordEngine

def load_real_data(csv_path: str = "data/raw/legal_docs_modified.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please check the path and directory structure.")

    df = pd.read_csv(csv_path)

    required_columns = ['clause_text', 'clause_status']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: The dataset is missing the required '{col}' column.")

    df = df.dropna(subset=['clause_text', 'clause_status'])

    # Risk Heuristic
    keyword_engine = KeywordEngine()
    
    def determine_risk(row):
        status = row["clause_status"]
        text = str(row["clause_text"]).strip()
        
        if status == 0:
            return "Low Risk"
            
        # Medium vs High
        kw_matches = keyword_engine.extract_keywords_with_positions(text)
        
        # Base limit
        total_kw_weight = sum(m["weight"] for m in kw_matches)
        
        # Condition
        if total_kw_weight >= 1.5 or (len(text) > 400 and total_kw_weight >= 1.0):
            return "High Risk"
        else:
            return "Medium Risk"

    df["risk_label"] = df.apply(determine_risk, axis=1)
    
    scrambled_dataframe = df.sample(frac=1, random_state=42)
    clean_final_dataframe = scrambled_dataframe.reset_index(drop=True)
    
    return clean_final_dataframe
