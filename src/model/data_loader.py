import os
import pandas as pd

def load_real_data(csv_path: str = "data/raw/legal_docs_modified.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please check the path and directory structure.")

    df = pd.read_csv(csv_path)

    required_columns = ['clause_text', 'clause_status']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: The dataset is missing the required '{col}' column.")

    df = df.dropna(subset=['clause_text', 'clause_status'])

    label_mapping_dictionary = {
        1: "High Risk", 
        0: "Low Risk"
    }
    
    df["risk_label"] = df["clause_status"].map(label_mapping_dictionary)
    
    scrambled_dataframe = df.sample(frac=1, random_state=42)
    clean_final_dataframe = scrambled_dataframe.reset_index(drop=True)
    
    return clean_final_dataframe
