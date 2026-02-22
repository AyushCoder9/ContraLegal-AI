from src.model.data_loader import load_real_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_and_save_metrics, save_artifacts

if __name__ == "__main__":
    print("\n🔧  ContraLegal-AI — Model Trainer")
    print("─" * 40)

    df = load_real_data()
    print(f"  Dataset loaded: {len(df)} samples")
    print(f"  Label distribution:\n{df['risk_label'].value_counts().to_string()}\n")

    vectorizer, model, y_test, y_pred, unique_labels = train_model(df)
    
    evaluate_and_save_metrics(y_test, y_pred, unique_labels)
    
    save_artifacts(vectorizer, model)

    print("\n✅  Training complete. Artifacts ready in models/\n")
