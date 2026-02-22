import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(
    df: pd.DataFrame,
    text_col: str = "clause_text",
    label_col: str = "risk_label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    features = df[text_col]
    targets = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        targets, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=targets
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state, 
        n_jobs=-1,
    )
    
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    unique_target_labels = df[label_col].unique()

    return vectorizer, clf, y_test, y_pred, unique_target_labels
