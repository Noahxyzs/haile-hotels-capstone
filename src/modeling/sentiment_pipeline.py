# src/modeling/sentiment_pipeline.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CLEAN_PATH = "datasets/clean/haile_reviews_cleaned.csv"
OUT_DIR = "models/sentiment"
os.makedirs(OUT_DIR, exist_ok=True)

def label_from_rating(r):
    try:
        if np.isnan(r):
            return "unknown"
        v = float(r)
        if v >= 4.0:
            return "positive"
        if v == 3.0:
            return "neutral"
        if v <= 2.0:
            return "negative"
    except:
        return "unknown"

def prepare_data(df):
    # If sentiment exists, use it; else infer from rating_0_5
    if "sentiment" not in df.columns or df['sentiment'].isnull().all():
        df['sentiment'] = df['rating_0_5'].apply(label_from_rating)
    # Use only known labels
    df = df[df['sentiment'].isin(['positive','neutral','negative'])].copy()
    df['text'] = df['clean_full_text'].fillna("")
    # optionally balance classes or show counts
    print("Class distribution:\n", df['sentiment'].value_counts())
    return df

def train_and_save(df):
    X = df['text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    Xtr = tfidf.fit_transform(X_train)
    Xt = tfidf.transform(X_test)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "nb": MultinomialNB(),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    for name, m in models.items():
        print("Training", name)
        m.fit(Xtr, y_train)
        preds = m.predict(Xt)
        print(f"=== {name} classification report ===")
        print(classification_report(y_test, preds))
        results[name] = m
        joblib.dump(m, os.path.join(OUT_DIR, f"{name}.joblib"))

    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
    print("Saved vectorizer and models to", OUT_DIR)

def run():
    df = pd.read_csv(CLEAN_PATH)
    df = prepare_data(df)
    train_and_save(df)

if __name__ == "__main__":
    run()

