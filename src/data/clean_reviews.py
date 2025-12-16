import os
import pandas as pd
import re
import nltk
import spacy

from nltk.corpus import stopwords

# ---------------------------------------
# INITIAL SETUP
# ---------------------------------------

RAW_COMBINED = "datasets/clean/haile_reviews_combined.csv"
OUT_FILE = "datasets/clean/haile_reviews_cleaned.csv"

# Download stopwords if missing
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load spaCy English model (small, fast, perfect for lemmatization)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ---------------------------------------
# TEXT CLEANING FUNCTIONS
# ---------------------------------------

def clean_text(text: str) -> str:
    """Full cleaning pipeline with normalization, stopwords, lemmatization."""
    if pd.isna(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # 3. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 4. Remove punctuation & special chars
    text = re.sub(r"[^\w\s]", " ", text)

    # 5. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Remove stopwords & lemmatize
    doc = nlp(text)
    clean_tokens = []

    for token in doc:
        if token.text not in stop_words and len(token.text) > 2:
            clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)


# ---------------------------------------
# MAIN CLEANING PIPELINE
# ---------------------------------------

def clean_reviews():
    print("Loading dataset...")
    df = pd.read_csv(RAW_COMBINED)

    print("Original rows:", len(df))

    # Normalize missing values
    df["review_comment"] = df["review_comment"].fillna("")
    df["review_title"] = df["review_title"].fillna("")

    # Standardize rating (rating_raw → rating_0_5)
    if "rating_raw" in df.columns:
        df["rating_raw"] = pd.to_numeric(df["rating_raw"], errors="coerce")
        df["rating_0_5"] = (df["rating_raw"] / 5.0) * 5.0
    else:
        df["rating_0_5"] = None

    # Remove duplicates
    df.drop_duplicates(subset=["review_comment", "hotel_name"], inplace=True)

    # Apply text cleaning function
    print("Cleaning text... (lemmatization, stopwords, normalization)")
    df["clean_comment"] = df["review_comment"].apply(clean_text)

    # Optional: combine title + comment for stronger NLP performance
    df["clean_full_text"] = (
        df["review_title"].fillna("").astype(str) + " " +
        df["clean_comment"]
    ).str.strip()

    # Save cleaned file
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("\n=======================================")
    print("CLEANING COMPLETED")
    print("Saved cleaned data:", OUT_FILE)
    print("Total rows after cleaning:", len(df))
    print("=======================================")


if __name__ == "__main__":
    clean_reviews()

