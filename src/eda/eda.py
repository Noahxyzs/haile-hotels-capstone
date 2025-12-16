# src/eda/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

CLEAN_PATH = "datasets/clean/haile_reviews_cleaned.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def load():
    df = pd.read_csv(CLEAN_PATH)
    return df

def basic_stats(df):
    print("Rows:", len(df))
    print("Hotels:", df['hotel_name'].nunique())
    print(df[['hotel_name','source']].value_counts().head())

def rating_histogram(df):
    if "rating_0_5" in df.columns:
        fig = px.histogram(df, x="rating_0_5", nbins=10, title="Rating distribution (0-5)")
        fig.write_html(os.path.join(OUT_DIR, "rating_histogram.html"))
        fig.write_image(os.path.join(OUT_DIR, "rating_histogram.png"))
    else:
        print("rating_0_5 not found")

def sentiment_counts(df):
    if "sentiment" not in df.columns:
        print("No sentiment column found.")
        return
    fig = px.histogram(df, x="sentiment", color="sentiment", title="Sentiment counts")
    fig.write_html(os.path.join(OUT_DIR, "sentiment_counts.html"))
    fig.write_image(os.path.join(OUT_DIR, "sentiment_counts.png"))

def hotel_rating_box(df):
    if "rating_0_5" in df.columns:
        fig = px.box(df, x="hotel_name", y="rating_0_5", title="Rating by hotel")
        fig.write_html(os.path.join(OUT_DIR, "rating_by_hotel.html"))
        fig.write_image(os.path.join(OUT_DIR, "rating_by_hotel.png"))

def wordcloud_by_sentiment(df):
    for s in df['sentiment'].dropna().unique():
        text = " ".join(df[df['sentiment']==s]['clean_full_text'].fillna("").tolist())
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400).generate(text)
        out = os.path.join(OUT_DIR, f"wordcloud_{s}.png")
        wc.to_file(out)
        print("Saved", out)

def top_topics(df):
    if "topics" in df.columns:
        # topics may be pipe-separated
        toks = df['topics'].dropna().astype(str).str.split("|").explode()
        top = toks.value_counts().head(20)
        plt.figure(figsize=(8,6))
        sns.barplot(x=top.values, y=top.index)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "topic_freq.png"))
        plt.close()
        print("Saved topic_freq.png")

def run_all():
    df = load()
    basic_stats(df)
    rating_histogram(df)
    sentiment_counts(df)
    hotel_rating_box(df)
    wordcloud_by_sentiment(df)
    top_topics(df)
    print("EDA completed. Figures saved to", OUT_DIR)

if __name__ == "__main__":
    run_all()

