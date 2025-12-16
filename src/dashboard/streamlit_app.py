import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# -------------------------
# SETTINGS
# -------------------------
st.set_page_config(
    page_title="Haile Hotels & Resorts — Review Analytics",
    layout="wide",
)

DATA_PATH = "datasets/clean/haile_reviews_with_topics.csv"
MODEL_DIR = "models/sentiment"

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "rating_1_5" in df.columns:
        df = df.rename(columns={"rating_1_5": "rating_0_5"})
    return df

@st.cache_resource
def load_models():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))
    return tfidf, model

df = load_data()
tfidf, model = load_models()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Filters")

hotel_list = ["All Hotels"] + sorted(df["hotel_name"].unique().tolist())
selected_hotel = st.sidebar.selectbox("Select Hotel", hotel_list)

sentiment_list = ["All Sentiments", "positive", "neutral", "negative"]
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_list)

topic_list = ["All Topics"] + sorted(df["lda_topic"].dropna().unique().tolist())
selected_topic = st.sidebar.selectbox("LDA Topic", topic_list)

filtered_df = df.copy()

if selected_hotel != "All Hotels":
    filtered_df = filtered_df[filtered_df["hotel_name"] == selected_hotel]

if selected_sentiment != "All Sentiments":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

if selected_topic != "All Topics":
    filtered_df = filtered_df[filtered_df["lda_topic"] == selected_topic]

# -------------------------
# HEADER
# -------------------------
st.title("Haile Hotels & Resorts — Review Analytics Dashboard")
st.markdown(
    "A professional dashboard for **sentiment analysis, ratings, and topic insights** "
    "based on real customer reviews."
)

# -------------------------
# KPI CARDS
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Avg Rating (0–5)", round(filtered_df["rating_0_5"].dropna().mean(), 2))
col3.metric("Positive Reviews", (filtered_df["sentiment"] == "positive").sum())
col4.metric("Negative Reviews", (filtered_df["sentiment"] == "negative").sum())

# -------------------------
# RATING DISTRIBUTION
# -------------------------
st.markdown("Rating Distribution")
fig = px.histogram(
    filtered_df,
    x="rating_0_5",
    nbins=10,
    color="hotel_name",
    title="Rating Distribution",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SENTIMENT DISTRIBUTION
# -------------------------
st.markdown("Sentiment Distribution")
fig2 = px.pie(
    filtered_df,
    names="sentiment",
    title="Sentiment Breakdown",
    color="sentiment",
    color_discrete_map={
        "positive": "green",
        "neutral": "gray",
        "negative": "red"
    }
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# TOPIC FREQUENCIES
# -------------------------
st.markdown("LDA Topic Frequencies")
topic_counts = filtered_df["lda_topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]

fig3 = px.bar(
    topic_counts,
    x="Topic",
    y="Count",
    text="Count",
    title="Most Common Topics",
    template="plotly_white"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# TF-IDF TERM IMPORTANCE (ACADEMIC REPLACEMENT)
# -------------------------
st.markdown("Important Terms by Sentiment (TF-IDF)")

tfidf_sentiment = st.selectbox(
    "Select sentiment for TF-IDF analysis:",
    filtered_df["sentiment"].unique()
)

subset = filtered_df[filtered_df["sentiment"] == tfidf_sentiment]

if subset.empty:
    st.warning("No reviews available for this sentiment.")
else:
    texts = subset["clean_full_text"].dropna().astype(str).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    tfidf_df = (
        pd.DataFrame({"Term": terms, "Score": scores})
        .sort_values("Score", ascending=False)
        .head(15)
    )

    fig_tfidf, ax = plt.subplots(figsize=(8, 5))
    ax.barh(tfidf_df["Term"][::-1], tfidf_df["Score"][::-1])
    ax.set_xlabel("Average TF-IDF Score")
    ax.set_title(f"Top TF-IDF Terms — {tfidf_sentiment.capitalize()} Reviews")
    plt.tight_layout()
    st.pyplot(fig_tfidf)

# -------------------------
# REVIEW TABLE
# -------------------------
st.markdown("Review Samples")

display_cols = [
    "hotel_name", "source", "rating_raw",
    "rating_0_5", "sentiment", "clean_full_text", "lda_topic"
]
display_cols = [c for c in display_cols if c in filtered_df.columns]

st.dataframe(filtered_df[display_cols].head(50), use_container_width=True)

# -------------------------
# SENTIMENT PREDICTOR
# -------------------------
st.markdown("---")
st.subheader("Sentiment Predictor")

user_input = st.text_area("Enter a review comment:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip():
        vect = tfidf.transform([user_input])
        prediction = model.predict(vect)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
    else:
        st.error("Please enter a review text.")

