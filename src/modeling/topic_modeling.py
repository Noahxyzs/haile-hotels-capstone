# src/modeling/topic_modeling.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

CLEAN_PATH = "datasets/clean/haile_reviews_cleaned.csv"
OUT_FILE = "datasets/clean/haile_reviews_with_topics.csv"
MODEL_DIR = "models/topics"
os.makedirs(MODEL_DIR, exist_ok=True)

def lda_topics(df, n_topics=6):
    texts = df['clean_full_text'].fillna("").tolist()
    vec = CountVectorizer(max_features=5000, stop_words='english')
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)
    # get dominant topic for each doc
    doc_topic = lda.transform(X)
    dominant = doc_topic.argmax(axis=1)
    df['lda_topic'] = dominant
    # save topic keywords
    words = vec.get_feature_names_out()
    topic_keywords = {}
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-15:][::-1]
        topic_keywords[i] = [words[t] for t in top_idx]
    joblib.dump({'lda':lda, 'vectorizer':vec, 'keywords':topic_keywords}, os.path.join(MODEL_DIR, "lda_topics.joblib"))
    return df, topic_keywords

def try_bertopic(df):
    try:
        from bertopic import BERTopic
        print("Running BERTopic (this may take longer)...")
        topics_model = BERTopic(language="english")
        docs = df['clean_full_text'].fillna("").tolist()
        topics, probs = topics_model.fit_transform(docs)
        df['bertopic_topic'] = topics
        topics_model.save(os.path.join(MODEL_DIR, "bertopic_model"))
        return df, topics_model
    except Exception as e:
        print("BERTopic not available or failed:", e)
        return df, None

def run():
    df = pd.read_csv(CLEAN_PATH)
    df_lda, keywords = lda_topics(df, n_topics=8)
    print("LDA topic keywords:")
    for k,v in keywords.items():
        print(k, v[:10])
    # optional: try bertopic
    # df_bert, bert_model = try_bertopic(df_lda)
    df_lda.to_csv(OUT_FILE, index=False)
    print("Saved with topics to", OUT_FILE)

if __name__ == "__main__":
    run()

