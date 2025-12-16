import pandas as pd

IN_PATH = "datasets/clean/haile_reviews_with_topics.csv"
OUT_PATH = "datasets/clean/haile_reviews_with_topics.csv"   # overwrite

df = pd.read_csv(IN_PATH)

if "rating_1_5" in df.columns:
    df = df.rename(columns={"rating_1_5": "rating_0_5"})

df.to_csv(OUT_PATH, index=False)
print("Rating column fixed.")

