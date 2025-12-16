import os
import pandas as pd

RAW_DIR = "datasets/raw/haile_reviews"
OUT_FILE = "datasets/clean/haile_reviews_combined.csv"

def combine_reviews():
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Raw folder not found: {RAW_DIR}")

    all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not all_files:
        raise ValueError("No CSV files found in the raw reviews directory.")

    print("Found files:")
    for f in all_files:
        print(" -", f)

    dfs = []
    for fname in all_files:
        path = os.path.join(RAW_DIR, fname)
        try:
            df = pd.read_csv(path)
            df["source_file"] = fname  # Keep track of origin
            dfs.append(df)
        except Exception as e:
            print(f"[Error] Failed reading {fname}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Create output directory
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Save combined file
    combined.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("\n====================================")
    print(f"Combined dataset saved to:\n{OUT_FILE}")
    print("Total rows:", len(combined))
    print("====================================")

if __name__ == "__main__":
    combine_reviews()

