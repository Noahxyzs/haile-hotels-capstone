import requests
from bs4 import BeautifulSoup
import time, csv, os
import pandas as pd
from typing import List, Dict, Any

OUT_DIR = "datasets/raw/tripadvisor"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

# TripAdvisor URLs for Haile Hotels (review pages)
HOTEL_URLS = {
    "haile_addis_ababa_grand":
        "https://www.tripadvisor.com/Hotel_Review-g293791-d17362800-Reviews-Haile_Grand_Addis_Ababa-Addis_Ababa.html",

    "haile_adama_grand":
        "https://www.tripadvisor.com/Hotel_Review-g2193239-d15572966-Reviews-Haile_Resort_Adama-Adama_Oromiya_Region.html",

    "haile_arba_minch_grand":
        "https://www.tripadvisor.com/Hotel_Review-g1064678-d12987631-Reviews-Haile_Resort_Arba_Minch-Arba_Minch_Southern_Nations_Nationalities_and_People_s_Region.html",

    "haile_hawassa":
        "https://www.tripadvisor.com/Hotel_Review-g676757-d4579642-Reviews-Haile_Resort_Hawassa-Hawassa_Southern_Nations_Nationalities_and_People_s_Region.html",

    "haile_ziway_batu":
        "https://www.tripadvisor.com/Hotel_Review-g677357-d20384368-Reviews-Haile_Resort_Ziway-Batu_Oromiya_Region.html",

    "haile_gondar":
        "https://www.tripadvisor.com/Hotel_Review-g480191-d19479510-Reviews-Haile_Resort_Gondar-Gonder_Amhara_Region.html",
}


def extract_reviews_from_page(soup) -> List[Dict[str, Any]]:
    """Extract review blocks from one TripAdvisor page."""
    reviews = []

    review_blocks = soup.select("div.review-container") or soup.select("div.YibKl")

    for block in review_blocks:
        try:
            # Rating
            rating_tag = block.select_one("span.ui_bubble_rating")
            rating_raw = None
            if rating_tag:
                cls = rating_tag.get("class", [])
                for c in cls:
                    if "bubble_" in c:
                        rating_raw = int(c.replace("bubble_", "")) / 10  # e.g., bubble_40 -> 4.0 stars

            # Title
            title_tag = block.select_one("span.noQuotes")
            title = title_tag.text.strip() if title_tag else None

            # Comment
            comment_tag = block.select_one("p.partial_entry")
            comment = comment_tag.text.strip() if comment_tag else None

            reviews.append({
                "source": "tripadvisor",
                "rating": rating_raw,
                "review_title": title,
                "review_comment": comment
            })

        except Exception as e:
            print(f"[Skip] Failed to parse review: {e}")
            continue

    return reviews


def get_next_page(soup):
    """Find next page URL."""
    next_btn = soup.select_one("a.next")
    if next_btn and next_btn.get("href"):
        return "https://www.tripadvisor.com" + next_btn["href"]
    return None


def scrape_tripadvisor(hotel_key: str, url: str, max_pages: int = 5) -> str:
    print("\n========================================")
    print(f"Scraping TripAdvisor for: {hotel_key}")
    print("========================================\n")

    all_reviews = []
    current_url = url

    for page in range(max_pages):
        print(f"[Page {page+1}] {current_url}")

        try:
            r = requests.get(current_url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"[Error] Failed to load page: {e}")
            break

        reviews = extract_reviews_from_page(soup)
        all_reviews.extend(reviews)

        # Find next page
        next_url = get_next_page(soup)
        if not next_url:
            print("[Info] No more pages.")
            break

        current_url = next_url
        time.sleep(2)

    # Save to CSV
    out_file = f"{OUT_DIR}/{hotel_key}_tripadvisor.csv"

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df["hotel_name"] = hotel_key
        df.to_csv(out_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"[Success] Saved → {out_file} ({len(df)} reviews)\n")
    else:
        print(f"[Stop] No reviews collected for {hotel_key}")

    return out_file


if __name__ == "__main__":
    for key, url in HOTEL_URLS.items():
        scrape_tripadvisor(hotel_key=key, url=url, max_pages=5)

