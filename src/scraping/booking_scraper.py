# booking_scraper.py
import requests
import time
import os
import pandas as pd
from bs4 import BeautifulSoup

# ------------------------------
# CONFIG
# ------------------------------
OUT_DIR = "datasets/raw/booking"
os.makedirs(OUT_DIR, exist_ok=True)

HOTEL_URLS = {
    "haile_addis_ababa_grand": "https://www.booking.com/reviews/et/hotel/haile-grand-addis-ababa.html",
    "haile_adama_grand": "https://www.booking.com/reviews/et/hotel/haile-resort-adama.html",
    "haile_arba_minch_grand": "https://www.booking.com/reviews/et/hotel/haile-resort-arbaminch.html",
    "haile_hawassa": "https://www.booking.com/reviews/et/hotel/haile-resort-hawassa.html",
    "haile_ziway_batu": "https://www.booking.com/reviews/et/hotel/haile-resort-ziway-batu.html",
    "haile_gondar": "https://www.booking.com/reviews/et/hotel/haile-resort-gondar.html",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
}


# ------------------------------
# HELPERS
# ------------------------------
def normalize_rating(rating_raw):
    """Convert Booking.com 0–10 rating to 0–5 scale."""
    try:
        r10 = float(rating_raw)
        r5 = (r10 / 10) * 5
        return round(r5, 1)
    except:
        return None


# ------------------------------
# SCRAPER FUNCTION
# ------------------------------
def scrape_booking(hotel_key, base_url, pages=3, delay=2):
    print("")
    print("========================================")
    print("Scraping Booking.com for:", hotel_key)
    print("========================================")
    print("")

    reviews = []

    for page in range(pages):
        offset = page * 10
        url = f"{base_url}?offset={offset}"

        print("[Page", page+1, "] Fetching:", url)

        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print("[Error] Failed to load page:", e)
            break

        soup = BeautifulSoup(r.text, "html.parser")

        # MAIN SELECTORS
        blocks = soup.select(".review_list_item")

        # FALLBACK SELECTOR
        if not blocks:
            blocks = soup.select("div[data-review-id]")

        # If no reviews appear → stop
        if not blocks:
            print("[Warning] No review blocks found on this page. Stopping.")
            break

        for b in blocks:
            try:
                # Rating
                rating_tag = (
                    b.select_one(".review-score-badge") or
                    b.select_one(".bui-review-score__badge")
                )
                rating_raw = rating_tag.get_text(strip=True) if rating_tag else None
                rating = normalize_rating(rating_raw)

                # Title
                title_tag = (
                    b.select_one(".review_item_header_content") or
                    b.select_one(".c-review-block__title")
                )
                title = title_tag.get_text(strip=True) if title_tag else ""

                # Comment
                comment_tag = (
                    b.select_one(".review_item_review_content") or
                    b.select_one(".c-review__body")
                )
                comment = comment_tag.get_text(strip=True) if comment_tag else ""

                reviews.append({
                    "hotel_name": hotel_key,
                    "source": "booking",
                    "rating": rating,
                    "review_title": title,
                    "review_comment": comment,
                })
            except Exception:
                continue

        time.sleep(delay)

    # Save CSV
    out_file = f"{OUT_DIR}/{hotel_key}.csv"

    if reviews:
        df = pd.DataFrame(reviews)
        df.to_csv(out_file, index=False, encoding="utf-8")
        print("[Saved]", len(df), "reviews to", out_file)
    else:
        print("[Stop] No reviews collected for", hotel_key)

    return out_file


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    print("--- Starting Booking.com Scraper ---")
    for key, url in HOTEL_URLS.items():
        scrape_booking(key, url, pages=5)
    print("--- Finished ---")
