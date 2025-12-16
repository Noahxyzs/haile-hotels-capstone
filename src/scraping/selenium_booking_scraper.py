import time
import csv
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

OUT_DIR = "datasets/raw/booking"
os.makedirs(OUT_DIR, exist_ok=True)

HOTEL_URLS = {
    "haile_addis_ababa_grand": "https://www.booking.com/reviews/et/hotel/haile-grand-addis-ababa.en-gb.html",
    "haile_adama_grand": "https://www.booking.com/reviews/et/hotel/haile-resort-adama.en-gb.html",
    "haile_arba_minch_grand": "https://www.booking.com/reviews/et/hotel/haile-resort-arbaminch.en-gb.html",
    "haile_hawassa": "https://www.booking.com/reviews/et/hotel/haile-resort-hawassa.en-gb.html",
    "haile_ziway_batu": "https://www.booking.com/reviews/et/hotel/haile-resort-ziway-batu.en-gb.html",
    "haile_gondar": "https://www.booking.com/reviews/et/hotel/haile-resort-gondar.en-gb.html",
}

def normalize_rating(rating_text):
    """Convert Booking 1–10 rating to 0–5 scale."""
    try:
        val = float(rating_text)
        return round((val / 10) * 5, 1)
    except:
        return None

def init_driver():
    opts = Options()
    opts.add_argument("--headless")      # Remove this if you want to see the browser
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--log-level=3")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )

def scrape_booking_selenium(hotel_key, url, max_pages=5):
    print("\n========================================")
    print(f"Scraping Booking.com for: {hotel_key}")
    print("========================================\n")

    driver = init_driver()
    driver.get(url)
    time.sleep(3)

    reviews = []

    for page in range(max_pages):
        print(f"[Page {page+1}] Extracting...")

        time.sleep(3)

        review_blocks = driver.find_elements(By.CSS_SELECTOR, "div.review_item")
        if not review_blocks:
            review_blocks = driver.find_elements(By.CSS_SELECTOR, ".c-review-block")

        if not review_blocks:
            print("  [Warning] No review blocks found. Stopping.")
            break

        for block in review_blocks:
            try:
                # Rating
                rating_el = block.find_element(By.CSS_SELECTOR, ".bui-review-score__badge")
                rating_10 = rating_el.text.strip()
                rating = normalize_rating(rating_10)
            except:
                rating = None

            # Title
            try:
                title_el = block.find_element(By.CSS_SELECTOR, ".c-review-block__title")
                title = title_el.text.strip()
            except:
                title = None

            # Full comment
            try:
                comment_el = block.find_element(By.CSS_SELECTOR, ".c-review__body")
                comment = comment_el.text.strip()
            except:
                comment = None

            reviews.append({
                "hotel_name": hotel_key,
                "source": "booking.com",
                "rating": rating,
                "review_title": title,
                "review_comment": comment,
            })

        # Try to click "Next page"
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.pagenext')
            driver.execute_script("arguments[0].click();", next_btn)
        except:
            print("  [Info] No more pages.")
            break

        time.sleep(2)

    driver.quit()

    # Save CSV
    out_file = f"{OUT_DIR}/{hotel_key}_booking.csv"
    if reviews:
        df = pd.DataFrame(reviews)
        df.to_csv(out_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"[Success] Saved {out_file} ({len(reviews)} rows)\n")
    else:
        print("[Stop] No reviews collected.\n")

    return out_file


if __name__ == "__main__":
    print("--- Starting Selenium Booking Scraper ---")
    for key, url in HOTEL_URLS.items():
        scrape_booking_selenium(key, url, max_pages=5)
    print("--- Finished ---")
