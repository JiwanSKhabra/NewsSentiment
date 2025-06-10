import os
import re
import requests
import pandas as pd
from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine

from dotenv import load_dotenv
load_dotenv()

# === API Keys and MySQL Credentials ===
API_KEY = os.environ['API_KEY']  # NYT
GNEWS_API_KEY = os.environ['GNEWS_API_KEY']
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
encoded_password = quote_plus(MYSQL_PASSWORD)

# === MySQL Config ===
MYSQL_USER = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "news_data"
NYT_TABLE = "nyt_articles"
GNEWS_TABLE = "gnews_articles"

# === Query & Time Range ===
query = "climate OR politics OR economy OR technology OR environment"
begin_date = "20240601"
end_date = "20240630"

# === Utility ===
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

analyzer = SentimentIntensityAnalyzer()
nyt_rows = []
gnews_rows = []

# === NYT Scraping ===
nyt_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
for page in range(5):
    params = {
        "q": query,
        "api-key": API_KEY,
        "sort": "newest",
        "begin_date": begin_date,
        "end_date": end_date,
        "page": page,
        "fq": 'source:("The New York Times")'
    }
    try:
        response = requests.get(nyt_url, params=params)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
        if not docs:
            print(f"⚠️ No more NYT articles on page {page}.")
            break
        for doc in docs:
            snippet = doc.get("snippet", "")
            cleaned = clean_text(snippet)
            score = analyzer.polarity_scores(cleaned)["compound"]
            sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
            nyt_rows.append({
                "Title": doc["headline"]["main"],
                "Published_Date": doc["pub_date"][:10],
                "URL": doc["web_url"],
                "Snippet": snippet,
                "Cleaned_Snippet": cleaned,
                "Source": doc.get("source", "NYT"),
                "News_Desk": doc.get("news_desk", ""),
                "Sentiment": sentiment,
                "Sentiment_Score": score,
                "Bias": "Center-Left"
            })
        print(f"✅ NYT Page {page} scraped.")
    except Exception as e:
        print(f"❌ NYT error on page {page}: {e}")
        break

# === GNews Scraping ===
gnews_url = "https://gnews.io/api/v4/search"
try:
    gnews_params = {
        "q": query,
        "lang": "en",
        "max": 50,
        "apikey": GNEWS_API_KEY
    }
    gnews_response = requests.get(gnews_url, params=gnews_params)
    gnews_response.raise_for_status()
    gnews_articles = gnews_response.json().get("articles", [])

    for article in gnews_articles:
        desc = article.get("description", "")
        cleaned = clean_text(desc)
        score = analyzer.polarity_scores(cleaned)["compound"]
        sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        gnews_rows.append({
            "Title": article.get("title", ""),
            "Published_Date": article.get("publishedAt", "")[:10],
            "URL": article.get("url", ""),
            "Snippet": desc,
            "Cleaned_Snippet": cleaned,
            "Source": article.get("source", {}).get("name", "GNews"),
            "News_Desk": "GNews",
            "Sentiment": sentiment,
            "Sentiment_Score": score,
            "Bias": "Mixed"
        })
    print(f"✅ GNews scraped: {len(gnews_rows)} articles.")
except Exception as e:
    print(f"❌ GNews error: {e}")

# === Save to MySQL ===
try:
    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

    if nyt_rows:
        pd.DataFrame(nyt_rows).to_sql(NYT_TABLE, con=engine, if_exists="replace", index=False)
        print(f"✅ NYT data saved to `{NYT_TABLE}`")

    if gnews_rows:
        pd.DataFrame(gnews_rows).to_sql(GNEWS_TABLE, con=engine, if_exists="replace", index=False)
        print(f"✅ GNews data saved to `{GNEWS_TABLE}`")

    if not nyt_rows and not gnews_rows:
        print("⚠️ No articles collected from either source.")
except Exception as e:
    print(f"❌ MySQL save error: {e}")
