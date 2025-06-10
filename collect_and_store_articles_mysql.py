import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
import re
import os

api_key = os.environ['API_KEY']
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']

# === MYSQL DB CONFIG ===
MYSQL_USER = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "news_data"

TABLE_NAME = "nyt_articles"

# === NYT API CONFIG ===
API_KEY = api_key
query = "climate change"
begin_date = "20240601"
end_date = "20240630"
url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# === TEXT CLEANING FUNCTION ===
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === INIT ===
analyzer = SentimentIntensityAnalyzer()
all_rows = []

# === SCRAPE ARTICLES ===
for page in range(5):
    params = {
        "q": query,
        "api-key": API_KEY,
        "sort": "newest",
        "begin_date": begin_date,
        "end_date": end_date,
        "page": page
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            print(f"⚠️ No more articles found at page {page}.")
            break

        for doc in docs:
            snippet = doc.get("snippet", "")
            cleaned_snippet = clean_text(snippet)
            vs = analyzer.polarity_scores(cleaned_snippet)
            compound_score = vs["compound"]
            sentiment = (
                "Positive" if compound_score > 0.05 else
                "Negative" if compound_score < -0.05 else
                "Neutral"
            )

            all_rows.append({
                "Title": doc["headline"]["main"],
                "Published_Date": doc["pub_date"][:10],
                "URL": doc["web_url"],
                "Snippet": snippet,
                "Cleaned_Snippet": cleaned_snippet,
                "Source": doc.get("source", ""),
                "News_Desk": doc.get("news_desk", ""),
                "Sentiment": sentiment,
                "Sentiment_Score": compound_score,
                "Bias": "Center-Left"
            })

        print(f"✅ Page {page} processed with {len(docs)} articles.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error on page {page}: {e}")
        break
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        break

# === SAVE TO MYSQL ===
if all_rows:
    df = pd.DataFrame(all_rows)
    print(f"📄 Total articles collected: {len(df)}")

    try:
        connection_url = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        engine = create_engine(connection_url)

        df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        print(f"✅ Articles saved to MySQL table `{TABLE_NAME}` in DB `{MYSQL_DB}`.")
    except Exception as e:
        print(f"❌ Failed to save to MySQL: {e}")
else:
    print("⚠️ No articles to save.")
