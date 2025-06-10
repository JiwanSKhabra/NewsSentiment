import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from urllib.parse import quote_plus

load_dotenv()

# === MySQL Connection Info ===
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
encoded_password = quote_plus(MYSQL_PASSWORD)
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "news_data"
TABLE_NAME = "nyt_articles"

# === Load Data ===
@st.cache_data(show_spinner=True)
def load_articles():
    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    df["Published_Date"] = pd.to_datetime(df["Published_Date"])
    return df

# === Topic Clustering with Readable Labels ===
@st.cache_data(show_spinner=True)
def add_clusters(df, num_clusters=5):
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(df["Cleaned_Snippet"].fillna(""))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster_ID"] = kmeans.fit_predict(X)

    # Extract top keywords per cluster centroid
    terms = tfidf.get_feature_names_out()
    top_keywords = []
    for i in range(num_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[::-1][:5]
        keywords = [terms[ind] for ind in top_indices]
        top_keywords.append(", ".join(keywords))

    # Create human-readable labels for each cluster
    cluster_labels = {i: f"Topic {i}: {top_keywords[i]}" for i in range(num_clusters)}
    df["Topic_Cluster"] = df["Cluster_ID"].map(cluster_labels)

    return df, kmeans

# === Streamlit App ===
st.set_page_config(page_title="News Aggregator with Clustering", layout="wide")
st.title("🧠 News Aggregator with Topic Clustering")

df = load_articles()
df, kmeans_model = add_clusters(df, num_clusters=5)

# === Sidebar Filters ===
st.sidebar.header("📂 Filters")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=df["Sentiment"].unique(),
    default=list(df["Sentiment"].unique())
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=[df["Published_Date"].min(), df["Published_Date"].max()]
)

keyword = st.sidebar.text_input("Search Keyword (in snippet)")

topic_filter = st.sidebar.multiselect(
    "Filter by Topic Cluster",
    options=sorted(df["Topic_Cluster"].unique()),
    default=sorted(df["Topic_Cluster"].unique())
)

sort_by = st.sidebar.selectbox("Sort By", options=["Published_Date", "Sentiment", "Bias"])
sort_order = st.sidebar.radio("Sort Order", options=["Descending", "Ascending"])

# === Filtering Logic ===
filtered_df = df[
    (df["Sentiment"].isin(sentiment_filter)) &
    (df["Published_Date"] >= pd.to_datetime(date_range[0])) &
    (df["Published_Date"] <= pd.to_datetime(date_range[1])) &
    (df["Topic_Cluster"].isin(topic_filter))
]

if keyword:
    filtered_df = filtered_df[filtered_df["Cleaned_Snippet"].str.contains(keyword, case=False, na=False)]

ascending = sort_order == "Ascending"
filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

# === Display Table ===
st.markdown(f"### 🔍 Showing {len(filtered_df)} Articles")
st.dataframe(
    filtered_df[["Published_Date", "Title", "Sentiment", "Bias", "Topic_Cluster", "URL"]],
    use_container_width=True
)

# === Expandable Article View ===
for _, row in filtered_df.iterrows():
    with st.expander(f"📰 {row['Title']} ({row['Published_Date'].date()}) — {row['Topic_Cluster']}"):
        st.markdown(f"**Sentiment**: {row['Sentiment']}  |  **Bias**: {row['Bias']}  |  **Cluster**: {row['Topic_Cluster']}")
        st.markdown(f"**Source**: {row['Source']} | **News Desk**: {row['News_Desk']}")
        st.markdown(f"**Snippet**: {row['Snippet']}")
        st.markdown(f"[🔗 Read Full Article]({row['URL']})", unsafe_allow_html=True)
