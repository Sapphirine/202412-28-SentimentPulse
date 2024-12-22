import json
import logging
from google.cloud import storage
from datetime import datetime
from yahoo_fin import news
import pytz
from util import get_article_text

BUCKET_NAME = "us-central1-big-data-38be6a47-bucket"
FOLDER_NAME = "data"

def convert_to_est(date_str):
    utc_time = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    est_time = utc_time.astimezone(pytz.timezone('America/New_York'))
    return est_time.strftime('%Y-%m-%d %H:%M:%S %Z')

def get_yahoo_finance_news(ticker):
    news_items = news.get_yf_rss(ticker)
    articles_info = []

    for item in news_items[:5]:
        link = item.get('link', 'No link available')
        title = item.get('title', 'No title available')
        published = item.get('published', 'No date available')

        articles_info.append({
            "ticker": ticker,
            "link": link,
            "title": title,
            "date": convert_to_est(published),
            "text": None
        })

    return articles_info

def get_ticker_news(ticker):
    articles_info = get_yahoo_finance_news(ticker)
    for article in articles_info:
        link = article["link"]
        article_text = get_article_text(link)
        if article_text:
            article["text"] = article_text
        else:
            logging.warning(f"Article content not found for: {link}")
    return articles_info

def fetch_and_store_news(tickers):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    all_articles = []

    for ticker in tickers:
        articles_info = get_ticker_news(ticker)
        all_articles.extend(articles_info)

        # Create a timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{FOLDER_NAME}/{ticker}_news_{timestamp}.json"

        # Upload results to Google Cloud Storage
        blob = bucket.blob(output_file)
        blob.upload_from_string(json.dumps(articles_info), content_type="application/json")
        logging.info(f"News articles for {ticker} saved to {BUCKET_NAME}/{output_file}")
    print(all_articles)
    return all_articles