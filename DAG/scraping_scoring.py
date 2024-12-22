from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from tasks.scraping import fetch_and_store_news
from tasks.sentiment import get_sentiment_score
from tasks.database import connect_to_db, insert_article
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'news_scraper_and_sentiment_analysis',
    default_args=default_args,
    description='Scrape news articles, analyze sentiment, and store results in PostgreSQL',
    schedule_interval='@hourly',
    start_date=datetime(2024, 11, 20),
    catchup=False,
) as dag:

    def scrape_and_upload_news(**kwargs):
        tickers = Variable.get("ticker_list", default_var="AAPL,GOOGL,MSFT").split(",")
        articles = fetch_and_store_news(tickers)
        kwargs['ti'].xcom_push(key='scraped_articles', value=articles)

    def process_sentiment_scores(**kwargs):
        # Retrieve articles from XCom
        ti = kwargs['ti']
        articles = ti.xcom_pull(key='scraped_articles', task_ids='scrape_and_upload_news')

        if not articles:
            logging.error("No articles found in XCom.")
            return

        pool = connect_to_db()
        for article in articles:
            try:
                # Get sentiment score
                sentiment_score = get_sentiment_score(article["text"])
                if sentiment_score is None:
                    logging.warning(f"Sentiment score not obtained for article: {article['title']}. Skipping.")
                    continue

                # Add sentiment score to the article
                article["score"] = sentiment_score

                # Insert into the database
                insert_article(pool, article)
            except Exception as e:
                logging.error(f"Error processing article '{article['title']}': {e}")

    scrape_and_upload = PythonOperator(
        task_id='scrape_and_upload_news',
        python_callable=scrape_and_upload_news,
        provide_context=True,
    )

    process_sentiment = PythonOperator(
        task_id='process_sentiment_scores',
        python_callable=process_sentiment_scores,
        provide_context=True,
    )

    scrape_and_upload >> process_sentiment
