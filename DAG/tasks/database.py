import logging
from sqlalchemy import text
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine
import os

def connect_to_db():
    instance_connection_name = "big-data-441922:us-central1:ticker-info"
    db_user = "postgres"
    db_pass = "password"
    db_name = "postgres"
    ip_type = IPTypes.PRIVATE if os.getenv("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    return pool

def insert_article(pool, article):
    """
    Inserts the article data along with the sentiment score into the PostgreSQL database.
    """
    try:
        with pool.connect() as conn:
            insert_query = """
                INSERT INTO sentiment_scores (ticker, title, article_text, link, timestamp, score)
                VALUES (:ticker, :title, :article_text, :link, :timestamp, :score)
                ON CONFLICT (link) DO NOTHING
            """
            conn.execute(
                text(insert_query),
                {
                    "ticker": article["ticker"],
                    "title": article["title"],
                    "article_text": article["text"],
                    "link": article["link"],
                    "timestamp": article["date"],
                    "score": article["score"],
                },
            )
            logging.info(f"Inserted article '{article['title']}' into the database.")
    except Exception as e:
        logging.error(f"Error inserting into database: {e}")