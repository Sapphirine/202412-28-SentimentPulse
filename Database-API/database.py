from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Initialize Base
Base = declarative_base()

# Database connection setup
def connect_to_db():
    instance_connection_name = "big-data-441922:us-central1:ticker-info"
    db_user = "postgres"
    db_pass = "password"
    db_name = "postgres"
    ip_type = IPTypes.PRIVATE if os.getenv("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    return pool

# Global database variables
engine = connect_to_db()
Base = declarative_base()

# Use this to manage database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define Article model
class Article(Base):
    __tablename__ = 'sentiment_scores'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    ticker = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    title = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    article_text = sqlalchemy.Column(sqlalchemy.Text, nullable=False)
    link = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    score = sqlalchemy.Column(sqlalchemy.Float, nullable=False)

# Define Tweet model
class Tweet(Base):
    __tablename__ = 'tweets'
    tweet_id = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
    text = sqlalchemy.Column(sqlalchemy.Text, nullable=False)
    username = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    favorite_count = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    retweet_count = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
    score = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    ticker = sqlalchemy.Column(sqlalchemy.String, nullable=False)  # Assuming each tweet is associated with a ticker

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
