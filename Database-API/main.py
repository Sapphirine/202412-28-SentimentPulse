from flask import Flask, jsonify, request
from database import SessionLocal, Article, Tweet
from flask_cors import CORS
import sqlalchemy
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

@app.route('/api/scores', methods=['GET'])
def get_scores():
    db = SessionLocal()
    try:
        # Parse query parameters for start and end timestamps
        start_time = request.args.get('start', default=None, type=str)
        end_time = request.args.get('end', default=None, type=str)

        if not start_time or not end_time:
            return jsonify({"error": "Please provide both start and end timestamps."}), 400

        # Convert strings to datetime
        try:
            start_time = datetime.fromisoformat(start_time)
            end_time = datetime.fromisoformat(end_time)
        except ValueError:
            return jsonify({"error": "Invalid date format. Use ISO 8601 format (e.g., '2024-12-19T08:00:00')."}), 400

        # Calculate scores for each hour in the range
        results = []
        current_time = start_time
        while current_time <= end_time:
            # Calculate the 5-hour range
            range_start = current_time - timedelta(hours=5)
            range_end = current_time

            # Query scores within the 5-hour range
            scores = db.query(
                Article.ticker,
                sqlalchemy.func.avg(Article.score).label('avg_score'),
                sqlalchemy.func.max(Article.score).label('max_score'),
                sqlalchemy.func.min(Article.score).label('min_score'),
                sqlalchemy.func.stddev(Article.score).label('std_dev')
            ).filter(
                Article.timestamp >= range_start,
                Article.timestamp < range_end
            ).group_by(Article.ticker).all()

            # Append results
            for r in scores:
                results.append({
                    "ticker": r[0],
                    "hour": current_time.isoformat(),
                    "avg_score": r[1],
                    "max_score": r[2],
                    "min_score": r[3],
                    "std_dev": r[4],
                })

            # Move to the next hour
            current_time += timedelta(hours=1)

        return jsonify(results)

    finally:
        db.close()

@app.route('/api/tweet_scores', methods=['GET'])
def get_tweet_scores():
    """
    Endpoint to get sentiment scores for tweets within a specific time range.
    The `date` column is stored as a proper `timestamp`.
    Accepts input in ISO 8601 format (e.g., '2024-12-19T08:00:00').
    """
    db = SessionLocal()
    try:
        # Parse query parameters for start and end timestamps
        start_time = request.args.get('start', default=None, type=str)
        end_time = request.args.get('end', default=None, type=str)

        if not start_time or not end_time:
            return jsonify({"error": "Please provide both start and end timestamps."}), 400

        # Convert ISO 8601 strings to datetime objects
        try:
            start_time = datetime.fromisoformat(start_time)
            end_time = datetime.fromisoformat(end_time)
        except ValueError:
            return jsonify({"error": "Invalid date format. Use ISO 8601 format (e.g., '2024-12-19T08:00:00')."}), 400

        # Calculate scores for each hour in the range
        results = []
        current_time = start_time
        while current_time <= end_time:
            # Calculate the 5-hour range
            range_start = current_time - timedelta(hours=5)
            range_end = current_time

            # Query scores within the 5-hour range
            scores = db.query(
                Tweet.ticker,
                sqlalchemy.func.avg(Tweet.score).label('avg_score'),
                sqlalchemy.func.max(Tweet.score).label('max_score'),
                sqlalchemy.func.min(Tweet.score).label('min_score'),
                sqlalchemy.func.stddev(Tweet.score).label('std_dev')
            ).filter(
                Tweet.timestamp >= range_start,
                Tweet.timestamp < range_end
            ).group_by(Tweet.ticker).all()

            # Append results
            for r in scores:
                results.append({
                    "ticker": r[0],
                    "hour": current_time.isoformat(),  # Keep ISO 8601 format in the response
                    "avg_score": r[1],
                    "max_score": r[2],
                    "min_score": r[3],
                    "std_dev": r[4],
                })

            # Move to the next hour
            current_time += timedelta(hours=1)

        return jsonify(results)

    finally:
        db.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))
