import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from flask import Flask, request, render_template, send_file
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import io
import os

app = Flask(__name__)

# Fetch data from the API
def fetch_data(api_url, start_time, end_time):
    params = {
        "start": start_time,
        "end": end_time
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} {response.text}")
        return []

# Plot the data for all tickers
def plot_metrics_all_tickers(data, metric_key, title):
    plt.figure(figsize=(10, 6))
    tickers = ["TSLA", "GOOGL", "META", "DJT", "DIS"]

    for ticker in tickers:
        ticker_data = [d for d in data if d['ticker'] == ticker]
        if ticker_data:
            hours = [datetime.fromisoformat(d['hour']) for d in ticker_data]
            metric_values = [d[metric_key] for d in ticker_data]

            plt.plot(hours, metric_values, marker='o', label=ticker)

    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel(metric_key.replace('_', ' ').title())
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

@app.route('/')
def index():
    metrics = ["max_score", "avg_score", "std_dev", "min_score"]
    return render_template('index.html', metrics=metrics)

@app.route('/news_plot')
def news_plot():
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    metric = request.args.get('metric')

    if not start_time or not end_time or not metric:
        return "Missing parameters", 400

    api_url = "https://database-api-453150507971.us-central1.run.app/api/scores"
    data = fetch_data(api_url, start_time, end_time)

    print("News Data Fetched:", data)  # Debugging API response

    if not data:
        return "No data available for the selected parameters", 404

    title = f"News {metric.replace('_', ' ').title()} Across All Tickers"
    img = plot_metrics_all_tickers(data, metric, title)
    return send_file(img, mimetype='image/png')

@app.route('/tweet_plot')
def tweet_plot():
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    metric = request.args.get('metric')

    if not start_time or not end_time or not metric:
        return "Missing parameters", 400

    api_url = "https://database-api-453150507971.us-central1.run.app/api/tweet_scores"
    data = fetch_data(api_url, start_time, end_time)

    print("Tweet Data Fetched:", data)  # Debugging API response

    if not data:
        return "No data available for the selected parameters", 404

    title = f"Tweets {metric.replace('_', ' ').title()} Across All Tickers"
    img = plot_metrics_all_tickers(data, metric, title)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
