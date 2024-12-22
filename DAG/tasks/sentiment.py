import logging
import requests

PREDICT_URL = "http://34.118.74.116:8080/predict"

def get_sentiment_score(text):
    """
    Sends the article text to the prediction endpoint and retrieves the sentiment score.
    """
    try:
        if text is None:
            text = ''
        payload = {"instances": [text]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(PREDICT_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logging.error(f"Prediction API returned status code {response.status_code}: {response.text}")
            return None
        
        predictions = response.json().get('predictions', [])
        if not predictions:
            logging.warning("No predictions returned from the API.")
            return None
        
        return predictions[0]
    except Exception as e:
        logging.error(f"Error calling prediction API: {e}")
        return None