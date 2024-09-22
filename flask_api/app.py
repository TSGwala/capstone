

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS on all routes

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    # Predict sentiment
    prediction = model.predict(text_vectorized)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return jsonify({'sentiment': sentiment})

# Load the dataset (assuming 'sentiment' column contains 1 for Positive and 0 for Negative)
df = pd.read_csv('twitter_sentiment_data.csv')  # Replace with your actual dataset

@app.route('/filter-sentiments', methods=['GET'])
def filter_sentiments():
    # Get the sentiment type from the request (1 for positive, 0 for negative)
    sentiment_type = request.args.get('sentiment_type')  # e.g., '1' or '0'

    # Filter the dataset by sentiment type (1 for positive, 0 for negative)
    filtered_df = df.copy()

    if sentiment_type is not None:
        filtered_df = filtered_df[filtered_df['sentiment'] == int(sentiment_type)]

    # Count the filtered sentiment values
    sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()

    # Prepare the response
    response = {
        'Positive': sentiment_counts.get(1, 0),
        'Negative': sentiment_counts.get(0, 0)
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)


# import tweepy
# from flask import Flask, jsonify
# import os
# import traceback
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Twitter API credentials (replace these with the names of your environment variables)
# CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
# CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
# ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
# ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# # Print the values to debug if they are being pulled correctly
# print(f"CONSUMER_KEY: {CONSUMER_KEY}, CONSUMER_SECRET: {CONSUMER_SECRET}")

# # Ensure that the credentials are not None
# if not CONSUMER_KEY or not CONSUMER_SECRET or not ACCESS_TOKEN or not ACCESS_TOKEN_SECRET:
#     raise ValueError("Twitter API credentials are not set correctly. Check your environment variables.")

# # Authenticate to Twitter API
# auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# api = tweepy.API(auth)

# @app.route('/', methods=['GET'])
# def home():
#     return "<h1>Welcome to the Twitter Trending Topics API</h1><p>Use /trending-topics to get the latest trending topics.</p>"

# @app.route('/trending-topics', methods=['GET'])
# def get_trending_topics():
#     try:
#         # Example of handling different API calls or API v2 endpoint if available
#         # Since we don't have a direct replacement, handle as a placeholder
#         return jsonify({'error': 'This endpoint is not available with the current API access level.'}), 403
#     except Exception as e:
#         # Log the full traceback and error message
#         error_message = traceback.format_exc()
#         print(f"Error: {error_message}")  # Log the error message for debugging
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

