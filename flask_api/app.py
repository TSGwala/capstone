
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

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
    sentiment_type = request.args.get('sentiment_type')

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

# Initialize poll data
poll_options = ["I don't believe in Climate change", "I'm undecided", "Climate change is a serious threat"]
poll_votes = [0, 0, 0]  # Vote counts for each option

@app.route('/vote', methods=['POST'])
def vote():
    selected_option = request.form.get('option')
    if selected_option is None or selected_option not in ['1', '2', '3']:
        return jsonify({'status': 'error', 'message': 'Invalid option selected.'}), 400
    
    # Update the vote count
    poll_votes[int(selected_option) - 1] += 1
    return jsonify({'status': 'success'})

@app.route('/results', methods=['GET'])
def results():
    return jsonify({
        'options': poll_options,
        'votes': poll_votes
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable if available
    app.run(host='0.0.0.0', port=port, debug=True)