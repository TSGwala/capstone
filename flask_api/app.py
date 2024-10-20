


# import os
# import pandas as pd
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import joblib

# app = Flask(__name__, static_folder='public', static_url_path='')
# CORS(app)

# # Load the model and vectorizer
# model = joblib.load('logistic_regression_model.joblib')
# vectorizer = joblib.load('vectorizer.joblib')

# # Route to serve the frontend (index.html)
# @app.route('/')
# def serve_frontend():
#     return send_from_directory(app.static_folder, 'index.html')

# # Serve any other static assets (CSS, JS, images)
# @app.route('/<path:path>')
# def serve_static_files(path):
#     return send_from_directory(app.static_folder, path)

# # Sentiment Analysis API
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400
    
#     text = data['text']
#     # Vectorize the input text
#     text_vectorized = vectorizer.transform([text])
#     # Predict sentiment
#     prediction = model.predict(text_vectorized)
#     sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
#     return jsonify({'sentiment': sentiment})

# # Load the dataset
# df = pd.read_csv('twitter_sentiment_data.csv')

# @app.route('/filter-sentiments', methods=['GET'])
# def filter_sentiments():
#     sentiment_type = request.args.get('sentiment_type')
#     filtered_df = df.copy()
#     if sentiment_type is not None:
#         filtered_df = filtered_df[filtered_df['sentiment'] == int(sentiment_type)]
#     sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()
#     response = {
#         'Positive': sentiment_counts.get(1, 0),
#         'Negative': sentiment_counts.get(0, 0)
#     }
#     return jsonify(response)

# # Polling system
# poll_options = ["I don't believe in Climate change", "I'm undecided", "Climate change is a serious threat"]
# poll_votes = [0, 0, 0]

# @app.route('/vote', methods=['POST'])
# def vote():
#     selected_option = request.form.get('option')
#     if selected_option is None or selected_option not in ['1', '2', '3']:
#         return jsonify({'status': 'error', 'message': 'Invalid option selected.'}), 400
#     poll_votes[int(selected_option) - 1] += 1
#     return jsonify({'status': 'success'})

# @app.route('/results', methods=['GET'])
# def results():
#     return jsonify({
#         'options': poll_options,
#         'votes': poll_votes
#     })

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)


import os
import re
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Predefined climate-related keywords
climate_keywords = [
    'climate', 'global warming', 'carbon', 'emissions', 'flood', 
    'drought', 'greenhouse', 'temperature', 'hurricane', 'weather',
    'climate change', 'sustainability', 'renewable energy', 'ecosystem', 
    'deforestation', 'biodiversity', 'pollution', 'sea level rise', 
    'fossil fuels', 'climate action', 'adaptation', 'mitigation', 
    'carbon footprint', 'environment', 'natural disaster', 'extreme weather', 
    'carbon neutrality', 'climate policy', 'green technology', 
    'climate resilience', 'climate justice'
]

# Function to check if the text is climate-related
def is_climate_related(text):
    pattern = r'\b(' + '|'.join(climate_keywords) + r')\b'
    if re.search(pattern, text.lower()):
        return True
    return False

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Serve any other static assets (CSS, JS, images)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# Sentiment Analysis API with climate-related check
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Check if the text is climate-related
    if not is_climate_related(text):
        return jsonify({'message': 'This keyword or phrase is not related to climate change.'})
    
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_vectorized)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return jsonify({'sentiment': sentiment})

# Load the dataset
df = pd.read_csv('twitter_sentiment_data.csv')

# Sentiment filtering endpoint
@app.route('/filter-sentiments', methods=['GET'])
def filter_sentiments():
    sentiment_type = request.args.get('sentiment_type')
    filtered_df = df.copy()
    if sentiment_type is not None:
        filtered_df = filtered_df[filtered_df['sentiment'] == int(sentiment_type)]
    sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()
    response = {
        'Positive': sentiment_counts.get(1, 0),
        'Negative': sentiment_counts.get(0, 0)
    }
    return jsonify(response)

# Polling system
poll_options = ["I don't believe in Climate change", "I'm undecided", "Climate change is a serious threat"]
poll_votes = [0, 0, 0]

@app.route('/vote', methods=['POST'])
def vote():
    selected_option = request.form.get('option')
    if selected_option is None or selected_option not in ['1', '2', '3']:
        return jsonify({'status': 'error', 'message': 'Invalid option selected.'}), 400
    poll_votes[int(selected_option) - 1] += 1
    return jsonify({'status': 'success'})

@app.route('/results', methods=['GET'])
def results():
    return jsonify({
        'options': poll_options,
        'votes': poll_votes
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
