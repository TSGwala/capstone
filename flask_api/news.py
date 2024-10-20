import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

# Step 1: Set up your NewsAPI key and fetch climate news
api_key = '6c6d35e39bce43e9a8d7150e0a7394f6'

def fetch_climate_news(api_key, query="climate change", page_size=10):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()['articles']
        return articles
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

# Fetch articles
articles = fetch_climate_news(api_key, query="climate change", page_size=10)

# Step 2: Build your sentiment analysis model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Step 2a: Load your sentiment analysis dataset
data = pd.read_csv('twitter_sentiment_data.csv')  # Update with the path to your dataset
X_train = data['message']  # Replace with your text column name
y_train = data['sentiment']  # Replace with your sentiment column name

# Step 2b: Fit the model with your actual dataset
model.fit(X_train, y_train)

# Step 3: Perform sentiment analysis on the news article titles
def analyze_sentiment(news_articles, model):
    # Filter out articles with None or empty titles
    valid_articles = [article['title'] for article in news_articles if article['title']]
    
    if not valid_articles:
        print("No valid article titles to analyze.")
        return []
    
    # Perform prediction directly on the valid titles list
    sentiments = model.predict(valid_articles)
    return sentiments

# Analyze the sentiments of fetched articles
sentiments = analyze_sentiment(articles, model)

# Step 4: Display and save the results
results = []
if len(sentiments) > 0:  # Check if sentiments is not empty
    for i, (article, sentiment) in enumerate(zip(articles, sentiments)):
        print(f"Article {i+1}: {article['title']}, Sentiment: {sentiment}")
        results.append({'title': article['title'], 'sentiment': sentiment})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('sentiment_results.csv', index=False)
    print("Results saved to sentiment_results.csv")

else:
    print("No sentiments to display.")
