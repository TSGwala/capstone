import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Load dataset
data = pd.read_csv('train.csv')

# Preprocess data
def clean_text(text):
    text = re.sub(r'[!"#$%&()*+,/:;<=>?@\\\]^_`{|}~]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

data1 = data.dropna(subset=['message'])  # Ensure 'message' column is not null
data1["cleaned_message"] = data1["message"].apply(clean_text)

# Example sentiment analysis logic
def get_sentiment(text):
    if "happy" in text:
        return 'Positive'
    elif "sad" in text:
        return 'Negative'
    else:
        return 'Neutral'

data1["sentiment"] = data1["cleaned_message"].apply(get_sentiment)

# Check class distribution
print(data1['sentiment'].value_counts())

# Proceed with model training if there are multiple classes
if len(data1['sentiment'].unique()) > 1:
    X = data1['cleaned_message']
    y = data1['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear')  # Increase max_iter if needed
    model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer
    dump(model, 'logistic_regression_model.joblib')
    dump(vectorizer, 'vectorizer.joblib')

    # Predict the model
    y_pred = model.predict(X_test_tfidf)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

else:
    print("Insufficient classes for training the model.")
