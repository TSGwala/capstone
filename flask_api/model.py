import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Step 1: Load the dataset
data = pd.read_csv('twitter_sentiment_data.csv')

# Step 2: Preprocess the data
def clean_text(text):
    text = re.sub(r'[!"#$%&()*+,/:;<=>?@\\\]^_`{|}~]', '', text)  # Remove punctuation
    return text

data['cleaned_message'] = data['message'].apply(clean_text)
data['cleaned_message'] = data['cleaned_message'].str.replace(r'\d+', '', regex=True)
data['cleaned_message'] = data['cleaned_message'].str.lower()

# Step 3: Filter out non-climate-related queries using predefined keywords
climate_keywords = ['climate', 'global warming', 'carbon', 'emissions', 'flood', 'drought', 'greenhouse', 'temperature', 'hurricane', 'weather']

def is_climate_related(text):
    # Create a regex pattern that ensures we match whole words (not substrings)
    pattern = r'\b(' + '|'.join(climate_keywords) + r')\b'
    
    # Use re.search to check if any climate-related keyword is found in the text
    if re.search(pattern, text.lower()):
        return True
    return False

# Apply the filter and keep only climate-related data
data['is_climate_related'] = data['cleaned_message'].apply(is_climate_related)
data_climate = data[data['is_climate_related'] == True]

# Step 4: Split the filtered data for model training
X = data_climate['cleaned_message']
y = data_climate['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorize the data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer for future use
dump(model, 'logistic_regression_model.joblib')
dump(vectorizer, 'vectorizer.joblib')

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean():.2f}')

# Step 9: Function to handle user input and apply strict filtering
def predict_sentiment(user_input):
    # Check if the user input is climate-related
    if not is_climate_related(user_input):
        return "This is not a climate-related keyword."
    
    # Clean the user input
    cleaned_input = clean_text(user_input)
    
    # Transform the input with the vectorizer
    input_tfidf = vectorizer.transform([cleaned_input])
    
    # Predict sentiment
    prediction = model.predict(input_tfidf)
    
    if prediction == 1:
        return "Positive sentiment detected."
    else:
        return "Negative sentiment detected."

# Example usage
user_input = input("Enter a keyword or phrase to check sentiment: ")
result = predict_sentiment(user_input)
print(result)

