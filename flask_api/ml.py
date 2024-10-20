import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

"""# upload tha dataset"""

data=pd.read_csv('twitter_sentiment_data.csv')
data.head()

"""# Explore Data"""

data.shape

data.info()

data1 = data.dropna()
data1.head()

data1.isnull().sum()

"""# Preprocessing"""

# Remove duplicate rows
data1.drop_duplicates()
data1.head()

def clean_text(text):
  """
  This function removes specific characters from the text for sentiment analysis.

  Args:
      text: The text to be cleaned.

  Returns:
      The cleaned text.
  """
  # Replace with the characters you want to remove (e.g., punctuation)
  text = re.sub(r'[!"#$%&()*+,/:;<=>?@\\\]^_`{|}~]', '', text)  # Remove various punctuation
  # You can add more replacements here (e.g., emojis, symbols)
  return text

# Create a copy of the DataFrame to avoid modifying the original
data1_cleaned = data1.copy()

# Apply cleaning to the "message" column
data1_cleaned["cleaned_message"] = data1_cleaned["message"].apply(clean_text)

# Print the cleaned data (all columns preserved)
print(data1_cleaned)

# Apply cleaning directly to the DataFrame (modifies in-place)
data1["cleaned_message"] = data1["message"].apply(clean_text)

# Print the cleaned data (original DataFrame modified)
print(data1)

def clean_text(text):
  """
  This function removes specific characters from the text for sentiment analysis.

  Args:
      text: The text to be cleaned.

  Returns:
      The cleaned text.
  """
  # Replace with the characters you want to remove (e.g., punctuation)
  text = re.sub(r'[!"#$%&()*+,/:;<=>?@\\\]^_`{|}~]', '', text)  # Remove various punctuation
  # You can add more replacements here (e.g., emojis, symbols)
  return text

# Create a copy of the DataFrame (optional, recommended to avoid modifying original)
data1_cleaned = data1.copy()

# Apply cleaning to the "message" column
data1_cleaned["cleaned_message"] = data1_cleaned["message"].apply(clean_text)

# Use your sentiment analysis function (replace with your specific logic)
def get_sentiment(text):
  # Replace with your sentiment analysis logic (e.g., using TextBlob, VADER, etc.)
  # This is a placeholder, replace with your actual sentiment analysis code
  sentiment = "Positive"  # Replace with actual sentiment score or category
  return sentiment

# Apply sentiment analysis and add a new "sentiment" column
data1_cleaned["sentiment"] = data1_cleaned["cleaned_message"].apply(get_sentiment)

# Print the cleaned and analyzed data (all columns preserved)
print(data1_cleaned)

# Drop columns with names starting with "Unnamed:" (replace if different)
data1_cleaned = data1.loc[:, ~data1.columns.str.startswith("message:")]

# Print the data with uncleaned columns removed (based on column names)
print(data1_cleaned)

data1_cleaned.drop('message', axis=1, inplace=True)

print(data1_cleaned)

# Filter the DataFrame without altering the original
data1_cleaned = data1_cleaned[data1_cleaned['sentiment'] !=-1]

# Display the cleaned DataFrame
print(data1_cleaned)

# Filter the DataFrame without altering the original
data1_cleaned = data1_cleaned[data1_cleaned['sentiment'] !=2]

# Display the cleaned DataFrame
print(data1_cleaned)

# Remove numbers from the 'message' column
data1_cleaned['cleaned_message'] = data1_cleaned['cleaned_message'].str.replace(r'\d+', '', regex=True)

# Display the modified DataFrame
print(data1_cleaned)

# Convert 'message' column to lowercase
data1_cleaned['cleaned_message'] = data1_cleaned['cleaned_message'].str.lower()

# Display the modified DataFrame
print(data1_cleaned)

# Count the occurrences of each sentiment
sentiment_counts = data1_cleaned['sentiment'].value_counts()

# Create the bar graph
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])

plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

data1_cleaned.head()

import pandas as pd
from sklearn.model_selection import train_test_split

"""# Split Dataset into Training and Test dataset"""

X = data1_cleaned['cleaned_message']
y = data1_cleaned['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

"""# Model Training"""

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train_tfidf, y_train)

"""# Predict the Model"""

dump(model, 'logistic_regression_model.joblib')
dump(vectorizer, 'vectorizer.joblib')
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

"""# Model Evaluation"""

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

"""# Confusion Matrix"""

from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""# Classification Report"""

from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)

"""# Cross-Validation"""

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean():.2f}')



