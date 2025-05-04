import json
import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# load intent data from JSON file
with open('C:/Users/IIJet/Chatbot/intents.json') as f:
    data = json.load(f)

# Extract utterances and labels
utterances = []
labels = []
for item in data:
    intent = item['intent']
    for utterance in item['utterances']:
        utterances.append(utterance)
        labels.append(intent)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(utterances, labels, test_size=0.1, random_state=42, stratify=labels)

# Create pipeline
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()  # No stop_words
model = make_pipeline(vectorizer, LogisticRegression(C=10, max_iter=1000))

# Fit model
model.fit(X_train, y_train)

# Evaluate
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, utterances, labels, cv=5)
print("Accuracy:", scores.mean())
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'intent_model.pkl')
