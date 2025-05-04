import joblib
import pandas as pd
import json

#load model responses from intent_model.pkl
model = joblib.load('intent_model.pkl')

# Load chatbot responses from JSON file
with open('C:/Users/IIJet/Chatbot/responses.json', 'r') as f:
    responses = json.load(f)

#chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    # Predict the intent using the loaded model
    intent = model.predict([user_input])[0]
    # Get the response based on the predicted intent
    response = responses[intent]
    print("Chatbot:", response)

    