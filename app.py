import joblib
import pandas as pd
import streamlit as st
import json

#load model responses from intent_model.pkl
model = joblib.load('intent_model.pkl')

# Load chatbot responses from JSON file
with open('data/responses.json', 'r') as f:
    responses = json.load(f)

st.title("Intent Chatbot")
user_input = st.text_input("You:", "")
if user_input:
    intent = model.predict([user_input])[0]
    st.write(f"Chatbot: {responses.get(intent, 'Sorry, I don’t understand.')}")
