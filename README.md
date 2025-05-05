# Intent-Classification Chatbot

This project is an **Intent-Classification Chatbot** that identifies user intents (e.g., greeting, help, calculator) from text input and responds accordingly. Built with logistic regression achieving 80% accuracy.

## Features
- **Intents**: Recognizes 24 intents (e.g., greeting, goodbye, calculator, weather) with 9–10 example utterances each.
- **Model**: Logistic regression classifier with 80% prediction accuracy and 0.7 cross-validation accuracy.
- **Interface**: Offers a terminal-based chatbot and a web interface using Streamlit.
- **Data**: Intents and responses stored in JSON files for easy modification.

## Skills Demonstrated
- **NLP**: Text preprocessing, Count vectorization, intent classification.
- **Machine Learning**: Model training, evaluation, and deployment.
- **Deployment**: Streamlit-based web interface.
- **Data Handling**: JSON-based data management.

## Setup Instructions
Follow these steps to run the project locally:

## Setup
1. Clone: `git clone https://github.com/yourusername/intent-chatbot.git`
2. Install: `pip install -r requirements.txt`
3. Run terminal: `python chatbot.py`
4. Run web: `streamlit run app.py`

## Sample Dialog
- User: "Hi" → Chatbot: "Hello! How can I assist you today?"
- User: "Can you help with calculation?" → Chatbot: "Let’s calculate!"

## Skills
- NLP (CountVectorizer), Machine Learning, web deployment.
