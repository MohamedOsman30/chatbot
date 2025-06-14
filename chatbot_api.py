from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model and intents from local files
MODEL_PATH = "chatbot_model.h5"
INTENTS_PATH = "intents.json"

model = load_model(MODEL_PATH)
with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

# Preprocess vocabulary and classes
words = sorted(set(
    word.lower()
    for intent in intents['intents']
    for pattern in intent['patterns']
    for word in pattern.translate(str.maketrans('', '', string.punctuation)).split()
))
classes = sorted(set(intent['tag'] for intent in intents['intents']))

# Helper functions
def clean_up_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return [word.lower() for word in sentence.split()]

def is_known_sentence(sentence, vocab):
    return all(word in vocab for word in clean_up_sentence(sentence))

def bow(sentence, vocab):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if word in sentence_words else 0 for word in vocab])

def predict_class(sentence):
    if not is_known_sentence(sentence, words):
        return []
    input_data = np.array([bow(sentence, words)])
    predictions = model.predict(input_data, verbose=0)[0]
    ERROR_THRESHOLD = 0.4
    results = [
        {"intent": classes[i], "probability": str(prob)}
        for i, prob in enumerate(predictions)
        if prob > ERROR_THRESHOLD
    ]
    return sorted(results, key=lambda x: x["probability"], reverse=True)

def get_response(intents_list, intents_data):
    if not intents_list:
        return "I'm sorry, I don't understand that."
    intent_tag = intents_list[0]['intent']
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Chat API Route
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "No message provided, please send a valid message."}), 400
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
