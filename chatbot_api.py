from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
import os
import requests  # <-- used instead of gdown
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# === Step 1: Download from direct Google Drive download link ===
MODEL_URL = "https://drive.usercontent.google.com/download?id=1BVN7l4JcW72O1lO-k621wq6y0nF5J_82&export=download&authuser=0&confirm=t&uuid=d40e7f90-8243-4220-9500-7ecf3f3ed433&at=AN8xHoq_54mPXpZWhvigBncwtEfI:1749887597066"
INTENTS_URL = "https://drive.usercontent.google.com/download?id=1i9jdXegd5wy2wBN-O3_NDtq0lbPzFgEB&export=download&authuser=0&confirm=t&uuid=fa8ecbc7-6774-4914-a911-136831e79a72&at=AN8xHoodgxRNTAeYFIpG3bfCGW-U:1749887680988"

MODEL_PATH = "chatbot_model.h5"
INTENTS_PATH = "intents.json"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {dest}: HTTP {response.status_code}")

# Download both files
download_file(MODEL_URL, MODEL_PATH)
download_file(INTENTS_URL, INTENTS_PATH)

# === Step 2: Load model and intents ===
model = load_model(MODEL_PATH)

with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

words = sorted(set(
    word.lower()
    for intent in intents['intents']
    for pattern in intent['patterns']
    for word in pattern.translate(str.maketrans('', '', string.punctuation)).split()
))
classes = sorted(set(intent['tag'] for intent in intents['intents']))

# === Helper functions ===
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

# === Chat API Route ===
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "No message provided, please send a valid message."}), 400
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    return jsonify({"response": response}), 200

