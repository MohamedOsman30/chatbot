from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
from tensorflow.keras.models import load_model
import os
import pickle
app = Flask(__name__)
CORS(app)

# === Paths to model and intents ===
MODEL_PATH = "chatbot_model.h5"
INTENTS_PATH = "intents.json"
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# === Load the model ===
model = None
model_load_error = None
try:
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    model_load_error = str(e)
    print(f"[ERROR] Failed to load model: {model_load_error}")

# === Load intents file ===
intents = {}
try:
    with open(INTENTS_PATH, encoding="utf-8") as f:
        intents = json.load(f)
    print("[INFO] Intents loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load intents: {e}")
    intents = {"intents": []}

# === Build vocabulary and classes ===
words = sorted(set(
    word.lower()
    for intent in intents.get("intents", [])
    for pattern in intent.get("patterns", [])
    for word in pattern.translate(str.maketrans('', '', string.punctuation)).split()
))
classes = sorted(set(intent.get("tag") for intent in intents.get("intents", [])))

# === Preprocessing functions ===
def clean_up_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return [word.lower() for word in sentence.split()]

def is_known_sentence(sentence, vocab):
    return any(word in vocab for word in clean_up_sentence(sentence))

def bow(sentence, vocab):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if word in sentence_words else 0 for word in vocab])

def predict_class(sentence):
    if not model or not is_known_sentence(sentence, words):
        return []
    input_data = np.array([bow(sentence, words)])
    predictions = model.predict(input_data, verbose=0)[0]
    ERROR_THRESHOLD = 0.4
    results = [
        {"intent": classes[i], "probability": str(prob)}
        for i, prob in enumerate(predictions)
        if prob > ERROR_THRESHOLD
    ]
    return sorted(results, key=lambda x: float(x["probability"]), reverse=True)

def get_response(intents_list, intents_data):
    if not intents_list:
        return "I'm sorry, I don't understand that."
    intent_tag = intents_list[0]['intent']
    for intent in intents_data.get("intents", []):
        if intent.get("tag") == intent_tag:
            return random.choice(intent.get("responses", []))
    return "I'm not sure how to respond to that."

# === Health check route ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "not ready",
        "error": model_load_error
    }), 200 if model else 503

# === Chat route ===
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("[DEBUG] Received data:", data)

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "No message provided."}), 400

        predicted_intents = predict_class(message)
        print("[DEBUG] Predicted intents:", predicted_intents)

        response = get_response(predicted_intents, intents)
        print("[DEBUG] Response:", response)

        return jsonify({"response": response}), 200

    except Exception as e:
        print(f"[ERROR] Exception in /chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

# === Start app for local debug (not used in production) ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
