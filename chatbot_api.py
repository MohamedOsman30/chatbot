from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# === Load Resources ===
MODEL_PATH = "chatbot_model.h5"
INTENTS_PATH = "intents.json"

# Load model
try:
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# Load intents
try:
    with open(INTENTS_PATH, encoding="utf-8") as f:
        intents = json.load(f)
    print("[INFO] Intents loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load intents: {e}")
    intents = {"intents": []}

# Generate words and classes from intents
words = sorted(set(
    w.lower()
    for intent in intents.get('intents', [])
    for pattern in intent.get('patterns', [])
    for w in pattern.translate(str.maketrans('', '', string.punctuation)).split()
))
classes = sorted(set(intent.get('tag') for intent in intents.get('intents', [])))

print(f"[INFO] Vocabulary size: {len(words)}")
print(f"[INFO] Number of classes: {len(classes)}")

# === NLP Functions ===
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
    if input_data.shape[1] != model.input_shape[1]:
        print(f"[ERROR] Input shape mismatch: expected {model.input_shape[1]}, got {input_data.shape[1]}")
        return []
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

# === API Endpoints ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "not ready",
        "vocab_size": len(words),
        "classes_count": len(classes)
    }), 200 if model else 503

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("[DEBUG] Received data:", data)

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "No message provided, please send a valid message."}), 400

        predicted_intents = predict_class(message)
        print("[DEBUG] Predicted intents:", predicted_intents)

        response = get_response(predicted_intents, intents)
        print("[DEBUG] Response:", response)

        return jsonify({"response": response}), 200

    except Exception as e:
        print(f"[ERROR] Exception in /chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

# === Run App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
