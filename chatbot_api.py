from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
import pickle
import os
import logging
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load model and data ===
try:
    model = load_model('chatbot_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

try:
    with open('intents.json', encoding="utf8") as json_file:
        intents_data = json.load(json_file)
    logger.info("Intents file loaded successfully")
except Exception as e:
    logger.error(f"Failed to load intents.json: {str(e)}")
    raise

# Load or generate vocabulary
words = []
try:
    if os.path.exists('words.pkl'):
        with open('words.pkl', 'rb') as f:
            words = pickle.load(f)
        logger.info(f"Vocabulary loaded from words.pkl, size: {len(words)}")
    else:
        logger.warning("words.pkl not found, generating vocabulary from intents.json")
        words = sorted(set([
            w.lower()
            for intent in intents_data['intents']
            for pattern in intent['patterns']
            for w in pattern.translate(str.maketrans('', '', string.punctuation)).split()
        ]))
        logger.info(f"Generated vocabulary from intents.json, size: {len(words)}")
except Exception as e:
    logger.error(f"Failed to load or generate vocabulary: {str(e)}")
    raise

classes = sorted(set([intent['tag'] for intent in intents_data['intents']]))
logger.info(f"Number of classes: {len(classes)}")

# === NLP Functions ===
def clean_up_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence_words = sentence.split()
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def is_known_sentence(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    for word in sentence_words:
        if word not in words:
            return False
    return True

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if not is_known_sentence(sentence, words):
        logger.info(f"Unknown sentence: {sentence}")
        return []
    p = bow(sentence, words)
    try:
        res = model.predict(np.array([p]), verbose=0)[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return []
    ERROR_THRESHOLD = 0.4
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
        logger.info("No intents above threshold")
        return []
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_data):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_data['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I don't understand that."

# === Flask App ===
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json.get("message", "").strip()
        if not message:
            logger.warning("No message provided in request")
            return jsonify({"response": "No message provided."}), 400
        logger.info(f"Received message: {message}")
        logger.info(f"Vocabulary size: {len(words)}")
        logger.info(f"Model input shape: {model.input_shape}")
        intents_list = predict_class(message)
        response = get_response(intents_list, intents_data)
        return jsonify({"response": response}), 200
    except Exception as e:
        logger.error(f"[ERROR] /chat error: {str(e)}")
        return jsonify({"response": "Something went wrong."}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
