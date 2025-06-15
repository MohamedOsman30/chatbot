from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
from tensorflow.keras.models import load_model

# === Load model and intents ===
model = load_model('chatbot_model.h5')

with open('intents.json', encoding="utf8") as json_file:
    intents_data = json.load(json_file)

# === Generate words and classes dynamically ===
words = sorted(set([
    w.lower()
    for intent in intents_data['intents']
    for pattern in intent['patterns']
    for w in pattern.translate(str.maketrans('', '', string.punctuation)).split()
]))
classes = sorted(set([intent['tag'] for intent in intents_data['intents']]))

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
        return []
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.4
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
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
            return jsonify({"response": "No message provided."}), 400
        intents_list = predict_class(message)
        response = get_response(intents_list, intents_data)
        return jsonify({"response": response}), 200
    except Exception as e:
        print(f"[ERROR] /chat error: {e}")
        return jsonify({"response": "Something went wrong."}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
