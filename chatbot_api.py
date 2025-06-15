from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import string
from tensorflow.keras.models import load_model

# Load model
model = load_model('chatbot_model.h5')

# Load the intents JSON file
with open('intents.json', encoding="utf8") as json_file:
    data = json.load(json_file)

words = sorted(set([w.lower() for intent in data['intents'] for pattern in intent['patterns'] for w in pattern.split()]))
classes = sorted(set([intent['tag'] for intent in data['intents']]))

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
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.4
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
        return []
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "I'm sorry, I don't understand that."

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Chat Route
@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get("message")
    
    # Validate if message is provided
    if not message:
        return jsonify({"response": "No message provided, please send a valid message."}), 400
    
    # Predict the class and get the response
    ints = predict_class(message)
    response = get_response(ints, data)
    
    # Return the response as JSON
    return jsonify({"response": response}), 200

if __name__ == "__main__":
    # Run the app on the specific IP and port
    app.run(host='0.0.0.0', port=8000)
