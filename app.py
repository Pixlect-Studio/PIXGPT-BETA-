# app.py – Flask + Gunicorn entrypoint
import os
import re
import random
from flask import Flask, request, jsonify

from ai_module import AIStorage

app = Flask(__name__)

# Robust storage path: always resolve relative to this file
_STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'storage.json')
storage = AIStorage(_STORAGE_PATH)

def process_message(message: str) -> str:
    """Returns a string response."""
    if not message:
        return "Please say something."

    words = re.findall(r'\w+', message.lower())

    # 1. Try Markov
    markov_resp = storage.markov.generate(
        seed_words=words[-2:] if len(words) >= 2 else None
    )
    if markov_resp:
        return markov_resp

    # 2. Try RNN
    seed = message[:10] if len(message) > 5 else 'the'
    rnn_resp = storage.rnn.generate(seed, length=30)
    if rnn_resp:
        return rnn_resp

    # 3. Pattern Matcher
    response = storage.pattern_matcher.respond(message)
    if response:
        return response

    # 4. Fallback
    fallbacks = [
        "That's interesting – tell me more!",
        "I'm listening. What else?",
        "Hmm, I'll remember that.",
        "Can you teach me about that?",
        "I'm still learning, but I'm paying attention."
    ]
    return random.choice(fallbacks)

@app.get('/')
def index():
    # Serve front-end (needed for Railway)
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        return f.read()



@app.post('/chat')
def chat_message():
    payload = request.get_json(silent=True) or {}
    message = payload.get('message', '')
    return jsonify({"response": process_message(message)})


# Backwards-compatible alias
@app.post('/api/message')
def api_message():
    return chat_message()


@app.get('/status')
def status():
    # Pretty-print JSON for health checks
    return jsonify({"status": "ok"})



if __name__ == '__main__':
    # Local dev only; Railway uses gunicorn
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)

