# app.py – Browser‑compatible version
import re
import random
from ai_module import AIStorage

# Load the AI from storage.json (will be in virtual FS)
storage = AIStorage('storage.json')

def process_message(message):
    """Called from JavaScript – returns a string response."""
    if not message:
        return "Please say something."

    words = re.findall(r'\w+', message.lower())

    # 1. Try Markov
    markov_resp = storage.markov.generate(
        seed_words=words[-2:] if len(words) >= 2 else None
    )
    if markov_resp and len(markov_resp) > 0:
        return markov_resp

    # 2. Try RNN
    seed = message[:10] if len(message) > 5 else 'the'
    rnn_resp = storage.rnn.generate(seed, length=30)
    if rnn_resp and len(rnn_resp) > 0:
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