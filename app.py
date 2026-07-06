from flask import Flask, request, jsonify, render_template
import re
import os
import threading
import random
import requests
from ai_module import AIStorage, PatternMatcher, MarkovChain, RNN

app = Flask(__name__)
storage = AIStorage('storage.json')

# ============================================================
#  GLOBAL SETTINGS
# ============================================================
ANSWER_MODE = 'creative'  # 'clean' or 'creative'

# ============================================================
#  EXTERNAL LOOKUPS (no API keys)
# ============================================================
def get_definition(word):
    """Fetch definition from Free Dictionary API."""
    word = word.strip().lower()
    if not word:
        return None
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and 'meanings' in data[0]:
                meanings = data[0]['meanings']
                for meaning in meanings:
                    if 'definitions' in meaning:
                        defs = meaning['definitions']
                        if defs:
                            return defs[0]['definition']
        return None
    except:
        return None

def search_wikipedia(query):
    """Search Wikipedia and return a short summary."""
    query = query.strip()
    if not query:
        return None
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': 1
    }
    try:
        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()
        if 'query' in data and 'search' in data['query'] and data['query']['search']:
            page_title = data['query']['search'][0]['title']
            extract_params = {
                'action': 'query',
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'titles': page_title,
                'format': 'json'
            }
            extract_res = requests.get(search_url, params=extract_params, timeout=5)
            extract_data = extract_res.json()
            pages = extract_data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if 'extract' in page_info:
                    extract = page_info['extract']
                    if len(extract) > 350:
                        extract = extract[:350] + '...'
                    return extract
        return None
    except:
        return None

# (DuckDuckGo function removed)

# ============================================================
#  PARAPHRASE + PERSONALITY ENGINE
# ============================================================
def remake_answer(topic, raw_text, mode='creative'):
    """Paraphrase raw text, add conversational comments, and optionally append a Markov‑generated extra sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', raw_text)
    core = ' '.join(sentences[:3]) if len(sentences) >= 3 else raw_text[:300]

    # Light paraphrasing (word substitutions)
    substitutes = [
        ("is", "refers to"),
        ("are", "refer to"),
        ("was", "used to be"),
        ("have", "possess"),
        ("has", "possesses"),
        ("in", "within"),
        ("for", "for the purpose of"),
        ("of", "belonging to"),
    ]
    for old, new in substitutes:
        core = core.replace(f' {old} ', f' {new} ')

    # Intro pool
    intro_pool = [
        f"From what I've gathered about {topic}, ",
        f"I did a quick search and found that {topic} ",
        f"Here's the scoop on {topic}: ",
        f"Let me tell you what I know about {topic}: ",
        f"I looked into {topic} and it seems that ",
        f"Regarding {topic}, the main thing is "
    ]
    intro = random.choice(intro_pool)

    # Conversational comments
    comments = [
        " That's pretty interesting, isn't it?",
        " I didn't know that before!",
        " It's always fascinating to learn new things.",
        " That's just a quick summary – there's a lot more to it.",
        " Pretty cool, right?",
        " It's one of those things that you don't think about every day.",
        " Makes you wonder, doesn't it?",
    ]
    comment = random.choice(comments)

    # Extra sentence from Markov if available and mode != 'clean'
    extra = ''
    if mode != 'clean' and storage.markov.chain:
        seed_words = re.findall(r'\w+', topic.lower())
        if seed_words:
            markov_extra = storage.markov.generate(seed_words=seed_words, max_words=8)
            if markov_extra and len(markov_extra) > 3:
                extra = ' ' + markov_extra.capitalize() + '.'

    answer = intro + core.capitalize()
    if extra:
        answer += extra
    answer += comment

    if not answer.endswith(('.', '!', '?')):
        answer += '.'

    return answer

# ============================================================
#  CASUAL CHAT DETECTION (unchanged)
# ============================================================
casual_patterns = {
    r'hi|hello|hey': ["Hey there!", "Hello!", "Hi, how can I help?", "Hey, what's on your mind?"],
    r'how are you|how\'s it going|how are things': [
        "I'm doing great – always ready to chat! How about you?",
        "Pretty good, thanks for asking! What's new with you?",
        "All systems operational! 😊 How are you doing?"
    ],
    r'what\'s up|sup|wassup': [
        "Not much, just thinking about all the data I've learned. What's up with you?",
        "Just hanging out in the cloud. How can I assist you?",
        "Hey! Ready to talk about anything you like."
    ],
    r'good|fine|ok|okay': [
        "Glad to hear that!",
        "Awesome!",
        "Great! Feel free to ask me anything.",
    ],
    r'thanks|thank you': [
        "You're welcome!",
        "Anytime!",
        "Happy to help!",
    ],
}
def get_casual_response(text):
    for pattern, responses in casual_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return random.choice(responses)
    return None

# ============================================================
#  BACKGROUND TRAINING STATUS
# ============================================================
training_status = {
    'running': False,
    'logs': [],
    'done': False,
    'message': ''
}

def run_background_training(folder_path, mode='both'):
    global training_status
    training_status['running'] = True
    training_status['done'] = False
    training_status['logs'] = []
    training_status['message'] = f'Training started on {folder_path} (mode: {mode})...'

    try:
        train_markov = mode in ['markov', 'both']
        train_rnn = mode in ['rnn', 'both']
        storage.train_on_files(
            folder_path,
            train_markov=train_markov,
            train_rnn=train_rnn,
            max_rnn_chars=150000,
            rnn_epochs=35,
            rnn_max_bptt=150
        )
        storage.save()
        training_status['message'] = '✅ Training completed successfully!'
        training_status['logs'].append('Training complete and saved.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        training_status['message'] = f'❌ Error: {str(e)}'
        training_status['logs'].append(f'Error: {str(e)}')
    finally:
        training_status['running'] = False
        training_status['done'] = True

# ============================================================
#  ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global ANSWER_MODE
    data = request.json
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Please say something.'})

    # ---- COMMANDS ----
    lower_msg = user_msg.lower()
    if lower_msg == '/clean':
        ANSWER_MODE = 'clean'
        return jsonify({'response': '✅ Switched to **clean** mode (no Markov extras).'})
    elif lower_msg == '/creative':
        ANSWER_MODE = 'creative'
        return jsonify({'response': '✅ Switched to **creative** mode (adds Markov extras).'})
    elif lower_msg.startswith('/train_web '):
        url = user_msg[10:].strip()
        if not url.startswith(('http://', 'https://')):
            return jsonify({'response': '⚠️ Please include a valid URL starting with http:// or https://'})
        def background_web_train():
            from app import train_from_url
            result = train_from_url(url, epochs=30)
            print(f"🌐 Web training result: {result}")
        thread = threading.Thread(target=background_web_train)
        thread.daemon = True
        thread.start()
        return jsonify({'response': f'🌐 Training started on:\n{url}\n\n📚 I\'m learning from this page in the background. Check /debug for progress.'})

    # ---- CASUAL CHAT ----
    casual = get_casual_response(user_msg)
    if casual:
        return jsonify({'response': casual})

    # ---- DEFINITION / EXPLANATION QUERY ----
    patterns = [
        (r'^define\s+(.+)', 'define'),
        (r'^meaning of\s+(.+)', 'define'),
        (r'^what is\s+(.+)', 'what'),
        (r'^what does\s+(.+)\s+mean', 'what'),
        (r'^what\'s\s+(.+)', 'what'),
        (r'^what are\s+(.+)', 'what'),
        (r'^explain\s+(.+)', 'explain'),
        (r'^tell me about\s+(.+)', 'tell'),
        (r'^wikipedia\s+(.+)', 'wikipedia'),
        (r'^wiki\s+(.+)', 'wikipedia'),
    ]

    for pattern, typ in patterns:
        match = re.match(pattern, lower_msg)
        if match:
            phrase = match.group(1).strip().strip('?')
            # Try dictionary first
            if typ not in ['wikipedia']:
                definition = get_definition(phrase)
                if definition:
                    remade = remake_answer(phrase, definition, mode=ANSWER_MODE)
                    return jsonify({'response': remade})
            # Try Wikipedia
            wiki_result = search_wikipedia(phrase)
            if wiki_result:
                remade = remake_answer(phrase, wiki_result, mode=ANSWER_MODE)
                return jsonify({'response': remade})
            # If nothing found
            return jsonify({'response': f"I couldn't find information about '{phrase}'. Can you teach me?"})

    # ---- NORMAL CHAT (Markov / RNN / Pattern / Fallback) ----
    words = re.findall(r'\w+', user_msg.lower())
    markov_resp = storage.markov.generate(seed_words=words[-2:] if len(words) >= 2 else None)
    if markov_resp and len(markov_resp) > 0:
        return jsonify({'response': markov_resp})

    seed = user_msg[:10] if len(user_msg) > 5 else 'the'
    rnn_resp = storage.rnn.generate(seed, length=30)
    if rnn_resp and len(rnn_resp) > 0:
        return jsonify({'response': rnn_resp})

    response = storage.pattern_matcher.respond(user_msg)
    if response:
        return jsonify({'response': response})

    fallbacks = [
        "That's interesting – tell me more!",
        "I'm listening. What else?",
        "Hmm, I'll remember that.",
        "Can you teach me about that?",
        "I'm still learning, but I'm paying attention."
    ]
    return jsonify({'response': random.choice(fallbacks)})

# ============================================================
#  TRAIN ROUTES (unchanged)
# ============================================================
@app.route('/train', methods=['POST'])
def train():
    data = request.json
    text = data.get('text', '').strip()
    epochs = data.get('epochs', 20)
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided.'})

    try:
        storage.markov.train(text)
        storage.rnn.train(text, epochs=epochs, verbose=False)
        sentences = re.findall(r'[^.!?]*\?', text)
        for sent in sentences[:10]:
            words = sent.lower().split()
            if len(words) > 3:
                pattern = r'.*'.join(words[:3])
                storage.pattern_matcher.add_rule(pattern, [sent])
        storage.save()
        return jsonify({'status': 'success', 'message': f'Training completed with {epochs} epochs.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_folder', methods=['POST'])
def train_folder():
    global training_status
    data = request.json
    folder = data.get('folder', '').strip()
    mode = data.get('mode', 'both')

    possible_paths = [
        folder,
        'Training',
        '.Training',
        './Training',
        './.Training',
        os.path.join(os.getcwd(), 'Training'),
        os.path.join(os.getcwd(), '.Training')
    ]
    found_path = None
    for path in possible_paths:
        if path and os.path.isdir(path):
            found_path = path
            break

    if not found_path:
        return jsonify({'status': 'error', 'message': f'Folder not found. Tried: {possible_paths}'})

    if training_status['running']:
        return jsonify({'status': 'busy', 'message': 'Training already running.'})

    print(f"\n📁 Training on folder: {found_path} (mode: {mode})")
    thread = threading.Thread(target=run_background_training, args=(found_path, mode))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'started', 'message': f'Training started on {found_path} (mode: {mode}).'})

@app.route('/train_rnn_background', methods=['POST'])
def train_rnn_background():
    global training_status
    if training_status['running']:
        return jsonify({'status': 'busy', 'message': 'Training already running.'})

    possible_paths = ['Training', '.Training', './Training', './.Training']
    found_path = None
    for path in possible_paths:
        if os.path.isdir(path):
            found_path = path
            break

    if not found_path:
        return jsonify({'status': 'error', 'message': 'Training folder not found.'})

    print(f"\n🤖 Training RNN ONLY on folder: {found_path}")
    thread = threading.Thread(target=run_background_training, args=(found_path, 'rnn'))
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'started', 'message': f'RNN training started on {found_path}.'})

# ============================================================
#  WEB TRAINING (from URL) – unchanged
# ============================================================
def fetch_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')

        if 'text/plain' in content_type or 'application/json' in content_type or 'text/csv' in content_type:
            return response.text

        if 'text/html' in content_type:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                text = soup.get_text(separator='\n')
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
            except ImportError:
                text = re.sub(r'<[^>]+>', ' ', response.text)
                text = re.sub(r'\s+', ' ', text)
                return text
        return None
    except Exception as e:
        print(f"Web fetch error: {e}")
        return None

def train_from_url(url, epochs=30):
    print(f"🌐 Fetching text from: {url}")
    text = fetch_text_from_url(url)
    if not text or len(text) < 100:
        return {'status': 'error', 'message': 'Could not fetch enough text from URL.'}

    os.makedirs('Training', exist_ok=True)
    filename = f"Training/web_train_{int(time.time())}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"📁 Saved {len(text)} chars to {filename}")

    storage.markov.train(text)
    storage.rnn.train(text[:80000], epochs=epochs, verbose=True)

    sentences = re.findall(r'[^.!?]*\?', text)
    for sent in sentences[:10]:
        words = sent.lower().split()
        if len(words) > 3:
            pattern = r'.*'.join(words[:3])
            storage.pattern_matcher.add_rule(pattern, [sent])

    storage.save()
    return {'status': 'success', 'message': f'Trained on {len(text)} chars from {url}.'}

@app.route('/train_web', methods=['POST'])
def train_web():
    data = request.json
    url = data.get('url', '').strip()
    epochs = data.get('epochs', 30)
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided.'})

    def background_web_train():
        result = train_from_url(url, epochs=epochs)
        print(f"Web training result: {result}")

    thread = threading.Thread(target=background_web_train)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'started', 'message': f'Training started on {url} in background.'})

@app.route('/train_status', methods=['GET'])
def train_status():
    return jsonify(training_status)

@app.route('/clear', methods=['POST'])
def clear():
    storage.pattern_matcher = PatternMatcher()
    storage.markov = MarkovChain()
    storage.rnn = RNN()
    storage.save()
    return jsonify({'status': 'success', 'message': 'Models reset.'})

# ============================================================
#  DEBUG
# ============================================================
@app.route('/debug')
def debug():
    possible_folders = ['Training', '.Training', './Training', './.Training']
    training_folder = None
    for folder in possible_folders:
        if os.path.exists(folder):
            training_folder = folder
            break

    files = []
    if training_folder and os.path.exists(training_folder):
        for f in os.listdir(training_folder):
            if f.endswith('.txt'):
                path = os.path.join(training_folder, f)
                size = os.path.getsize(path)
                files.append({'name': f, 'size': size})

    markov_chain = storage.markov.chain
    markov_entries = len(markov_chain)
    vocab_size = len(storage.markov.vocab)
    sample_entries = list(markov_chain.items())[:5] if markov_chain else []

    rnn_trained = storage.rnn.trained
    rnn_hidden = storage.rnn.hidden_size if hasattr(storage.rnn, 'hidden_size') else 'N/A'
    rnn_vocab = storage.rnn.vocab_size if hasattr(storage.rnn, 'vocab_size') else 0
    pattern_count = len(storage.pattern_matcher.rules)

    sample_text = ''
    if rnn_trained:
        try:
            sample_text = storage.rnn.generate('the ', length=30)
            if sample_text is None:
                sample_text = '⚠️ RNN generation returned None.'
        except Exception as e:
            sample_text = f'⚠️ RNN generation error: {str(e)}'

    return render_template('debug.html',
                           files=files,
                           markov_entries=markov_entries,
                           vocab_size=vocab_size,
                           sample_entries=sample_entries,
                           rnn_trained=rnn_trained,
                           rnn_hidden=rnn_hidden,
                           rnn_vocab=rnn_vocab,
                           pattern_count=pattern_count,
                           sample_text=sample_text)

# ============================================================
#  RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)