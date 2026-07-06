import numpy as np
import json
import re
import os
import random
from collections import defaultdict

# ============================================================
#  PATTERN MATCHER (ELIZA‑style)
# ============================================================
class PatternMatcher:
    def __init__(self):
        self.rules = []  # list of (compiled_regex, list_of_responses)

    def add_rule(self, pattern, responses):
        self.rules.append((re.compile(pattern, re.IGNORECASE), responses))

    def respond(self, text):
        for pattern, responses in self.rules:
            if pattern.search(text):
                return random.choice(responses)
        return None

    def to_dict(self):
        return {'rules': [(p.pattern, r) for p, r in self.rules]}

    def from_dict(self, data):
        self.rules = []
        for pattern, responses in data['rules']:
            self.rules.append((re.compile(pattern, re.IGNORECASE), responses))


# ============================================================
#  MARKOV CHAIN (2nd order)
# ============================================================
class MarkovChain:
    def __init__(self):
        self.chain = defaultdict(list)
        self.vocab = set()

    def train(self, text):
        words = re.findall(r'\w+', text.lower())
        print(f"🔍 Markov: Found {len(words)} words")
        self.vocab.update(words)
        for i in range(len(words) - 2):
            key = (words[i], words[i+1])
            self.chain[key].append(words[i+2])
        print(f"🔍 Markov: Built {len(self.chain)} entries")

    def generate(self, seed_words=None, max_words=20):
        if not self.chain:
            return None
        if seed_words and len(seed_words) >= 2:
            key = (seed_words[-2].lower(), seed_words[-1].lower())
            if key not in self.chain:
                key = random.choice(list(self.chain.keys()))
        else:
            key = random.choice(list(self.chain.keys()))

        result = list(key)
        for _ in range(max_words - 2):
            if key not in self.chain:
                break
            next_word = random.choice(self.chain[key])
            result.append(next_word)
            key = (result[-2], result[-1])
        return ' '.join(result)

    def to_dict(self):
        return {
            'chain': {f"{k[0]} {k[1]}": v for k, v in self.chain.items()},
            'vocab': list(self.vocab)
        }

    def from_dict(self, data):
        self.chain = defaultdict(list)
        for key_str, values in data['chain'].items():
            parts = key_str.split(' ')
            if len(parts) == 2:
                self.chain[(parts[0], parts[1])] = values
        self.vocab = set(data.get('vocab', []))


# ============================================================
#  RNN (Character‑level, NumPy accelerated)
# ============================================================
class RNN:
    def __init__(self, hidden_size=128, learning_rate=0.1):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.Wxh = None
        self.Whh = None
        self.Why = None
        self.bh = None
        self.by = None
        self.char_to_idx = {}
        self.idx_to_char = []
        self.vocab_size = 0
        self.trained = False

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_deriv(self, x):
        return x * (1 - x)

    def _init_weights(self, vocab_size):
        self.vocab_size = vocab_size
        self.Wxh = np.random.randn(self.hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, self.hidden_size) * 0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def train(self, text, epochs=30, verbose=True, max_backprop_steps=100):
        if len(text) < 3:
            raise ValueError("Text too short (need at least 3 chars)")

        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = chars
        self.vocab_size = len(chars)
        if self.Wxh is None:
            self._init_weights(self.vocab_size)
        else:
            if self.Wxh.shape[1] != self.vocab_size:
                print(f"⚠️ RNN vocab size changed ({self.Wxh.shape[1]} -> {self.vocab_size}). Reinitializing weights.")
                self._init_weights(self.vocab_size)

        data = [self.char_to_idx[ch] for ch in text]
        num_steps = len(data) - 1
        print(f"🤖 RNN: Training on {num_steps} chars, vocab={self.vocab_size}")

        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden_size, 1))
            loss = 0.0
            xs, hs, ys, ps = {}, {}, {}, {}

            # Forward pass
            for t in range(num_steps):
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][data[t]] = 1
                hs[t] = self._sigmoid(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, h_prev) + self.bh)
                ys[t] = np.dot(self.Why, hs[t]) + self.by
                exp_scores = np.exp(ys[t] - np.max(ys[t]))
                ps[t] = exp_scores / np.sum(exp_scores)
                loss += -np.log(ps[t][data[t+1], 0] + 1e-10)
                h_prev = hs[t]

            # Backward – truncated BPTT
            start = max(0, num_steps - max_backprop_steps)
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dWhy = np.zeros_like(self.Why)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])

            for t in range(num_steps - 1, start - 1, -1):
                dy = np.copy(ps[t])
                dy[data[t+1]] -= 1
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                dh = np.dot(self.Why.T, dy) + dh_next
                dh_raw = dh * self._sigmoid_deriv(hs[t])
                dWxh += np.dot(dh_raw, xs[t].T)
                if t > 0:
                    dWhh += np.dot(dh_raw, hs[t-1].T)
                else:
                    dWhh += np.dot(dh_raw, np.zeros_like(hs[t]).T)
                dbh += dh_raw
                dh_next = np.dot(self.Whh.T, dh_raw)

            # clip gradients
            for grad in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(grad, -5, 5, out=grad)

            # update weights
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.Why -= self.lr * dWhy
            self.bh -= self.lr * dbh
            self.by -= self.lr * dby

            if (epoch+1) % 10 == 0:
                self.lr *= 0.9

            if verbose and (epoch+1) % 5 == 0:
                print(f"RNN Epoch {epoch+1}/{epochs}, loss: {loss/num_steps:.4f}, lr: {self.lr:.5f}")

        self.trained = True
        print("✅ RNN training complete.")

    def generate(self, seed, length=30):
        if not self.trained:
            return None
        if not self.idx_to_char:
            return None

        if self.Wxh is not None and self.Wxh.shape[1] != self.vocab_size:
            print(f"⚠️ RNN generate: vocab mismatch ({self.Wxh.shape[1]} != {self.vocab_size}). Returning None.")
            return None

        if not seed:
            seed = random.choice(self.idx_to_char)
        result = list(seed)
        h = np.zeros((self.hidden_size, 1))

        for ch in seed:
            if ch not in self.char_to_idx:
                ch = random.choice(self.idx_to_char)
            x = np.zeros((self.vocab_size, 1))
            x[self.char_to_idx[ch]] = 1
            h = self._sigmoid(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

        for _ in range(length):
            x = np.zeros((self.vocab_size, 1))
            last_char = result[-1]
            if last_char not in self.char_to_idx:
                last_char = random.choice(self.idx_to_char)
            x[self.char_to_idx[last_char]] = 1
            h = self._sigmoid(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            probs = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            idx = np.random.choice(self.vocab_size, p=probs.ravel())
            next_char = self.idx_to_char[idx]
            result.append(next_char)
        return ''.join(result)

    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'lr': self.lr,
            'initial_lr': self.initial_lr,
            'Wxh': self.Wxh.tolist() if self.Wxh is not None else None,
            'Whh': self.Whh.tolist() if self.Whh is not None else None,
            'Why': self.Why.tolist() if self.Why is not None else None,
            'bh': self.bh.tolist() if self.bh is not None else None,
            'by': self.by.tolist() if self.by is not None else None,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'trained': self.trained
        }

    def from_dict(self, data):
        self.hidden_size = data['hidden_size']
        self.lr = data['lr']
        self.initial_lr = data.get('initial_lr', 0.1)
        self.Wxh = np.array(data['Wxh']) if data['Wxh'] is not None else None
        self.Whh = np.array(data['Whh']) if data['Whh'] is not None else None
        self.Why = np.array(data['Why']) if data['Why'] is not None else None
        self.bh = np.array(data['bh']) if data['bh'] is not None else None
        self.by = np.array(data['by']) if data['by'] is not None else None
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']
        self.trained = data['trained']


# ============================================================
#  STORAGE MANAGER
# ============================================================
class AIStorage:
    def __init__(self, filename='storage.json'):
        self.filename = filename
        self.pattern_matcher = PatternMatcher()
        self.markov = MarkovChain()
        self.rnn = RNN()
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
            self.pattern_matcher.from_dict(data.get('pattern_matcher', {'rules': []}))
            self.markov.from_dict(data.get('markov', {'chain': {}, 'vocab': []}))
            self.rnn.from_dict(data.get('rnn', {}))
        else:
            self.pattern_matcher.add_rule(r'hello|hi|hey', ['Hello! How can I help?', 'Hi there!', 'Hey!'])
            self.pattern_matcher.add_rule(r'how are you', ["I'm doing great! Thanks for asking.", "All systems operational."])
            self.save()

    def save(self):
        data = {
            'pattern_matcher': self.pattern_matcher.to_dict(),
            'markov': self.markov.to_dict(),
            'rnn': self.rnn.to_dict()
        }
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)
        print("💾 Storage saved.")

    def train_on_files(self, directory, extensions=['.txt', '.md', '.json', '.csv'],
                       train_markov=True, train_rnn=True, max_rnn_chars=150000,
                       rnn_epochs=35, rnn_max_bptt=150):
        texts = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content.strip():
                                texts.append(content)
                                print(f"✅ Read {path} ({len(content)} chars)")
                    except Exception as e:
                        print(f"❌ Could not read {path}: {e}")

        if not texts:
            print("❌ No text files found.")
            return

        full_text = '\n'.join(texts)
        print(f"\n📊 Total training text: {len(full_text)} characters")

        if train_markov:
            print("\n🧠 Training Markov chain...")
            self.markov.train(full_text)
            print(f"✅ Markov entries: {len(self.markov.chain)}")

        print("\n📊 Extracting question patterns...")
        sentences = re.findall(r'[^.!?]*\?', full_text)
        for sent in sentences[:15]:
            words = sent.lower().split()
            if len(words) > 3:
                pattern = r'.*'.join(words[:3])
                self.pattern_matcher.add_rule(pattern, [sent])

        if train_rnn and len(full_text) > 100:
            sample = full_text[:max_rnn_chars]
            print(f"\n🤖 Training RNN on first {len(sample)} chars...")
            self.rnn.train(sample, epochs=rnn_epochs, verbose=True, max_backprop_steps=rnn_max_bptt)
        else:
            print("⏭️ RNN skipped (disabled or text too short).")

        self.save()
        print("✅ Training complete.")