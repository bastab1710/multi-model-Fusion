"""
tokenizer.py
Lightweight vocabulary builder and text tokenizer for the Text Branch.
Standardizes text, builds word-to-index mapping, and converts captions into padded token sequences.
"""

import os
import re
import json
from collections import Counter


class CaptionTokenizer:
    """Tokenizes captions into fixed-length integer sequences."""

    PAD_TOKEN = "<pad>"  # index 0
    UNK_TOKEN = "<unk>"  # index 1

    def __init__(self, vocab_size=5000, max_seq_len=30):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    @staticmethod
    def clean_text(text):
        """Lowercases and strips punctuation."""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def fit_on_texts(self, texts):
        """Builds vocabulary from a collection of text strings."""
        counter = Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            counter.update(words)

        # Most frequent words up to vocab_size
        available_slots = self.vocab_size - 2
        for word, _ in counter.most_common(available_slots):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"[Tokenizer] Vocabulary built with {len(self.word2idx)} tokens (max {self.vocab_size}).")

    def encode(self, text):
        """Converts a single string into a fixed-length list of token IDs."""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        tokens = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in words]
        # Truncate or pad to max_seq_len
        tokens = tokens[:self.max_seq_len]
        if len(tokens) < self.max_seq_len:
            tokens += [self.word2idx[self.PAD_TOKEN]] * (self.max_seq_len - len(tokens))
        return tokens

    def save_vocab(self, save_path="vocab.json"):
        """Saves vocabulary mapping to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "word2idx": self.word2idx,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Tokenizer] Vocabulary saved to {save_path}")

    @classmethod
    def load_vocab(cls, load_path="vocab.json"):
        """Loads vocabulary mapping from JSON."""
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], max_seq_len=data["max_seq_len"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(idx): word for word, idx in tok.word2idx.items()}
        return tok

