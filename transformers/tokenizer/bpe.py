from collections import defaultdict
import json
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.char_to_int = {}
        self.int_to_char = {}
        self.merge_rules = []
        self.vocab_size = 0

    def train(self, text, target_vocab_size):
        chars = sorted(set(text))
        vocab_size = len(chars)
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}
        corpus = [char_to_int[c] for c in text]
        merge_rules = []
        with tqdm(total=target_vocab_size - vocab_size, desc="BPE merges") as pbar:
            while vocab_size < target_vocab_size:
                frequency = defaultdict(int)
                max_freq_pair = None

                for x, y in zip(corpus[:-1], corpus[1:]):
                    pair = (x, y)
                    frequency[pair] += 1
                    if max_freq_pair is None:
                        max_freq_pair = pair
                    elif frequency[max_freq_pair] < frequency[pair]:
                        max_freq_pair = pair

                if max_freq_pair is None:
                    break

                c1, c2 = int_to_char[max_freq_pair[0]], int_to_char[max_freq_pair[1]]
                new_pair_text = f"{c1}{c2}"
                char_to_int[new_pair_text] = vocab_size
                int_to_char[vocab_size] = new_pair_text
                merge_rules.append((max_freq_pair, vocab_size))
                vocab_size += 1
                pbar.update(1)

                new_corpus = []
                i = 0
                while i < len(corpus):
                    idx = corpus[i]
                    if idx == max_freq_pair[0] and i + 1 < len(corpus) and corpus[i + 1] == max_freq_pair[1]:
                        new_corpus.append(char_to_int[new_pair_text])
                        i += 2
                    else:
                        new_corpus.append(idx)
                        i += 1
                corpus = new_corpus

        # update members
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char
        self.merge_rules = merge_rules
        self.vocab_size = vocab_size

    def encode(self, text):
        tokens = [self.char_to_int[c] for c in text]
        for merge_rule in self.merge_rules:
            i = 0
            new_tokens = []
            while i < len(tokens):
                x, y = merge_rule[0]
                new_id = merge_rule[1]
                if tokens[i] == x and i + 1 < len(tokens) and tokens[i + 1] == y:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens


    def decode(self, tokens):
        text = ""
        for token in tokens:
            decoded_text = self.int_to_char[token]
            text += decoded_text

        return text

    def save(self, path):
        data = {
            "char_to_int": self.char_to_int,
            "int_to_char": self.int_to_char,
            "merge_rules": self.merge_rules,
            "vocab_size": self.vocab_size
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        
        self.char_to_int = data["char_to_int"]
        self.int_to_char = {int(k): v for k, v in data["int_to_char"].items()}
        self.merge_rules = data["merge_rules"]
        self.vocab_size = data["vocab_size"]

def test_tokenizer():
    corpus = "hello world " * 100
    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=30)

    text = "hello world"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text, f"Round-trip failed: {decoded!r} != {text!r}"
    assert len(encoded) < len(text), f"No compression: {len(encoded)} >= {len(text)}"
    assert tok.vocab_size <= 30, f"Wrong vocab size: {tok.vocab_size}"

    print(f"vocab_size: {tok.vocab_size}")
    print(f"encode({text!r}): {encoded}")
    print(f"decode: {decoded!r}")
    print("All tests passed.")

if __name__ == "__main__":
    import argparse, urllib.request, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, help="Path to text file")
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--save", type=str, default=None, help="Path to save tokenizer JSON")
    parser.add_argument("--sample", type=str, default="To be, or not to be, that is the question.")
    args = parser.parse_args()

    if args.test:
        test_tokenizer()
        exit(0)

    if args.dataset:
        with open(args.dataset, "r") as f:
            text = f.read()
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        path = "tinyshakespeare.txt"
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        with open(path, "r") as f:
            text = f.read()

    tok = BPETokenizer()
    tok.train(text, target_vocab_size=args.vocab_size)
    if args.save:
        tok.save(args.save)
        print(f"Tokenizer saved to {args.save}")

    encoded = tok.encode(args.sample)
    print(f"Text: {args.sample!r}")
    print(f"Tokens ({len(encoded)}): {encoded}")
    print(f"Decoded: {tok.decode(encoded)!r}")
    print(f"Vocab size: {tok.vocab_size}, Merges: {len(tok.merge_rules)}")