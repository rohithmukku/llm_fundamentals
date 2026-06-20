import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.bpe import BPETokenizer
import tiktoken

def get_tokenizer(tokenizer: str = "tiktoken"):
    # tokenizer can be either tiktoken or path
    if tokenizer == "tiktoken":
        enc = tiktoken.get_encoding("gpt2")
        return enc, enc.n_vocab
    else:
        tok = BPETokenizer()
        tok.load(tokenizer)
        return tok, tok.vocab_size

class WikiTextDataset(Dataset):
    def __init__(self, max_seq_len: int, tokenizer: str, split: str = "train") -> None:
        split_map = {
            "train": "train",
            "val": "validation",
            "test": "test"
        }

        self.tokenizer, self.vocab_size = get_tokenizer(tokenizer)

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        text = "\n".join(ds[split_map[split]]["text"])

        # character level tokenization
        cache_path = f"wikitext_{split}_tokens.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            torch.save(self.data, cache_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.max_seq_len]
        y = self.data[idx + 1 : idx + self.max_seq_len + 1]
        return x, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--out", type=str, default="tokenizer.json")
    args = parser.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n".join(ds["train"]["text"])

    tok = BPETokenizer()
    tok.train(text, args.vocab_size)
    tok.save(args.out)
    print(f"Tokenizer saved to {args.out} (vocab_size={tok.vocab_size})")
