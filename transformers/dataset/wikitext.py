import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.bpe import BPETokenizer

class WikiTextDataset(Dataset):
    def __init__(self, max_seq_len: int, tokenizer: BPETokenizer, split: str = "train") -> None:
        split_map = {
            "train": "train",
            "val": "validation",
            "test": "test"
        }

        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text = "\n".join(ds[split_map[split]]["text"])

        # character level tokenization
        cache_path = f"wikitext_{split}_tokens.pt"
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            torch.save(self.data, cache_path)
        self.max_seq_len = max_seq_len
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.max_seq_len]
        y = self.data[idx + 1 : idx + self.max_seq_len + 1]
        return x, y
