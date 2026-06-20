import torch
from torch.utils.data import Dataset
import urllib.request

class ShakespeareDataset(Dataset):
    def __init__(self, max_seq_len, split="train"):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        file_path = "tinyshakespeare.txt"
        urllib.request.urlretrieve(url, file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # character level tokenization
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.max_seq_len = max_seq_len

        data = torch.tensor([self.char_to_int[c] for c in text])
        n = len(data)

        if split == "train":
            self.data = data[:int(0.8 * n)]
        elif split == "val":
            self.data = data[int(0.8 * n) : int(0.9 * n)]
        else:
            self.data = data[int(0.9 * n):]
        self.len = len(self.data)

    def __len__(self):
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.max_seq_len]
        y = self.data[idx + 1 : idx + self.max_seq_len + 1]
        return x, y
