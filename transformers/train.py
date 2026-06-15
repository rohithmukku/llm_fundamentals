from gpt2.config import TrainerConfig, ModelConfig
import argparse
import dataclasses
import urllib.request
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from gpt2.gpt2 import GPT2
from torch.optim import AdamW
import math

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
        

class Trainer:
    def __init__(self, args, trainer_config: TrainerConfig, model_config: ModelConfig):
        self.args = args
        self.trainer_config = trainer_config
        self.model_config = model_config
        # default device to cpu
        self.device = torch.device("cpu")
        if torch.accelerator.is_available():
            self.device = torch.device(torch.accelerator.current_accelerator().type)
            print(f"Using device type: {self.device}")

        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optim = None

    def set_seed(self):
        seed = self.trainer_config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def prepare_setup(self, split="train"):
        self.dataset = ShakespeareDataset(self.model_config.max_seq_len, split)
        self.dataloader = DataLoader(self.dataset, batch_size=self.trainer_config.batch_size, shuffle=True)

        self.model = GPT2(self.model_config, vocab_size=self.dataloader.dataset.vocab_size)
        self.model.to(self.device)
        if split == "train":
            lr = self.trainer_config.lr
            self.optim = AdamW(self.model.parameters(), lr=lr)
            self.model.train()
        else:
            self.model.eval()

    def train(self):
        step = 0
        stop = False
        while not stop:
            for batch in self.dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                logits, loss = self.model(x, y)

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_norm_clip)
                self.optim.step()

                step += 1

                # adjust learning rate/decay
                lr = self.trainer_config.lr * 0.5 * (1 + math.cos(math.pi * step / self.trainer_config.steps))
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr

                if step % self.trainer_config.log_step == 0:
                    print(f"Step: {step}, Loss: {loss.item()}")

                if step >= self.trainer_config.steps:
                    stop = True
                    break
        self.save_model(step, loss)

    def eval(self):
        losses = []
        with torch.no_grad():
            for batch in self.dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                logits, loss = self.model(x, y)
                losses.append(loss.cpu().item())
        
        avg_loss = sum(losses)/len(losses)
        print(f"Average loss: {avg_loss}")

    def run(self):
        self.set_seed()

        if self.args.mode == "train":
            self.prepare_setup("train")
            self.train()
        elif self.args.mode == "eval":
            self.prepare_setup("val")
            self.eval()
        else:
            self.prepare_setup("test")
            self.eval()

    def save_model(self, step, loss):
        path = "./minigpt.pth"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "model_config": dataclasses.asdict(self.model_config),
            "step": step,
            "loss": loss.item(),
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path} at step {step}")


def _add_dataclass_args(parser, cls):
    for f in dataclasses.fields(cls):
        if isinstance(f.default, bool):
            parser.add_argument(f"--{f.name}", action="store_true", default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    _add_dataclass_args(parser, ModelConfig)
    _add_dataclass_args(parser, TrainerConfig)
    return parser.parse_args()

def prepare_configs(args):
    args_dict = vars(args)
    model_config = ModelConfig(**{f.name: args_dict[f.name] for f in dataclasses.fields(ModelConfig)})
    trainer_config = TrainerConfig(**{f.name: args_dict[f.name] for f in dataclasses.fields(TrainerConfig)})
    return model_config, trainer_config

if __name__ == "__main__":
    args = parse_args()
    model_config, trainer_config = prepare_configs(args)
    trainer = Trainer(args, trainer_config, model_config)
    trainer.run()