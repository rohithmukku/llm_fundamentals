from gpt2.config import TrainerConfig, ModelConfig
import argparse
import dataclasses
import urllib.request
import torch
import numpy as np
import csv
import os
from torch.utils.data import DataLoader, Dataset
from gpt2.gpt2 import GPT2
from torch.optim import AdamW
import math
import time
from dataset import ShakespeareDataset, WikiTextDataset


def get_dataset(ds, max_seq_len, tokenizer, split):
    if ds == "shakespeare":
        return ShakespeareDataset(max_seq_len, split)
    elif ds == "wikitext":
        return WikiTextDataset(max_seq_len, tokenizer, split)

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
        self.tokenizer = None

    def set_seed(self):
        seed = self.trainer_config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def prepare_setup(self, split="train"):
        tokenizer_arg = self.args.tokenizer or "tiktoken"
        self.dataset = get_dataset(self.args.dataset, self.model_config.max_seq_len, tokenizer_arg, split)
        self.dataloader = DataLoader(self.dataset, batch_size=self.trainer_config.batch_size, shuffle=True)

        self.model = GPT2(self.model_config, vocab_size=self.dataloader.dataset.vocab_size)
        self.model.to(self.device)
        if split == "train":
            val_dataset = get_dataset(self.args.dataset, self.model_config.max_seq_len, tokenizer_arg, "val")
            self.val_dataloader = DataLoader(val_dataset, batch_size=self.trainer_config.batch_size, shuffle=False)
            lr = self.trainer_config.lr
            self.optim = AdamW(self.model.parameters(), lr=lr)
            self.model.train()
        else:
            self.model.eval()

    def train(self):
        log_file = getattr(self.args, "log_file", None)
        csv_writer = None
        if log_file:
            f = open(log_file, "w", newline="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(["step", "train_loss", "val_loss", "lr"])

        step = 0
        stop = False
        t0 = time.perf_counter()
        while not stop:
            for batch in self.dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type != "cpu"):
                    logits, loss = self.model(x, y)

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_norm_clip)
                self.optim.step()

                step += 1

                lr = self.trainer_config.lr * 0.5 * (1 + math.cos(math.pi * step / self.trainer_config.steps))
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr

                if step % self.trainer_config.log_step == 0:
                    val_loss = self.eval(self.val_dataloader, max_batches=20)
                    elapsed = time.perf_counter() - t0
                    print(f"Step: {step}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}, Time: {elapsed:.1f}s")
                    if csv_writer:
                        csv_writer.writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}", f"{lr:.6f}"])
                        f.flush()

                if step >= self.trainer_config.steps:
                    stop = True
                    break

        total_time = time.perf_counter() - t0
        print(f"Training complete. Total time: {total_time/60:.1f} min ({total_time:.1f}s)")
        if log_file:
            f.close()
        self.save_model(step, loss)

    def eval(self, dataloader=None, max_batches=None):
        dataloader = dataloader or self.dataloader
        was_training = self.model.training
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
                if max_batches and len(losses) >= max_batches:
                    break
        if was_training:
            self.model.train()
        avg_loss = sum(losses) / len(losses)
        if not max_batches:
            print(f"Average loss: {avg_loss:.4f}")
        return avg_loss

    def run(self):
        self.set_seed()

        if self.args.mode == "train":
            self.prepare_setup("train")
            self.train()
        elif self.args.mode == "eval":
            self.prepare_setup("val")
            self.eval(self.dataloader)
        else:
            self.prepare_setup("test")
            self.eval(self.dataloader)

    def save_model(self, step, loss):
        out_dir = self.args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "model_config": dataclasses.asdict(self.model_config),
            "step": step,
            "loss": loss.item(),
        }
        model_path = os.path.join(out_dir, "model.pth")
        torch.save(checkpoint, model_path)
        print(f"Saved checkpoint to {model_path} at step {step}")
        if self.args.tokenizer:
            import shutil
            tok_dst = os.path.join(out_dir, "tokenizer.json")
            shutil.copy(self.args.tokenizer, tok_dst)
            print(f"Saved tokenizer to {tok_dst}")


def _add_dataclass_args(parser, cls):
    for f in dataclasses.fields(cls):
        if isinstance(f.default, bool):
            parser.add_argument(f"--{f.name}", action="store_true", default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--dataset", type=str, default="shakespeare", choices=["shakespeare", "wikitext"])
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to Tokenizer")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save model and tokenizer")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    _add_dataclass_args(parser, ModelConfig)
    _add_dataclass_args(parser, TrainerConfig)
    return parser.parse_args()

def prepare_configs(args):
    args_dict = vars(args)
    model_config = ModelConfig(**{f.name: args_dict[f.name] for f in dataclasses.fields(ModelConfig)})
    trainer_config = TrainerConfig(**{f.name: args_dict[f.name] for f in dataclasses.fields(TrainerConfig)})
    return model_config, trainer_config

def plot_loss(log_file, out="loss_curve.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["train_loss"], label="train")
    plt.plot(df["step"], df["val_loss"], label="val")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")

if __name__ == "__main__":
    args = parse_args()
    if args.plot:
        plot_loss(args.log_file)
    else:
        model_config, trainer_config = prepare_configs(args)
        trainer = Trainer(args, trainer_config, model_config)
        trainer.run()