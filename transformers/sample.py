import torch
import urllib.request
import os
import argparse
from gpt2.gpt2 import GPT2
from gpt2.config import ModelConfig

def load_vocab():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "tinyshakespeare.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_int, int_to_char, len(chars)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="minigpt.pth")
    parser.add_argument("--prompt", type=str, default="O Romeo")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--do_sample", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.accelerator.is_available():
        device = torch.device(torch.accelerator.current_accelerator().type)

    char_to_int, int_to_char, vocab_size = load_vocab()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config_dict = checkpoint["model_config"]
    model_config_dict.setdefault("use_rope", False)  # old checkpoints predate RoPE
    model_config = ModelConfig(**model_config_dict)
    model = GPT2(model_config, vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    prompt_ids = [char_to_int[c] for c in args.prompt]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.do_sample,
    )

    generated = ''.join([int_to_char[i] for i in output[0].tolist()])
    print(generated)

if __name__ == "__main__":
    main()
