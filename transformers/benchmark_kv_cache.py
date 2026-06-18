import torch
import time
import argparse
from gpt2.gpt2 import GPT2
from gpt2.config import ModelConfig

def benchmark(model, idx, max_new_tokens, use_kv_cache, n_runs=3):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.generate(idx, max_new_tokens=max_new_tokens, use_kv_cache=use_kv_cache)
        times.append(time.perf_counter() - start)
    return sum(times) / n_runs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.accelerator.is_available():
        device = torch.device(torch.accelerator.current_accelerator().type)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    config = ModelConfig(
        n_embed=384, n_heads=6, n_layers=6,
        max_seq_len=512, dropout=0.0, bias=False,
        use_torch_dot_product=True, use_rope=True
    )
    model = GPT2(config, vocab_size=65).to(device)
    model.eval()

    prompt_len = 32
    idx = torch.randint(0, 65, (1, prompt_len), device=device)

    print(f"{'tokens':>8} {'no cache (s)':>14} {'kv cache (s)':>14} {'speedup':>10}")
    print("-" * 52)

    for max_new_tokens in [32, 64, 128, 256]:
        t_no_cache = benchmark(model, idx, max_new_tokens, use_kv_cache=False)
        t_kv_cache = benchmark(model, idx, max_new_tokens, use_kv_cache=True)
        speedup = t_no_cache / t_kv_cache
        print(f"{max_new_tokens:>8} {t_no_cache:>14.3f} {t_kv_cache:>14.3f} {speedup:>9.2f}x")

if __name__ == "__main__":
    main()
