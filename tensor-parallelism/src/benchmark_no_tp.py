import argparse
import json
import time

import torch
import torch.nn as nn

from utils import count_parameters, get_gpu_memory_mb, get_gpu_peak_memory_mb

MODELS = {
    "gpt": ("model_gpt", "StandardGPT"),
    "llama": ("model_llama", "StandardLlama"),
}


def load_model(name):
    """Import and instantiate the requested model."""
    import importlib
    module_name, class_name = MODELS[name]
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)(), mod


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark (no TP)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="gpt",
                        help="Model architecture to benchmark (default: gpt)")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    model, mod = load_model(args.model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mem_model = get_gpu_memory_mb(device)
    n_params = count_parameters(model)

    BATCH_SIZE = mod.BATCH_SIZE
    SEQ_LEN = mod.SEQ_LEN
    VOCAB_SIZE = mod.VOCAB_SIZE
    NUM_WARMUP = mod.NUM_WARMUP
    NUM_BENCHMARK = mod.NUM_BENCHMARK

    torch.manual_seed(123)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    for _ in range(NUM_WARMUP):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    fwd_t, bwd_t, step_t = [], [], []
    for _ in range(NUM_BENCHMARK):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        fwd_t.append(t1-t0); bwd_t.append(t2-t1); step_t.append(t3-t0)

    peak = get_gpu_peak_memory_mb(device)
    avg = lambda lst: sum(lst)/len(lst)

    results = dict(
            mode="no_tp",
            arch=args.model,
            num_gpus=1,
            total_params=n_params,
            params_per_gpu=n_params,
            mem_model_mb=round(mem_model, 2),
            mem_peak_mb=round(peak, 2),
            fwd_ms=round(avg(fwd_t)*1000, 3),
            bwd_ms=round(avg(bwd_t)*1000, 3),
            step_ms=round(avg(step_t)*1000, 3),
            tokens_per_sec=round(BATCH_SIZE*SEQ_LEN/avg(step_t), 1),
            loss=round(loss.item(), 4)
        )

    out_file = f"results_no_tp_{args.model}.json"
    with open(out_file, "w") as fout:
        json.dump(results, fout, indent=2)

    print("\n" + "="*60)
    print(f"  BASELINE (No TP) - 1 GPU - {args.model.upper()}")
    print("="*60)
    for k in ["params_per_gpu","mem_model_mb","mem_peak_mb","fwd_ms","bwd_ms","step_ms","tokens_per_sec","loss"]:
        v = results[k]
        label = k.replace("_"," ").title()
        if isinstance(v, float):
            print(f"  {label:<25} {v:>12.2f}")
        else:
            print(f"  {label:<25} {v:>12,}")
    print("="*60)

if __name__ == "__main__":
    main()