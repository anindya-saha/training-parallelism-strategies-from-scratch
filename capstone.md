# Capstone: From Zero to 5D Parallel Training

This document is a narrative walkthrough of building `capstone/src/train.py` --
a single-file GPT-2 124M pretraining script that starts with pure data-parallel
(DDP) and will progressively add TP, PP, CP, and EP.

Everything lives in one file during development. We refactor into modules once
the full pipeline is working.

---

## Step 0: Infrastructure

### Environment

```bash
source env.dev
```

### Docker image

The capstone image is built from `capstone/containers/` -- a DLC-style PyTorch
image (CUDA, EFA, Amazon Linux 2023). Code is NOT baked in; mount via FSx at
runtime.

```bash
cd ~/training-parallelism-strategies-from-scratch/capstone/

source dev.env
source docker/pytorch/versions-cuda.env

DOCKER_BUILDKIT=1 docker build \
  -f docker/pytorch/Dockerfile.cuda \
  --target runtime \
  -t "${REGISTRY}/${USER}/dist-train/capstone:$(git rev-parse HEAD)" \
  .
```

Image naming convention follows the experiments pattern:
`${REGISTRY}/${USER}/dist-train/capstone:<git-commit-hash>`.

### Syncing code to FSx

```bash
rsync -av ~/training-parallelism-strategies-from-scratch/capstone/src/ \
  /mnt/fsx-static/asaha/training-parallelism-strategies-from-scratch/capstone/src/ \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__'
```

`rsync` is one-shot (source -> destination only). Run it before each job submission.

### Submitting a job

```bash
# Smoke test: run train.py on 1 p5en node (8x H200). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name dist-train \
    --node-type p5en \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 300"

# Smoke test: run train.py on 1 p4d node (8x A100). Skip rebuild, reuse last image.
./scripts/run_hpto_job.sh --skip-build \
    --job-name dist-train \
    --node-type p4d \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 100"

./scripts/run_hpto_job.sh \
    --job-name capstone-ddp \
    --node-type p5en \
    --image-uri mlp.docker.zooxlabs.com/asaha/dist-train/capstone:b8f199f7766d43374c124c509ddfb13ae59ae404 \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 300"

./scripts/run_hpto_job.sh --skip-build \
    --job-name dist-train \
    --image-uri mlp.docker.zooxlabs.com/asaha/dist-train/capstone:01e0f729e544b9d7019a2ed0e7c692bf33cef946 \
    --node-type p5en \
    --train-script /mnt/fsx/asaha/training-parallelism-strategies-from-scratch/capstone/src/train.py \
    --train-args "--data_root /mnt/fsx/asaha/parallelism-experiments/data/edu_fineweb10B --max_steps 300"

    
```

`--image-uri` implies `--skip-build`. Add `--dry-run` to preview the rendered YAML.

---

## Step 1: Data Preparation (`prepare.py`)

**Run once before training.** Downloads FineWeb-Edu 10B tokens, tokenizes with
GPT-2 BPE (`tiktoken`), and writes fixed-size `.npy` shards.

### Tokenizer

```python
_enc = tiktoken.get_encoding("gpt2")       # vocab size 50257
_eot = _enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    tokens = [_eot]                         # document separator
    tokens.extend(_enc.encode_ordinary(doc["text"]))
    return np.array(tokens).astype(np.uint16)
```

Each document is prepended with `<|endoftext|>`. Token ids fit in uint16.

### Shard layout

Shard 0 goes to `val/`, the rest to `train/`. 100M tokens per shard, ~100 shards
for 10B tokens. Documents span shard boundaries (continuous stream, not
document-aligned).

```
data_root/
  val/edufineweb_000000.npy
  train/edufineweb_000001.npy
  train/edufineweb_000002.npy
  ...
```

Subdirectories are created upfront:

```python
os.makedirs(os.path.join(data_cache_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(data_cache_dir, "val"), exist_ok=True)
```

### Parallelized tokenization

`multiprocessing.Pool` with `cpu_count // 2` workers. Each worker loads its own
`tiktoken` instance in `_pool_init_worker` (safe for both fork and spawn).

---

## Step 2: Configuration Dataclasses

Three dataclasses define everything. Defaults match GPT-2 124M and Karpathy's
nanogpt hyperparameters.

```python
@dataclass
class ModelConfig:
    block_size: int = 1024       # max sequence length
    vocab_size: int = 50304      # padded for efficiency (divisible by 128)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

@dataclass
class ParallelConfig:
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_size: int = -1            # auto-computed

    def validate(self, world_size):
        model_parallel = self.tp_size * self.pp_size * self.cp_size * self.ep_size
        self.dp_size = world_size // model_parallel

@dataclass
class TrainConfig:
    total_batch_size: int = 524288   # ~0.5M tokens
    micro_batch_size: int = 64
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 715
    max_steps: int = 19073           # ~1 epoch for 10B tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
```

CLI args map to dataclass fields automatically via `dataclass_from_args`:

```python
def dataclass_from_args(cls, args):
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in vars(args).items() if k in field_names})
```

Fields not present in `args` (like `vocab_size`, `min_lr`) keep their defaults.

---

## Step 3: Distributed Setup and 5D DeviceMesh

### NCCL initialization

`torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`. We detect these, call
`init_process_group(backend="nccl")`, and pin each rank to its GPU:

```python
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    device_str = f"cuda:{local_rank}"
    torch.cuda.set_device(device_str)
```

### Pure DDP vs. N-D mesh

When all parallelism sizes are 1, we skip the DeviceMesh and use simple DDP
(every rank is a DP replica). Otherwise we build the 5D mesh:

```python
mesh_shape = (dp, pp, cp, ep, tp)     # outer to inner
mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=("dp","pp","cp","ep","tp"))
```

TP is the innermost (fastest-changing) dimension so adjacent ranks share a TP
group -- they should be on the same node where NVLink provides maximum bandwidth.

**Example: 8 GPUs, TP=2, PP=2, DP=2**

```
mesh_shape = (2, 2, 1, 1, 2)

         TP group
         |---|
rank 0   [GPU0, GPU1]  -- PP stage 0 -- DP replica 0
rank 2   [GPU2, GPU3]  -- PP stage 1
rank 4   [GPU4, GPU5]  -- PP stage 0 -- DP replica 1
rank 6   [GPU6, GPU7]  -- PP stage 1
```

Process groups are extracted per dimension. Size-1 dimensions get `None` (no
communication needed):

```python
for name, size in zip(mesh_dim_names, mesh_shape):
    groups[name] = mesh.get_group(name) if size > 1 else None
```

**Why DeviceMesh over manual `new_group`?** Declarative (specify shape, not rank
lists), composable (FSDP2, `parallelize_module` accept it directly), less
error-prone (no manual rank arithmetic).

---

## Step 4: Data Loading (`FinewebTokenIterableDataset`)

A PyTorch `IterableDataset` that reads pre-tokenized `.npy` shards. Each call
to `__iter__` yields `(x, y)` of shape `(B, T)`.

Key design: ranks stride by `dp_world_size` so data-parallel replicas see
disjoint data:

```python
pos = B * T * dp_rank               # each rank starts at a different offset
# ...
pos += B * T * dp_world              # stride by total DP width
```

Wrapped in a standard `DataLoader` with `batch_size=None` (the dataset itself
yields pre-formed batches) and `num_workers=0`:

```python
train_loader = build_fineweb_dataloader(
    micro_batch_size=B, block_size=T, split="train",
    dp_rank=dp_rank, dp_world_size=dp_world, ...
)
```

---

## Step 5: GPT-2 124M Model

Four classes, all driven by `ModelConfig`:

**CausalSelfAttention** -- fused QKV projection, `scaled_dot_product_attention`:

```python
self.c_attn = nn.Linear(n_embd, 3 * n_embd)    # fused Q, K, V
self.c_proj = nn.Linear(n_embd, n_embd)         # output projection
self.c_proj.NANOGPT_SCALE_INIT = 1              # tag for scaled init
```

**MLP** -- `c_fc` -> GELU -> `c_proj`:

```python
self.c_fc   = nn.Linear(n_embd, 4 * n_embd)
self.c_proj = nn.Linear(4 * n_embd, n_embd)
self.c_proj.NANOGPT_SCALE_INIT = 1
```

**Block** -- pre-norm transformer block (LayerNorm before attention/MLP):

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

**GPT** -- full model with weight tying and custom init:

```python
self.transformer = nn.ModuleDict(dict(
    wte=nn.Embedding(vocab_size, n_embd),
    wpe=nn.Embedding(block_size, n_embd),
    h=nn.ModuleList([Block(config) for _ in range(n_layer)]),
    ln_f=nn.LayerNorm(n_embd),
))
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight   # weight tying
self.apply(self._init_weights)                       # visitor pattern
```

### Weight tying

The input embedding (`wte`) and output head (`lm_head`) share the **same tensor**.
For GPT-2 124M, `wte` is `50304 x 768 = ~38.6M` params -- 31% of the model.
Without tying, you'd double that. Both layers are approximate inverses of each
other (token -> embedding vs. embedding -> logit), so sharing is a natural
inductive bias. (Press & Wolf, 2017)

### Residual stream scaling (NANOGPT_SCALE_INIT)

Each block adds to the residual stream twice (attention + MLP). After 12 layers
that is 24 additions. If each has the same variance, the residual stream variance
is ~24x at init -- unstable.

The fix: tag `c_proj` layers with `NANOGPT_SCALE_INIT`, then in `_init_weights`
scale their init std by `1 / sqrt(2 * n_layer)`:

```python
if hasattr(module, "NANOGPT_SCALE_INIT"):
    std *= (2 * self.config.n_layer) ** -0.5    # 0.02 * 0.204 ~= 0.004
```

Only the two output projections per block get this treatment. Everything else
uses the baseline `std=0.02`. (Radford et al., 2019)

### Optimizer (`configure_optimizers`)

Weight-decay / no-decay split: 2D+ parameters (weight matrices) get decay,
1D parameters (biases, LayerNorm) don't. Uses fused AdamW when available on CUDA:

```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), fused=True)
```

---

## Step 6: DDP Training Loop

### Gradient accumulation

The effective batch size is fixed at `total_batch_size` (524288 tokens, ~0.5M).
With micro-batches of `B * T` tokens per rank, gradient accumulation steps are
computed as:

```python
grad_accum_steps = total_batch_size // (B * T * dp_world)
```

### Why `loss = loss / grad_accum_steps`

`optimizer.zero_grad()` is called **once**, before the inner loop. After that,
every `loss.backward()` call **adds** to the `.grad` tensors on every parameter
-- they are never reset between micro-steps. The sequence looks like:

```
optimizer.zero_grad()             # all param.grad = 0

micro_step 0: loss.backward()  -> param.grad += d(loss_0)/d(param)
micro_step 1: loss.backward()  -> param.grad += d(loss_1)/d(param)
...
micro_step 7: loss.backward()  -> param.grad += d(loss_7)/d(param)

optimizer.step()                  # updates params using accumulated .grad
```

Without the division, `param.grad` after the loop is the **sum** of 8 gradients
-- 8x too large compared to a single forward/backward on the full batch. By
dividing the scalar `loss` before `.backward()`, autograd produces proportionally
smaller gradients. After all micro-steps the accumulated `.grad` equals the
**mean**, which is mathematically correct:

- **Without division**: `grad = g0 + g1 + ... + g7` (sum, 8x too large)
- **With division**: `grad = g0/8 + g1/8 + ... + g7/8 = mean(g0..g7)` (correct)

The `loss` variable itself is reinitialized every iteration of the inner loop
(fresh scalar from `model(x, y)`). The division is not about accumulating `loss`
-- it's about scaling the gradient that flows into the persistent `.grad` tensors.

### DDP grad sync control

During gradient accumulation, all-reduce is expensive and redundant for
intermediate micro-steps. DDP lets you skip it:

```python
for micro_step in range(grad_accum_steps):
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    if ddp and dp_world > 1:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()
```

`require_backward_grad_sync = False` on intermediate steps means gradients
accumulate locally. Only the final micro-step triggers the NCCL all-reduce.

### Loss averaging across ranks

After the accumulation loop, the logged loss is all-reduced for consistent
reporting:

```python
if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
```

### LR schedule

Cosine decay with linear warmup:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

Applied manually per step by updating `param_group["lr"]` -- no PyTorch scheduler
object needed.

### Mixed precision

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps both forward
passes (training and validation). bfloat16 has the same exponent range as fp32
so no loss scaling is needed (unlike fp16).

### Validation

Every `eval_interval` steps, runs 20 micro-batches on the val shard with
`torch.no_grad()` and all-reduces the result:

```python
if step % t_cfg.eval_interval == 0 or last_step:
    model.eval()
    with torch.no_grad():
        val_loss_accum = torch.tensor(0.0, device=device)
        for _ in range(20):
            x, y = next(val_iter)
            ...
    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
```

### Logging

Each step logs loss, LR, grad norm, wall time, and throughput (tokens/sec).
Written to both the Python logger and `log/log.txt` for post-hoc analysis:

```
step     0 | loss 10.955667 | lr 8.39e-07 | norm 3.1792 | dt 1543.21ms | tok/s 339788
step    10 | loss 9.384215  | lr 8.39e-06 | norm 2.0451 | dt 287.55ms  | tok/s 1823215
```

---

## Step 7: Model wrapping and cleanup

### DDP wrapping

```python
model = GPT(m_cfg)
model.to(device)
raw_model = model

if ddp and dp_world > 1:
    model = DDP(model, device_ids=[local_rank], process_group=groups["dp"])
    raw_model = model.module
```

`raw_model` is the unwrapped GPT -- used for optimizer creation and grad
clipping (DDP wrapper doesn't expose `.parameters()` in the same way).

In pure-DDP mode, `groups["dp"]` is `None`, which makes DDP use the default
(world) process group. When using the 5D mesh with other parallelism dimensions,
`groups["dp"]` is the DP-specific subgroup.

### Cleanup

```python
if ddp:
    destroy_process_group()
```

---

## What's next

The script currently supports DDP-only training. The infrastructure is ready
for N-D parallelism -- the DeviceMesh, process groups, and `ParallelConfig`
are all in place. Next steps:

1. **Tensor Parallelism** -- replace `c_attn`, `c_proj`, `c_fc` with
   column/row-parallel linear layers using the TP process group.
2. **Context Parallelism** -- add ring attention dispatch in
   `CausalSelfAttention.forward` using the CP process group.
3. **Pipeline Parallelism** -- split `transformer.h` into stages and add
   a 1F1B schedule using the PP process group.
4. **Expert Parallelism** -- replace selected MLP blocks with MoE layers
   using the EP process group.
5. **Checkpointing** -- distributed save/load of sharded model state.
6. **Evaluation** -- HellaSwag accuracy tracking.
