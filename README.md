# Parallel Recursive Reasoner (PRR)

A **parallel multi-hypothesis recursive reasoning** model for ARC-AGI, inspired by TRM and HRM but with a fundamentally different approach: instead of one recursive trajectory, PRR maintains **K parallel latent thought branches** that independently refine their hypotheses and softly communicate.

## Core Idea

ARC puzzles often have early ambiguity — multiple plausible rules could explain the training examples. A single-trajectory model (TRM/HRM) commits to one interpretation and refines it. If the initial guess is wrong, it's stuck.

PRR keeps **multiple interpretations alive** simultaneously:

1. **K parallel branches** (default: 8) each maintain a latent state
2. Each step, every branch independently recurses through a **shared tiny transformer core**
3. A **learned scorer** predicts which branch is most promising
4. Branches **softly communicate** via score-weighted averaging (good ideas spread, bad ideas fade)
5. At the end, **decode all branches** and select the best by scorer confidence

Training uses **Best-of-K** loss: only the best-performing branch's error matters. This naturally encourages diversity — branches explore different solution strategies.

## Architecture

```
Input: ARC grid x
  │
  ▼
Encode → input_emb [B, L, D]
  │
  ▼
Init K branches with learned diverse vectors + noise
  z^(1)...z^(K)  [B, K, L, D]
  │
  ├── For T steps:
  │     ├── Recurse: z^(k) = Core(z^(k), input_emb)   [shared weights]
  │     ├── Score:   s^(k) = Scorer(z^(k))
  │     └── Communicate: soft pull toward weighted average
  │
  ▼
Decode all branches → logits [B, K, L, V]
Select best by scorer → final prediction
```

**Key design choices:**
- **Latent-only**: branches stay in latent space until the final decode (cheap)
- **Shared weights**: the same tiny core processes all branches at all times (parameter efficient)
- **No-grad exploration**: first 6 of 8 steps run without gradients (memory efficient)
- **Soft communication**: differentiable, unlike hard resampling/killing

## Comparison with TRM

| | TRM | PRR |
|---|---|---|
| Trajectories | 1 | K (default: 8) |
| Recursion | H_cycles × L_cycles nested loops | T flat steps |
| ACT halting | Yes (Q-learning) | No (fixed T steps) |
| Communication | z_H ↔ z_L hierarchy | K branches ↔ score-weighted avg |
| Loss | Per-token CE | Best-of-K soft-min CE |
| Parameters | ~7M (512-dim) | ~3M (256-dim) |
| Compute/sample | ~13 core passes × 16 ACT steps | 8 steps × K branches (batched) |

## Requirements

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
wandb login YOUR-LOGIN
```

## Dataset Preparation

Uses the **same data format as TRM/HRM**. If you already built the data for TRM, just point `data_paths` to it.

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

## Training

### ARC-AGI-1 (single GPU)

```bash
run_name="prr_arc1_1gpu"
python pretrain.py \
  arch=prr \
  data_paths="[data/arc1concept-aug-1000]" \
  global_batch_size=128 \
  +run_name=${run_name} ema=True
```

### ARC-AGI-1 (4x H100 GPUs)

```bash
run_name="prr_arc1_4gpu"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=prr \
  data_paths="[data/arc1concept-aug-1000]" \
  +run_name=${run_name} ema=True
```

### ARC-AGI-2 (4x H100 GPUs)

```bash
run_name="prr_arc2_4gpu"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=prr \
  data_paths="[data/arc2concept-aug-1000]" \
  +run_name=${run_name} ema=True
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_branches` | 8 | Number of parallel thought branches (K) |
| `num_steps` | 8 | Total recursion steps (T) |
| `grad_steps` | 2 | Steps with gradients (last N) |
| `hidden_size` | 256 | Hidden dimension |
| `num_layers` | 2 | Transformer layers in recurrent core |
| `comm_alpha` | 0.1 | Communication pull strength |
| `comm_interval` | 2 | Communicate every N steps |
| `tau` | 0.1 | Best-of-K temperature |
| `scorer_weight` | 0.5 | Scorer loss weight |
| `diversity_weight` | 0.1 | Diversity penalty weight |

**Scaling tips:**
- Increase `hidden_size` to 512 if you have 4+ H100s
- Decrease `global_batch_size` if OOM (effective batch is `batch × K`)
- Set `grad_steps=4` for better gradient signal (costs more memory)
- Increase `comm_alpha` if branches collapse; decrease if diversity is low

## Training Loss Components

1. **Best-of-K CE** (main): `-τ · log(Σ_k exp(-loss_k / τ))` — focuses on the best branch
2. **Scorer CE**: trains the scorer to predict which branch will be correct
3. **Diversity penalty**: penalizes high cosine similarity between branch latent states

## Metrics to Watch

- `oracle_exact_accuracy`: best branch is correct (upper bound on performance)
- `scorer_exact_accuracy`: scorer-selected branch is correct (actual performance)
- `scorer_picks_best`: how often scorer picks the oracle-best branch
- `diversity_loss`: should stay moderate (not too high, not zero)

If `oracle_exact_accuracy` is high but `scorer_exact_accuracy` is low, the model is finding solutions but the scorer can't identify them — increase `scorer_weight`.

If `oracle_exact_accuracy` is low, branches aren't diverse enough or the core is too small — increase `diversity_weight`, `hidden_size`, or `num_steps`.

## Credits

Architecture inspired by:
- [TRM: Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [HRM: Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)

Data pipeline and evaluation adapted from the TRM codebase.
