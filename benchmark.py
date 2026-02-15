#!/usr/bin/env python3
"""
Standalone ARC-AGI benchmark script for Parallel Recursive Reasoner (PRR).

Loads a trained checkpoint and evaluates it on the ARC test set,
reporting pass@1, pass@2, pass@5 accuracy, and saving a submission.json.

Usage (single GPU):
    python benchmark.py \
        --checkpoint checkpoints/Arc1concept-aug-1000-PRR/prr_v1_h100/step_23484 \
        --data-dir data/arc1concept-aug-1000

Usage (with custom config):
    python benchmark.py \
        --checkpoint checkpoints/.../step_23484 \
        --data-dir data/arc1concept-aug-1000 \
        --batch-size 16 \
        --num-branches 8 \
        --num-steps 8 \
        --output-dir results/

All flags:
    --checkpoint      Path to the saved state_dict file (required)
    --data-dir        Path to the preprocessed ARC dataset dir (required)
    --config          Path to the all_config.yaml (auto-detected from checkpoint dir)
    --batch-size      Evaluation batch size (default: 8)
    --num-branches    Override K branches at inference (default: from config)
    --num-steps       Override T recursion steps at inference (default: from config)
    --output-dir      Where to write submission.json + results (default: results/)
    --device          cuda or cpu (default: cuda)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

# Ensure project root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.prr import ParallelRecursiveReasoner, PRRConfig
from models.losses import BestOfKLossHead
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.arc import ARC, _crop
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PRR on ARC-AGI")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved state_dict (e.g. step_23484)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to preprocessed ARC dataset (e.g. data/arc1concept-aug-1000)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to all_config.yaml (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size")
    parser.add_argument("--num-branches", type=int, default=None,
                        help="Override number of branches K (default: from config)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override number of recursion steps T (default: from config)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load config from checkpoint directory
# ---------------------------------------------------------------------------

def load_config(args) -> dict:
    """Load the all_config.yaml from the checkpoint directory."""
    if args.config is not None:
        config_path = args.config
    else:
        # Auto-detect: look in the same dir as the checkpoint file
        ckpt_dir = Path(args.checkpoint).parent
        config_path = ckpt_dir / "all_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find all_config.yaml at {config_path}. "
                f"Please specify --config explicitly."
            )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Build model from config + checkpoint
# ---------------------------------------------------------------------------

def build_model(config: dict, args, metadata) -> BestOfKLossHead:
    """Reconstruct the model and load weights."""
    arch_cfg = config["arch"]

    # Build model config dict
    model_cfg = {
        k: v for k, v in arch_cfg.items()
        if k not in ("name", "loss")
    }
    model_cfg["batch_size"] = args.batch_size
    model_cfg["vocab_size"] = metadata.vocab_size
    model_cfg["seq_len"] = metadata.seq_len
    model_cfg["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers

    # Apply overrides
    if args.num_branches is not None:
        model_cfg["num_branches"] = args.num_branches
    if args.num_steps is not None:
        model_cfg["num_steps"] = args.num_steps

    # Create model
    with torch.device(args.device):
        model = ParallelRecursiveReasoner(model_cfg)
        loss_cfg = arch_cfg.get("loss", {})
        loss_head = BestOfKLossHead(
            model,
            loss_type=loss_cfg.get("loss_type", "stablemax_cross_entropy"),
            tau=loss_cfg.get("tau", 0.1),
            scorer_weight=loss_cfg.get("scorer_weight", 0.5),
            diversity_weight=loss_cfg.get("diversity_weight", 0.1),
        )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=args.device)

    # Strip _orig_mod. prefix added by torch.compile
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[new_key] = v
    state_dict = cleaned_state_dict

    # Handle puzzle embedding shape mismatch
    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in state_dict:
        expected_shape = loss_head.model.puzzle_emb.weights.shape
        if state_dict[puzzle_emb_name].shape != expected_shape:
            print(f"  Resizing puzzle embeddings: {state_dict[puzzle_emb_name].shape} -> {expected_shape}")
            state_dict[puzzle_emb_name] = (
                torch.mean(state_dict[puzzle_emb_name], dim=0, keepdim=True)
                .expand(expected_shape).contiguous()
            )

    loss_head.load_state_dict(state_dict, assign=True)
    loss_head.eval()
    print(f"  Model loaded ({sum(p.numel() for p in loss_head.parameters()):,} params)")
    return loss_head


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model: BestOfKLossHead, dataloader, metadata, args):
    """Run evaluation on the test split and collect predictions."""
    device = args.device
    K = model.model.inner.config.num_branches

    # Metrics accumulators
    total_main_loss = 0.0
    total_oracle_acc = 0.0
    total_scorer_acc = 0.0
    total_oracle_exact = 0.0
    total_scorer_exact = 0.0
    total_scorer_picks_best = 0.0
    total_count = 0
    num_batches = 0

    # For ARC evaluator
    all_preds = []  # list of (batch, preds_dict)

    for set_name, batch, global_batch_size in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["inputs"].shape[0]

        # Forward pass
        loss, metrics, preds = model(batch=batch, return_keys={"preds", "q_halt_logits"})

        # Accumulate metrics
        count = metrics["count"].item()
        total_count += count
        total_main_loss += metrics["main_loss"].item()
        total_oracle_acc += metrics["oracle_accuracy"].item()
        total_scorer_acc += metrics["scorer_accuracy"].item()
        total_oracle_exact += metrics["oracle_exact_accuracy"].item()
        total_scorer_exact += metrics["scorer_exact_accuracy"].item()
        total_scorer_picks_best += metrics["scorer_picks_best"].item()
        num_batches += 1

        all_preds.append((batch, preds))

        if num_batches % 10 == 0:
            avg_loss = total_main_loss / total_count if total_count > 0 else 0
            avg_oracle = total_oracle_acc / total_count if total_count > 0 else 0
            avg_scorer = total_scorer_acc / total_count if total_count > 0 else 0
            print(f"  Batch {num_batches}: loss={avg_loss:.4f} oracle_acc={avg_oracle:.3f} scorer_acc={avg_scorer:.3f}")

    # Compute averages
    if total_count > 0:
        results = {
            "main_loss": total_main_loss / total_count,
            "oracle_token_accuracy": total_oracle_acc / total_count,
            "scorer_token_accuracy": total_scorer_acc / total_count,
            "oracle_exact_accuracy": total_oracle_exact / total_count,
            "scorer_exact_accuracy": total_scorer_exact / total_count,
            "scorer_picks_best_rate": total_scorer_picks_best / total_count,
            "total_examples": total_count,
            "num_batches": num_batches,
        }
    else:
        results = {"error": "No valid examples found"}

    return results, all_preds


def run_arc_evaluator(all_preds, data_dir, metadata, output_dir):
    """Run the ARC evaluator to compute pass@K metrics."""
    try:
        # Look for test split data
        test_data_path = os.path.join(data_dir, "test")
        if not os.path.exists(test_data_path):
            print(f"  No test split found at {test_data_path}, skipping ARC evaluator")
            return {}

        evaluator = ARC(
            data_path=data_dir,
            eval_metadata=metadata,
            submission_K=2,
            pass_Ks=(1, 2, 5, 10),
            aggregated_voting=True,
        )
        evaluator.begin_eval()

        for batch, preds in all_preds:
            evaluator.update_batch(batch, preds)

        # ARC evaluator needs dist, mock single-process
        import torch.distributed as dist
        if not dist.is_initialized():
            # Use a simple gather that works without dist
            return _arc_result_no_dist(evaluator, output_dir)

        result = evaluator.result(save_path=output_dir, rank=0, world_size=1)
        return result or {}

    except Exception as e:
        print(f"  ARC evaluator error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _arc_result_no_dist(evaluator, save_path):
    """Compute ARC results without torch.distributed (single GPU)."""
    submission = {}
    correct = [0.0 for _ in range(len(evaluator.pass_Ks))]

    for name, puzzle in evaluator.test_puzzles.items():
        submission[name] = []
        num_test_correct = [0 for _ in range(len(evaluator.pass_Ks))]

        for pair in puzzle["test"]:
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))

            p_map = {}
            # Single process: just use local data
            for h, q in evaluator._local_preds.get(name, {}).get(input_hash, []):
                p_map.setdefault(h, [0, 0])
                p_map[h][0] += 1
                p_map[h][1] += q

            if not len(p_map):
                continue

            for h, stats in p_map.items():
                stats[1] /= stats[0]
            p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

            for i, k in enumerate(evaluator.pass_Ks):
                ok = False
                for h, stats in p_map[:k]:
                    ok |= h == label_hash
                num_test_correct[i] += ok

            pred_grids = []
            for h, stats in p_map[:evaluator.submission_K]:
                if h in evaluator._local_hmap:
                    pred_grids.append(evaluator._local_hmap[h])
            while len(pred_grids) < evaluator.submission_K:
                if pred_grids:
                    pred_grids.append(pred_grids[0])
                else:
                    pred_grids.append(np.zeros((1, 1), dtype=np.uint8))
            submission[name].append({
                f"attempt_{i + 1}": grid.tolist()
                for i, grid in enumerate(pred_grids)
            })

        for i in range(len(evaluator.pass_Ks)):
            if len(puzzle["test"]) > 0:
                correct[i] += num_test_correct[i] / len(puzzle["test"])

    total_puzzles = len(evaluator.test_puzzles)
    results = {
        f"ARC/pass@{k}": correct[i] / total_puzzles if total_puzzles > 0 else 0
        for i, k in enumerate(evaluator.pass_Ks)
    }
    results["ARC/total_puzzles"] = total_puzzles

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "submission.json"), "w") as f:
            json.dump(submission, f, indent=2)
        print(f"  Submission saved to {os.path.join(save_path, 'submission.json')}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  PRR Benchmark - ARC-AGI Evaluation")
    print("=" * 60)

    # Load config
    config = load_config(args)
    print(f"\nConfig loaded from checkpoint directory")
    print(f"  Architecture: {config['arch']['name']}")

    # Load test dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[args.data_dir],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )
    metadata = test_dataset.metadata
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Seq length: {metadata.seq_len}")
    print(f"  Test groups: {metadata.total_groups}")
    print(f"  Puzzles: {metadata.num_puzzle_identifiers}")

    # Build model
    print(f"\nBuilding model...")
    model = build_model(config, args, metadata)
    K = model.model.inner.config.num_branches
    T = model.model.inner.config.num_steps
    print(f"  Branches (K): {K}")
    print(f"  Steps (T):    {T}")

    # Evaluate
    print(f"\nRunning evaluation...")
    start_time = time.time()
    results, all_preds = evaluate(model, test_loader, metadata, args)
    eval_time = time.time() - start_time

    # Run ARC evaluator for pass@K
    print(f"\nRunning ARC evaluator...")
    arc_results = run_arc_evaluator(all_preds, args.data_dir, metadata, args.output_dir)
    results.update(arc_results)
    results["eval_time_seconds"] = round(eval_time, 1)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:35s} {v:.4f}")
        else:
            print(f"  {k:35s} {v}")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
