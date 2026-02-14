"""
Loss heads for Parallel Recursive Reasoner.

BestOfKLossHead:
    - Best-of-K (soft minimum) cross-entropy: only the best branch matters
    - Scorer loss: train scorer to identify the best branch
    - Diversity loss: penalize cosine similarity between branch latents
    - Optional auxiliary grounding loss at intermediate decode steps

Also includes stablemax cross-entropy from TRM for numerical stability.
"""

from typing import Any, Tuple, Dict, Sequence, Optional, List

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


# ---------------------------------------------------------------------------
# Stablemax cross-entropy (from TRM)
# ---------------------------------------------------------------------------

def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


# ---------------------------------------------------------------------------
# BestOfK Loss Head
# ---------------------------------------------------------------------------

class BestOfKLossHead(nn.Module):
    """Wraps a ParallelRecursiveReasoner and computes the full training loss.

    Args:
        model: ParallelRecursiveReasoner instance
        loss_type: 'stablemax_cross_entropy' or 'softmax_cross_entropy'
        tau: temperature for best-of-K soft minimum (lower -> harder min)
        scorer_weight: weight on the scorer cross-entropy loss
        diversity_weight: weight on the branch diversity penalty
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "stablemax_cross_entropy",
        tau: float = 0.1,
        scorer_weight: float = 0.5,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.tau = tau
        self.scorer_weight = scorer_weight
        self.diversity_weight = diversity_weight

    @property
    def puzzle_emb(self):
        return self.model.puzzle_emb

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            loss: scalar
            metrics: dict of scalar metrics (detached)
            outputs: dict of requested outputs (detached)
        """
        # Forward the inner model
        final_logits, final_scores, scores_history, z = self.model(batch)
        # final_logits: [B, K, seq_len, V]
        # final_scores: [B, K]
        # z:            [B, K, L, D]   (full latent incl. puzzle-emb positions)

        labels = batch["labels"]  # [B, seq_len]
        B, K, L_out, V = final_logits.shape

        # Expand labels for all branches
        labels_exp = labels.unsqueeze(1).expand(-1, K, -1)      # [B, K, seq_len]
        mask = (labels_exp != IGNORE_LABEL_ID)                   # [B, K, seq_len]

        # ------------------------------------------------------------------
        # 1. Per-branch cross-entropy
        # ------------------------------------------------------------------
        per_token_loss = self.loss_fn(
            final_logits.reshape(B * K, L_out, V),
            labels_exp.reshape(B * K, L_out),
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=mask.reshape(B * K, L_out),
        ).reshape(B, K, L_out)   # [B, K, seq_len]

        loss_counts = mask.float().sum(-1).clamp(min=1)          # [B, K]
        per_branch_loss = per_token_loss.sum(-1) / loss_counts   # [B, K]

        # ------------------------------------------------------------------
        # 2. Best-of-K loss  (soft minimum via log-sum-exp)
        #    L = -tau * log( sum_k exp(-l_k / tau) )
        #    As tau->0 this becomes min(l_k)
        # ------------------------------------------------------------------
        best_of_k = -self.tau * torch.logsumexp(-per_branch_loss / self.tau, dim=1)  # [B]
        main_loss = best_of_k.sum()

        # ------------------------------------------------------------------
        # 3. Scorer loss  (cross-entropy: predict which branch is best)
        # ------------------------------------------------------------------
        with torch.no_grad():
            best_branch_idx = per_branch_loss.argmin(dim=1)  # [B]
        scorer_loss = F.cross_entropy(
            final_scores.to(torch.float32), best_branch_idx, reduction="sum"
        )

        # ------------------------------------------------------------------
        # 4. Diversity loss  (penalize high cosine similarity between branches)
        # ------------------------------------------------------------------
        # Pool each branch over sequence -> summary vector [B, K, D]
        z_summary = z.mean(dim=2)
        z_norm = F.normalize(z_summary.float(), dim=-1)
        # Pairwise cosine similarity [B, K, K]
        sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))
        # Mask diagonal
        diag_mask = ~torch.eye(K, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)
        # Average off-diagonal similarity per sample -> penalize
        diversity_loss = sim_matrix[diag_mask.expand(B, -1, -1)].view(B, K * (K - 1)).mean(dim=1).sum()

        # ------------------------------------------------------------------
        # Total loss
        # ------------------------------------------------------------------
        total_loss = (
            main_loss
            + self.scorer_weight * scorer_loss
            + self.diversity_weight * diversity_loss
        )

        # ------------------------------------------------------------------
        # Metrics (all detached)
        # ------------------------------------------------------------------
        with torch.no_grad():
            preds_all = final_logits.argmax(dim=-1)                    # [B, K, seq_len]
            # Best branch (oracle)
            best_preds = preds_all[torch.arange(B, device=preds_all.device), best_branch_idx]      # [B, seq_len]
            # Scorer-selected branch
            scorer_branch_idx = final_scores.argmax(dim=1)             # [B]
            scorer_preds = preds_all[torch.arange(B, device=preds_all.device), scorer_branch_idx]   # [B, seq_len]

            mask_single = (labels != IGNORE_LABEL_ID)                  # [B, seq_len]
            count_single = mask_single.float().sum(-1).clamp(min=1)    # [B]

            # Oracle accuracy (best branch)
            oracle_correct = mask_single & (best_preds == labels)
            oracle_exact = oracle_correct.sum(-1) == mask_single.sum(-1)

            # Scorer accuracy
            scorer_correct = mask_single & (scorer_preds == labels)
            scorer_exact = scorer_correct.sum(-1) == mask_single.sum(-1)

            # Scorer selection accuracy (does scorer pick the same branch as oracle?)
            scorer_picks_best = (scorer_branch_idx == best_branch_idx).float().sum()

            valid = (count_single > 0)

            metrics = {
                "count":                   valid.float().sum(),
                "main_loss":               main_loss.detach(),
                "scorer_loss":             scorer_loss.detach(),
                "diversity_loss":          diversity_loss.detach(),
                "oracle_accuracy":         torch.where(valid, (oracle_correct.float().sum(-1) / count_single), 0.0).sum(),
                "oracle_exact_accuracy":   (valid & oracle_exact).float().sum(),
                "scorer_accuracy":         torch.where(valid, (scorer_correct.float().sum(-1) / count_single), 0.0).sum(),
                "scorer_exact_accuracy":   (valid & scorer_exact).float().sum(),
                "scorer_picks_best":       scorer_picks_best,
                "mean_branch_diversity":   (1.0 - diversity_loss / max(B, 1)).detach(),
            }

        # ------------------------------------------------------------------
        # Outputs for evaluator (compatible with ARC evaluator)
        # ------------------------------------------------------------------
        outputs: Dict[str, torch.Tensor] = {}
        if "preds" in return_keys:
            outputs["preds"] = scorer_preds.detach()
        if "q_halt_logits" in return_keys:
            # Use scorer confidence of selected branch as "halt logits"
            outputs["q_halt_logits"] = final_scores[
                torch.arange(B, device=final_scores.device), scorer_branch_idx
            ].detach()

        return total_loss, metrics, outputs
