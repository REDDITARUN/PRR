"""
Parallel Recursive Reasoner (PRR)

Core idea: Instead of one recursive trajectory (TRM/HRM), maintain K parallel
latent "thought branches" that independently recurse and softly communicate.
Decode only at the end; select the best branch via a learned scorer.

Architecture:
    1. Encode input -> input_emb [B, L, D]
    2. Initialize K branches with learned diverse inits + noise
    3. For T steps:
       a. Recurse: update each branch with shared tiny transformer core
       b. Score: predict confidence per branch
       c. Communicate: score-weighted soft averaging (every comm_interval steps)
    4. Decode all K branches at the end -> pick best by scorer

Training:
    - Best-of-K (soft minimum) cross-entropy loss
    - Scorer loss (predict which branch is best)
    - Diversity loss (prevent branch collapse)
    - Optional auxiliary grounding decode at intermediate steps
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin,
    CastedEmbedding, CastedLinear
)
from models.sparse_embedding import CastedSparseEmbedding


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PRRConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # Architecture
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 4
    num_layers: int = 2         # layers in the shared recurrent core

    # Parallel branches
    num_branches: int = 8       # K
    num_steps: int = 8          # T (total recursion steps)
    grad_steps: int = 2         # how many final steps carry gradients

    # Communication
    comm_alpha: float = 0.1     # pull strength toward weighted average
    comm_temp: float = 1.0      # temperature for score softmax
    comm_interval: int = 2      # communicate every N steps

    # Branch init noise
    init_noise_std: float = 0.1

    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16

    # Position encodings
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PRRBlock(nn.Module):
    """Single transformer block: self-attention + SwiGLU MLP, post-norm."""

    def __init__(self, config: PRRConfig):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm residual (same as TRM)
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class PRRRecurrentCore(nn.Module):
    """Shared recurrent core: input injection + stack of PRRBlocks."""

    def __init__(self, layers: List[PRRBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor, input_injection: torch.Tensor,
                cos_sin: CosSin) -> torch.Tensor:
        z = z + input_injection
        for layer in self.layers:
            z = layer(cos_sin=cos_sin, hidden_states=z)
        return z


class PRRScorer(nn.Module):
    """Lightweight head that predicts a scalar confidence for each branch.
    Uses the first token position (puzzle-emb slot) as a summary."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            CastedLinear(hidden_size, hidden_size // 2, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size // 2, 1, bias=True),
        )
        # Init to near-zero so early scores are uninformative
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.fill_(-5.0)

    def forward(self, z_first_token: torch.Tensor) -> torch.Tensor:
        """z_first_token: [N, D] -> scores [N]"""
        return self.head(z_first_token).squeeze(-1)


# ---------------------------------------------------------------------------
# Inner model (the actual forward logic)
# ---------------------------------------------------------------------------

class PRRInner(nn.Module):
    def __init__(self, config: PRRConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        K = config.num_branches
        D = config.hidden_size

        # I/O embeddings
        self.embed_scale = math.sqrt(D)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size, D, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(D, config.vocab_size, bias=False)

        # Puzzle embeddings (optional, zero-init)
        self.puzzle_emb_len = config.puzzle_emb_len if config.puzzle_emb_ndim > 0 else 0
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype,
            )

        # Position encodings
        total_seq = config.seq_len + self.puzzle_emb_len
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=D // config.num_heads,
                max_position_embeddings=total_seq,
                base=config.rope_theta,
            )

        # K learned branch initializations (each is a D-dim vector broadcast over seq)
        self.branch_inits = nn.ParameterList([
            nn.Parameter(trunc_normal_init_(torch.empty(D, dtype=self.forward_dtype), std=1.0))
            for _ in range(K)
        ])

        # Shared recurrent core
        self.core = PRRRecurrentCore([
            PRRBlock(config) for _ in range(config.num_layers)
        ])

        # Branch scorer
        self.scorer = PRRScorer(D)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """Encode tokenized input grid + puzzle embedding -> [B, L+P, D]"""
        embedding = self.embed_tokens(inputs.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        return self.embed_scale * embedding

    def _init_branches(self, B: int, device: torch.device) -> torch.Tensor:
        """Create [B, K, L, D] initial branch states with diversity."""
        K = self.config.num_branches
        L = self.config.seq_len + self.puzzle_emb_len
        D = self.config.hidden_size

        # Stack K learned init vectors, broadcast over batch and sequence
        z = torch.stack(
            [init.unsqueeze(0).unsqueeze(0).expand(B, L, -1) for init in self.branch_inits],
            dim=1,
        )  # [B, K, L, D]

        # Add noise for extra diversity during training
        if self.training and self.config.init_noise_std > 0:
            z = z + self.config.init_noise_std * torch.randn_like(z)

        return z

    def _communicate(self, z: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Soft communication: pull each branch toward score-weighted average.
        z: [B, K, L, D], scores: [B, K] -> updated z [B, K, L, D]"""
        weights = F.softmax(scores / self.config.comm_temp, dim=1)  # [B, K]
        weighted_avg = (weights[:, :, None, None] * z).sum(dim=1, keepdim=True)  # [B, 1, L, D]
        z = z + self.config.comm_alpha * (weighted_avg.expand_as(z) - z)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode branch states -> logits, removing puzzle-emb positions.
        z: [B, K, L, D] -> logits [B, K, seq_len, V]"""
        B, K, L, D = z.shape
        logits = self.lm_head(z.reshape(B * K, L, D))  # [B*K, L, V]
        logits = logits.view(B, K, L, -1)
        # Slice off puzzle embedding positions
        if self.puzzle_emb_len > 0:
            logits = logits[:, :, self.puzzle_emb_len:]
        return logits

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]):
        B = batch["inputs"].shape[0]
        K = self.config.num_branches
        T = self.config.num_steps
        grad_steps = self.config.grad_steps

        # 1. Encode input
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        # [B, L, D] where L = seq_len + puzzle_emb_len

        # 2. Init K branches
        z = self._init_branches(B, input_emb.device)
        # [B, K, L, D]

        # 3. Expand input for all branches
        input_expanded = input_emb.unsqueeze(1).expand(-1, K, -1, -1)
        # [B, K, L, D]

        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        no_grad_steps = T - grad_steps
        scores_history: List[torch.Tensor] = []

        # ---- No-grad exploration steps ----
        with torch.no_grad():
            for t in range(no_grad_steps):
                # Recurse
                z_flat = z.reshape(B * K, *z.shape[2:])
                inp_flat = input_expanded.reshape(B * K, *input_expanded.shape[2:])
                z_flat = self.core(z_flat, inp_flat, cos_sin=cos_sin)
                z = z_flat.reshape(B, K, *z.shape[2:])

                # Score
                scores = self.scorer(z[:, :, 0, :].reshape(B * K, -1)).reshape(B, K)
                scores_history.append(scores)

                # Communicate
                if (t + 1) % self.config.comm_interval == 0:
                    z = self._communicate(z, scores)

        # ---- Grad refinement steps ----
        for t_offset in range(grad_steps):
            t = no_grad_steps + t_offset

            # Recurse
            z_flat = z.reshape(B * K, *z.shape[2:])
            inp_flat = input_expanded.reshape(B * K, *input_expanded.shape[2:])
            z_flat = self.core(z_flat, inp_flat, cos_sin=cos_sin)
            z = z_flat.reshape(B, K, *z.shape[2:])

            # Score
            scores = self.scorer(z[:, :, 0, :].reshape(B * K, -1)).reshape(B, K)
            scores_history.append(scores)

            # Communicate (except last step)
            if (t + 1) % self.config.comm_interval == 0 and t < T - 1:
                z = self._communicate(z, scores)

        # 4. Final decode
        final_logits = self._decode(z)           # [B, K, seq_len, V]
        final_scores = scores_history[-1]         # [B, K]

        return final_logits, final_scores, scores_history, z


# ---------------------------------------------------------------------------
# Outer wrapper (matches TRM/HRM interface pattern)
# ---------------------------------------------------------------------------

class ParallelRecursiveReasoner(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = PRRConfig(**config_dict)
        self.inner = PRRInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.inner(batch)
