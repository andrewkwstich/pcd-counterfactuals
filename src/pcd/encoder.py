from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutput:
    soft_tokens: torch.Tensor
    topk_indices: torch.Tensor
    topk_values: torch.Tensor
    pre_acts: torch.Tensor


class TopKEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 4096,
        n_concepts: int = 32768,
        k: int = 16,
        aux_k: int = 500,
        aux_coef: float = 1e-4,
        dead_window_tokens: int = 1_000_000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_concepts = n_concepts
        self.k = k
        self.aux_k = aux_k
        self.aux_coef = aux_coef
        self.dead_window_tokens = dead_window_tokens

        self.W_enc = nn.Parameter(torch.empty(n_concepts, d_model, dtype=torch.float32))
        self.b_enc = nn.Parameter(torch.zeros(n_concepts, dtype=torch.float32))
        self.W_emb = nn.Parameter(torch.empty(d_model, n_concepts, dtype=torch.float32))

        self.register_buffer(
            "tokens_since_active",
            torch.zeros(n_concepts, dtype=torch.long),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        w = torch.randn(self.n_concepts, self.d_model, dtype=torch.float32)
        w = F.normalize(w, dim=1)
        with torch.no_grad():
            self.W_enc.copy_(w)
            self.W_emb.copy_(w.t())
            self.b_enc.zero_()
            self.tokens_since_active.zero_()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def encode(self, a: torch.Tensor, out_dtype: torch.dtype | None = None) -> EncoderOutput:
        lead_shape = a.shape[:-1]
        a_flat = a.reshape(-1, self.d_model).to(torch.float32)

        pre_acts = F.linear(a_flat, self.W_enc, self.b_enc)
        topk_values, topk_indices = pre_acts.topk(self.k, dim=-1)

        sparse = torch.zeros_like(pre_acts)
        sparse.scatter_(-1, topk_indices, topk_values)
        soft = F.linear(sparse, self.W_emb)

        dtype = out_dtype or a.dtype
        soft = soft.to(dtype)

        return EncoderOutput(
            soft_tokens=soft.reshape(*lead_shape, self.d_model),
            topk_indices=topk_indices.reshape(*lead_shape, self.k),
            topk_values=topk_values.reshape(*lead_shape, self.k),
            pre_acts=pre_acts,
        )

    # ------------------------------------------------------------------
    # auxiliary dead-concept loss
    # ------------------------------------------------------------------
    def aux_loss(self, pre_acts: torch.Tensor) -> torch.Tensor:
        dead_mask = self.tokens_since_active > self.dead_window_tokens
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return pre_acts.new_zeros(())
        dots = pre_acts - self.b_enc
        masked = dots.masked_fill(~dead_mask, float("-inf"))
        k_eff = min(self.aux_k, n_dead)
        topk_dead, _ = masked.topk(k_eff, dim=-1)
        return -(self.aux_coef / self.aux_k) * topk_dead.sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # activity tracking
    # ------------------------------------------------------------------
    @torch.no_grad()
    def batch_active_mask(self, topk_indices: torch.Tensor) -> torch.Tensor:
        active = torch.zeros_like(self.tokens_since_active, dtype=torch.bool)
        active[topk_indices.reshape(-1)] = True
        return active

    @torch.no_grad()
    def update_activity(self, active_mask: torch.Tensor, n_tokens: int) -> None:
        self.tokens_since_active += int(n_tokens)
        self.tokens_since_active[active_mask] = 0

    @torch.no_grad()
    def activity_stats(self) -> dict:
        dead = self.tokens_since_active > self.dead_window_tokens
        n_dead = int(dead.sum())
        return {
            "n_alive": self.n_concepts - n_dead,
            "n_dead": n_dead,
            "frac_alive": (self.n_concepts - n_dead) / self.n_concepts,
        }
