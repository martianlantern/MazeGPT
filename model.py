# model.py
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root-mean-square norm
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        # reshape for heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, H, T, Hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out(y)
        y = self.dropout(y)
        return y

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        d_ff: int = 4 * 256,
        max_seq: int = 1024,
        dropout: float = 0.1,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute a causal mask for max_seq
        mask = torch.tril(torch.ones(max_seq, max_seq, dtype=torch.uint8))
        # (1, 1, T, T) broadcastable across batch and heads
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq, max_seq), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) -> logits: (B, T, vocab)
        """
        B, T = idx.shape
        if T > self.max_seq:
            raise ValueError(f"Sequence length {T} exceeds model max_seq {self.max_seq}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        attn_mask = self.causal_mask[:, :, :T, :T]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Autoregressive generation. idx: (B, T). Returns (B, T+max_new_tokens or earlier if EOS hit).
        """
        self.eval()
        B = idx.size(0)
        for _ in range(max_new_tokens):
            T = idx.size(1)
            if T > self.max_seq:
                # Trim left if we ever exceed; shouldn't happen with our setting but keep safe
                idx = idx[:, -self.max_seq:]
                T = idx.size(1)
            logits = self.forward(idx)[:, -1, :]  # (B, vocab)
            if temperature != 1.0:
                logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)

            if top_k and top_k > 0:
                # Zero out everything not in top_k
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                new_probs = torch.zeros_like(probs)
                new_probs.scatter_(1, topk_idx, topk_vals)
                probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-8)

            next_tok = torch.multinomial(probs, num_samples=1) if temperature > 0 else torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)

            # Early break if all sequences ended with EOS (id 3)
            if (next_tok.squeeze(1) == 3).all():
                break
        return idx
