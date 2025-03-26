from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import * TODO: use absolute


class Attention(nn.Module):
    def __init__(self, config: TransformerArgs):
        super().__init__()
        self.config = config
        self.q_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.dim = config.dim

        self.kv_dim = self.dim // self.q_heads * self.kv_heads
        self.head_dim = config.dim // self.q_heads

        self.wq = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.wk = nn.Linear(self.dim, self.kv_dim, bias=config.bias)
        self.wv = nn.Linear(self.dim, self.kv_dim, bias=config.bias)
        self.wo = nn.Linear(self.kv_dim, self.dim, bias=config.bias)

    def _gqa_dot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
    ):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if scale is None:
            scale = query.size(-1) ** 0.5
        query = query / scale

        _bsz, q_head, _seq, _dim = query.shape
        _, k_head, _, _ = key.shape
        _, v_head, _, _ = value.shape

        n_head_groups = q_head // k_head
        if n_head_groups > 1:
            ## Grouped Query Attention
            query = query.view(_bsz, n_head_groups, k_head, _seq, _dim)
            scores = torch.einsum("bghnd,bhsd->bhns", query, key)
        else:
            scores = torch.matmul(query, key.transpose(2, 3))

        if causal:
            mask = torch.ones(
                (_bsz, _seq, _seq),
                device=query.device,
                dtype=torch.bool,
            ).tril_()

        if mask is not None:
            if mask.ndim == 2:
                mask.unsqueeze_(1).unsqueeze_(1)
            elif mask.ndim == 3:
                mask.unsqueeze_(1)

            scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores / scale, dim=-1)
        if self.config.dropout > 0.0 and self.training:
            attention = F.dropout(attention, p=self.config.dropout)

        out = torch.matmul(attention, value).transpose(1, 2)

        return out

    def forward(
        self,
        x: torch.Tensor,
        freq_complex: torch.Tensor,
        causal: bool = False,
    ):
        _bsz, _seq_len, _ = x.shape

        # (B, seq_len, q_heads * dim)
        q: torch.Tensor = self.wq(x)
        # (B, seq_len, dim * kv_dim)
        ## kv_dim = dim // q_heads * kv_heads
        k: torch.Tensor = self.wk(x)
        v: torch.Tensor = self.wv(x)

        q = q.view(_bsz, _seq_len, self.q_heads, self.head_dim)
        k = k.view(_bsz, _seq_len, self.kv_heads, self.head_dim)
        v = v.view(_bsz, _seq_len, self.kv_heads, self.head_dim)

        k = apply_rope(k, freq_complex)
        q = apply_rope(q, freq_complex)

        y = self._gqa_dot(query=q, key=k, value=v, causal=causal).type_as(x)

        return self.wo(y.reshape(_bsz, _seq_len, self.kv_dim))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerArgs):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config)
        self.norm1 = RMSNorm(config.dim, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, causal: bool = False):
        n = self.norm1(x)
        x = x + self.attn(n, freqs_complex, causal)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerArgs):
        logger.debug(f"Initializing Transformer model.")
        super().__init__()
        self.config = config
        pos_freqs = precompute_theta_pos_frequencies(config.dim // config.n_heads, config.max_seq_len * 2)
        self.register_buffer("pos_freqs", pos_freqs)

        self.emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_id)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])
        self.output_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.bias)
        logger.debug("Transformer model created.")

    def __str__(self):
        base_str = super().__str__()
        parms = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{base_str}\nNumber of parameters: {parms:,}  "

    def forward(self, x: torch.Tensor, causal: bool = False):
        x = self.emb(x)
        for block in self.blocks:
            x = block(x, self.pos_freqs, causal)

        x = self.output_norm(x)
        x = self.output(x)
        return x
