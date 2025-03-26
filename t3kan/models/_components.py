import torch
import torch.nn as nn
import torch.nn.functional as F

from t3kan.utils import ModelArgs


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor):
    _, _seq_len, _, _ = x.shape
    if _seq_len != freqs_complex.shape[0]:
        freqs_complex = freqs_complex[:_seq_len, :]

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    x_rot = x_complex * freqs_complex
    x_rot = torch.view_as_real(x_rot)
    x_rot = x_rot.reshape(*x.shape)

    return x_rot.type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"Dimensions for RoPE must be even! But got head_dim: {head_dim}")

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta_tensor = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta_tensor).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        n = self._norm(x.float())
        return self.weight.type_as(x) * n


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dropout = config.dropout
        intermediate_size = config.intermediate_size

        self.post_act_ln = RMSNorm(intermediate_size, eps=config.norm_eps) if config.ffn_post_act_ln else None

        self.ff1 = nn.Linear(config.dim, intermediate_size, bias=config.bias)
        self.ff2 = nn.Linear(config.dim, intermediate_size, bias=config.bias)
        self.ffo = nn.Linear(intermediate_size, config.dim, bias=config.bias)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.ff1(x))
        x = swish * self.ff2(x)
        if self.post_act_ln is not None:
            x = self.post_act_ln(x)

        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout)

        return self.ffo(x)
