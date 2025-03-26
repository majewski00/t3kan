from dataclasses import dataclass, fields
from typing import Optional
import torch


@dataclass
class TTTInput:
    Q: torch.Tensor  # shape: (B, n_heads, SEQ / mini_batch_size, mini_batch_size, head_dim)
    K: torch.Tensor
    V: torch.Tensor

    W_states: Optional[torch.Tensor] = None
    b_states: Optional[torch.Tensor] = None
    W_grad: Optional[torch.Tensor] = None
    b_grad: Optional[torch.Tensor] = None
    # For MLP
    W1_states: Optional[torch.Tensor] = None
    b1_states: Optional[torch.Tensor] = None
    W1_grad: Optional[torch.Tensor] = None
    b1_grad: Optional[torch.Tensor] = None

    inner_loop_lr: Optional[torch.Tensor] = (
        None  # shape: (B, n_heads, SEQ / mini_batch_size, 1, mini_batch_size) (or reminder)
    )
    gradient_scale: Optional[torch.Tensor] = None  # shape: (B, n_heads, SEQ / mini_batch_size, mini_batch_size, 1)

    def update(self, **kwargs):
        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])

    def permute(self, *dims) -> None:
        self.Q = self.Q.permute(*dims)
        self.K = self.K.permute(*dims)
        self.V = self.V.permute(*dims)
        if self.inner_loop_lr is not None:
            self.inner_loop_lr = self.inner_loop_lr.permute(*dims)
        if self.gradient_scale is not None:
            self.gradient_scale = self.gradient_scale.permute(*dims)

        return None
