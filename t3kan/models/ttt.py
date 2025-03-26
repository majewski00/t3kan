from abc import abstractmethod
from typing import Callable, Dict, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from t3kan.models._components import FeedForward, RMSNorm, apply_rope, precompute_theta_pos_frequencies
from t3kan.models._types import TTTInput
from t3kan.utils import TTTArgs


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )
    return z


class TTTBase(nn.Module):
    def __init__(self, config: TTTArgs):
        super().__init__()
        self.config = config
        self.mini_batch_size = config.mini_batch_size
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dim = config.dim

        self.ttt_bias = config.ttt_bias

        # Initialize projections
        self.q = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.v = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        self.norm_eps = config.norm_eps
        self.post_norm = nn.LayerNorm(self.dim, eps=self.norm_eps)

        # Initialize inner loop learnable learning rate
        self.base_lr = config.ttt_base_lr
        self.inner_loop_lr_W = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=(1, self.dim)) for _ in range(self.n_heads)], dim=0)
        )
        self.inner_loop_lr_b = nn.Parameter(torch.zeros(self.n_heads, 1))

        # Scale factors for the inner loop W gradients (lower the magnitude for the latest gradients in mini-batch)
        gradient_scale = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("gradient_scale", gradient_scale, persistent=False)
        self.learnable_gradient_scale = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_ttt_ln()

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.n_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.n_heads, 1)))

    @abstractmethod
    def compute_mini_batch(self, params: TTTInput, i: int, mini_batch_size: int): ...

    def get_learning_rate(self, x: torch.Tensor, mini_batch_size: int) -> Dict[str, torch.Tensor]:
        """

        Args:
            x: Input tensor of shape (B, mini_batch_size * num_mini_batches, dim)
            mini_batch_size: This is the size of the mini-batch. Can be either the mini-batch size or the reminder of the sequence length.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the inner loop learning rate and the gradient. Keys as in the TTTInput class.
        """
        B, SEQ = x.shape[:2]
        num_mini_batches = SEQ // mini_batch_size
        x = x.reshape(B, num_mini_batches, mini_batch_size, -1)
        inner_lr = torch.einsum("bnkd,hod->bhnko", x, self.inner_loop_lr_W) + self.inner_loop_lr_b.reshape(
            1, -1, 1, 1, 1
        )
        # inner_lr shape = (B, n_heads, num_mini_batches, mini_batch_size, 1)

        inner_lr = F.sigmoid(inner_lr)
        inner_lr = self.base_lr * inner_lr.permute(0, 1, 2, 4, 3) / self.head_dim
        # inner_lr Shape: (B, n_heads, num_mini_batches, 1, mini_batch_size)

        gradient_scale = torch.clamp_min(self.gradient_scale + self.learnable_gradient_scale, 0.0)[:mini_batch_size]
        gradient_scale = torch.broadcast_to(
            gradient_scale.reshape(1, 1, 1, mini_batch_size, 1),
            (B, self.n_heads, num_mini_batches, mini_batch_size, 1),
        )

        return {"inner_loop_lr": inner_lr, "gradient_scale": gradient_scale}

    def ttt(self, x: TTTInput):
        x.permute(2, 0, 1, 3, 4)

        mini_batch_size = x.Q.shape[3]
        num_mini_batches = x.Q.shape[0]
        B = x.Q.shape[1]

        if self.config.ttt_type in ["linear", "mlp"] and x.W_states is None:
            x.W_states = self.W0[None, ...].repeat(B, 1, 1, 1)
            x.W_grad = torch.zeros_like(x.W_states)
            x.b_states = self.b0[None, ...].repeat(B, 1, 1, 1)
            if self.ttt_bias:
                x.b_grad = torch.zeros_like(x.b_states)
        if self.config.ttt_type == "mlp" and x.W1_states is None:
            x.W1_states = self.W1[None, ...].repeat(B, 1, 1, 1)
            x.W1_grad = torch.zeros_like(x.W1_states)
            x.b1_states = self.b1[None, ...].repeat(B, 1, 1, 1)
            if self.ttt_bias:
                x.b1_grad = torch.zeros_like(x.b1_states)
        if self.config.ttt_type == "kan" and x.W_states is None:
            x.W_states = torch.broadcast_to(self.coefficients, (B, *self.coefficients.shape))
            x.W_grad = torch.zeros_like(x.W_states)

        # Allocate memory
        output = torch.empty_like(x.Q)

        for i in range(num_mini_batches):
            y = self.compute_mini_batch(x, i, mini_batch_size)
            output[i] = y

        # output shape: (num_mini_batches, B, n_heads, mini_batch_size, head_dim)
        return output

    def forward(self, x: torch.Tensor, pos_frequencies: torch.Tensor):
        B, SEQ = x.shape[:2]
        mini_batch_reminder = SEQ % self.mini_batch_size
        num_mini_batches = SEQ // self.mini_batch_size

        Q, K, V = self.q(x), self.k(x), self.v(x)  # TODO: No share_qk

        Q = Q.view(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)

        K = apply_rope(K, pos_frequencies)
        Q = apply_rope(Q, pos_frequencies)
        V = apply_rope(V, pos_frequencies)

        output = torch.empty((B, SEQ, self.dim), device=x.device, dtype=x.dtype)

        inputs: TTTInput
        if num_mini_batches > 0:
            inputs = TTTInput(
                Q=Q[:, :, : num_mini_batches * self.mini_batch_size].reshape(
                    B, self.n_heads, num_mini_batches, self.mini_batch_size, self.head_dim
                ),
                K=K[:, :, : num_mini_batches * self.mini_batch_size].reshape(
                    B, self.n_heads, num_mini_batches, self.mini_batch_size, self.head_dim
                ),
                V=V[:, :, : num_mini_batches * self.mini_batch_size].reshape(
                    B, self.n_heads, num_mini_batches, self.mini_batch_size, self.head_dim
                ),
                **self.get_learning_rate(x[:, : num_mini_batches * self.mini_batch_size], self.mini_batch_size),
            )
            # hidden_states shape: (num_mini_batches, B, n_heads, mini_batch_size, head_dim)
            hidden_states = self.ttt(inputs)
            hidden_states = hidden_states.permute(1, 0, 3, 2, 4).reshape(B, -1, self.dim)
            output[:, : num_mini_batches * self.mini_batch_size] = hidden_states

        if mini_batch_reminder > 0:
            if num_mini_batches > 0:
                # keep gradients for the last mini-batch
                inputs.update(
                    Q=Q[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    K=K[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    V=V[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    **self.get_learning_rate(x[:, -mini_batch_reminder:], mini_batch_reminder),
                )
            else:
                inputs = TTTInput(
                    Q=Q[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    K=K[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    V=V[:, :, -mini_batch_reminder:].reshape(B, self.n_heads, 1, mini_batch_reminder, self.head_dim),
                    **self.get_learning_rate(x[:, -mini_batch_reminder:], mini_batch_reminder),
                )
            # hidden_states shape: (1, B, n_heads, mini_batch_reminder, head_dim)
            hidden_states = self.ttt(inputs)
            hidden_states = hidden_states.permute(1, 0, 3, 2, 4).reshape(B, mini_batch_reminder, self.dim)
            output[:, -mini_batch_reminder:] = hidden_states

        output = self.post_norm(output)
        output = self.o(output)
        return output


class T3Linear(TTTBase):
    def __init__(self, config: TTTArgs):
        super().__init__(config)
        self.W0 = nn.Parameter(
            torch.normal(0, 0.02, size=(config.n_heads, self.head_dim, self.head_dim)), requires_grad=True
        )
        self.b0 = nn.Parameter(torch.zeros(config.n_heads, 1, self.head_dim), requires_grad=True)

    def compute_mini_batch(self, params: TTTInput, i: int, mini_batch_size: int):
        W = params.W_states  # corresponding to self.W0 = (16, 8, 16, 16)
        b = params.b_states

        K = params.K[i]
        V = params.V[i]
        Q = params.Q[i]
        inner_loop_lr = params.inner_loop_lr[i]
        gradient_scale = params.gradient_scale[i]

        ln_weight = self.ttt_norm_weight.reshape(self.n_heads, 1, self.head_dim)
        ln_bias = self.ttt_norm_bias.reshape(self.n_heads, 1, self.head_dim)

        # Linear operation for TTT linear
        Z0 = K @ W + b
        target = V - K
        grad_l_wrt_Z0 = ln_fused_l2_bwd(Z0, target, ln_weight, ln_bias, eps=self.norm_eps)

        # ! Inefficient way to calculate the gradient
        # Z0 = K + ln_fwd(Z0, ln_weight, ln_bias, eps=self.norm_eps)
        # inner_loss = ((Z0 - V) ** 2).mean()
        # grad_l_wrt_Z0 = torch.autograd.grad(inner_loss, Z0, create_graph=True, retain_graph=True)[0]

        # shape: (B, n_heads, mini_batch_size, mini_batch_size)
        inner_loop_lr = torch.tril(inner_loop_lr.repeat(1, 1, mini_batch_size, 1))

        # "bhki,bhij->bhkij" outer product (i == j = head_dim)
        grad_W = torch.einsum("bhki,bhkj->bhkij", K, grad_l_wrt_Z0)
        # n == k
        grad_W = torch.einsum("bhnk,bhkij->bhnij", inner_loop_lr, grad_W)

        # collect cumulative gradients for updates
        grad_W += params.W_grad.unsqueeze(2)
        # save the Wb gradients for the next mini-batch
        params.W_grad = grad_W[:, :, -1]

        # apply gradient scaling and update the states
        W = W.unsqueeze(2) - gradient_scale.unsqueeze(-1) * grad_W
        params.W_states = W[:, :, -1]

        # if self.ttt_bias:
        grad_b = torch.einsum("bhkn,bhki->bhni", inner_loop_lr, grad_l_wrt_Z0)
        grad_b += params.b_grad
        params.b_grad = grad_b[:, :, -1:]
        b = b - gradient_scale * grad_b
        params.b_states = b[:, :, -1:]

        Z1 = (Q.unsqueeze(3) @ W).squeeze(3) + b
        # shape: (B, n_heads, num_mini_batches, mini_batch_size, head_dim) ! probably wrong
        Z1 = Q + ln_fwd(Z1, ln_weight, ln_bias, eps=self.norm_eps)

        return Z1


class T3MLP(TTTBase):
    def __init__(self, config: TTTArgs):
        super().__init__(config)
        self.W0 = nn.Parameter(
            torch.normal(0, 0.02, size=(config.n_heads, self.head_dim, self.head_dim)), requires_grad=True
        )
        self.b0 = nn.Parameter(torch.zeros(config.n_heads, 1, self.head_dim), requires_grad=True)
        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(config.n_heads, self.head_dim, self.head_dim)), requires_grad=True
        )
        self.b1 = nn.Parameter(torch.zeros(config.n_heads, 1, self.head_dim), requires_grad=True)

        self.activation = F.silu if config.ttt_mlp_activation == "silu" else F.tanh
        self.activation_bwd = lambda x: (
            F.silu(x) * (1 + x * (1 - F.silu(x)))
            if config.ttt_mlp_activation == "silu"
            else lambda x: 1 - F.tanh(x) ** 2
        )

    def compute_mini_batch(self, params: TTTInput, i: int, mini_batch_size: int):
        W0 = params.W_states
        b0 = params.b_states
        W1 = params.W1_states
        b1 = params.b1_states

        K = params.K[i]
        V = params.V[i]
        Q = params.Q[i]
        inner_loop_lr = params.inner_loop_lr[i]
        gradient_scale = params.gradient_scale[i]

        ln_weight = self.ttt_norm_weight.reshape(self.n_heads, 1, self.head_dim)
        ln_bias = self.ttt_norm_bias.reshape(self.n_heads, 1, self.head_dim)

        # Linear operation for TTT linear
        Z0 = K @ W0 + b0
        A1 = self.activation(Z0)
        Z1 = A1 @ W1 + b1
        target = V - K
        grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, target, ln_weight, ln_bias, eps=self.norm_eps)
        grad_l_wrt_Z0 = grad_l_wrt_Z1 @ W1.transpose(-2, -1) * self.activation_bwd(Z0)

        # shape: (B, n_heads, mini_batch_size, mini_batch_size)
        inner_loop_lr = torch.tril(inner_loop_lr.repeat(1, 1, mini_batch_size, 1))

        grad_W1 = torch.einsum("bhki,bhkj->bhkij", A1, grad_l_wrt_Z1)
        grad_W1 = torch.einsum("bhnk,bhkij->bhnij", inner_loop_lr, grad_W1)
        grad_W1 += params.W1_grad.unsqueeze(2)
        grad_b1 = torch.einsum("bhkn,bhki->bhni", inner_loop_lr, grad_l_wrt_Z1)
        grad_b1 += params.b_grad

        grad_W0 = torch.einsum("bhki,bhkj->bhkij", K, grad_l_wrt_Z0)
        grad_W0 = torch.einsum("bhnk,bhkij->bhnij", inner_loop_lr, grad_W0)
        grad_W0 += params.W_grad.unsqueeze(2)
        grad_b0 = torch.einsum("bhkn,bhki->bhni", inner_loop_lr, grad_l_wrt_Z0)
        grad_b0 += params.b_grad

        W0 = W0.unsqueeze(2) - gradient_scale.unsqueeze(-1) * grad_W0
        b0 = b0 - gradient_scale * grad_b0
        W1 = W1.unsqueeze(2) - gradient_scale.unsqueeze(-1) * grad_W1
        b1 = b1 - gradient_scale * grad_b1

        params.W_states = W0[:, :, -1]
        params.b_states = b0[:, :, -1:]
        params.W1_states = W1[:, :, -1]
        params.b1_states = b1[:, :, -1:]
        params.W_grad = grad_W0[:, :, -1]
        params.b_grad = grad_b0[:, :, -1:]
        params.W1_grad = grad_W1[:, :, -1]
        params.b1_grad = grad_b1[:, :, -1:]

        Z = (Q.unsqueeze(3) @ W0).squeeze(3) + b0
        Z = self.activation(Z)
        Z = (Z.unsqueeze(3) @ W1).squeeze(3) + b1
        Z = Q + ln_fwd(Z, ln_weight, ln_bias, eps=self.norm_eps)
        return Z


class T3KAN(TTTBase):
    def __init__(self, config: TTTArgs):
        super().__init__(config)
        self.grid_range = config.kan_grid_range
        self.degree = config.kan_degree
        self.control_points = config.kan_control_points

        self.grid = nn.Parameter(
            torch.linspace(self.grid_range[0], self.grid_range[1], self.control_points + 1 + 2 * self.degree)[
                None, :
            ].repeat(self.head_dim, 1),
            requires_grad=config.kan_learnable_grid,
        )
        self.base = torch.nn.SiLU()
        self._init_coefficients(config.kan_init_method)

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data[None, :], (self.n_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data[None, :], (self.n_heads, 1)))

    def _init_coefficients(self, init_function: Union[Literal["sin", "sigmoid", "silu", "noise"], Callable] = "noise"):
        if isinstance(init_function, Callable):
            base = init_function
        elif init_function == "silu":
            base = lambda x: x / (1 + torch.exp(-(x * 4))) / 4
        elif init_function == "sigmoid":
            base = lambda x: 1 / (1 + torch.exp(-x * 4))
        elif init_function == "sin":
            base = lambda x: torch.sin(10 * x) / 10
        elif init_function == "noise":
            base = lambda x: (torch.randn_like(x) - 1 / 2) * 0.5 / self.control_points
        else:
            raise ValueError("Invalid init function. Choose from 'sin', 'sigmoid', 'silu' or Callable function.")

        x = torch.linspace(self.grid_range[0], self.grid_range[1], self.n_heads * self.head_dim**2).reshape(
            self.n_heads, self.head_dim, self.head_dim
        )
        basis = self.basis_function(x, self.grid[0, :], self.degree)
        G = base(x).unsqueeze(-1)
        coefficients = (basis.pinverse() @ G).squeeze(-1)
        # self.coefficients = nn.Parameter(coefficients[None, :, None, None].repeat(self.n_heads, 1, self.head_dim, self.head_dim))
        self.coefficients = nn.Parameter(coefficients.transpose(1, 2), requires_grad=True)

    def basis_function(self, x: torch.Tensor, grid: torch.Tensor, degree: int):
        if degree == 0:
            value = (x.unsqueeze(-1) >= grid[None, ..., :-1]) * (x.unsqueeze(-1) < grid[None, ..., 1:]).float()
        else:
            basis = self.basis_function(x, grid, degree - 1)

            x = x.unsqueeze(-1)
            grid = grid.unsqueeze(0)
            value_1 = (x - grid[..., : -(degree + 1)]) / (grid[..., degree:-1] - grid[..., : -(degree + 1)])
            value_2 = (grid[..., degree + 1 :] - x) / (grid[..., degree + 1 :] - grid[..., 1:-degree])

            value = value_1 * basis[..., :-1] + value_2 * basis[..., 1:]

        return torch.nan_to_num(value)

    def compute_mini_batch(self, params: TTTInput, i: int, mini_batch_size: int):
        coef = params.W_states  # shape: batch, n_heads, control_points+degree, dim, (dim)
        K = params.K[i]  # batch, n_heads, mini_batch, dim
        V = params.V[i]
        Q = params.Q[i]
        inner_loop_lr = params.inner_loop_lr[i]
        gradient_scale = params.gradient_scale[i]

        ln_weight = self.ttt_norm_weight.reshape(self.n_heads, 1, self.head_dim)
        ln_bias = self.ttt_norm_bias.reshape(self.n_heads, 1, self.head_dim)

        # Linear operation for T3 KAN
        # shape: (B, n_heads, mini_batch_size, head_dim, G)
        basis = self.basis_function(K, self.grid, self.degree)
        # Z0 = basis @ coef.unsqueeze(2)  # + b // "bhkig,bhgj->bhkij" ==> Batch, n_heads, mini_batch, dim, dim ! I don't like how it looks...
        Z0 = torch.einsum("bhkig,bhgj->bhkij", basis, coef)  # Batch, n_heads, mini_batch, dim, dim
        residual = self.base(K)  # // case2.
        Z0 = torch.einsum("bhki,bhkij->bhki", residual, Z0)  # batch, n_heads, mini_batch, dim # // case2.

        target = V  # - K  # self.base(K)  # (B, n_heads, mini_batch, dim)

        grad_l_wrt_Z0 = ln_fused_l2_bwd(
            Z0, target, ln_weight, ln_bias, eps=self.norm_eps
        )  # // case2. Now in this case shape: (B, nh, k, dim)

        # ? should I use target as in every other case? Or with residual?

        # grad_l_wrt_Z0 = ln_fused_l2_bwd(Z0, target.unsqueeze(-2), ln_weight, ln_bias, eps=self.norm_eps)
        # grad_l_wrt_Z0 = ln_fwd(basis, ln_weight, ln_bias, eps=self.norm_eps)
        # # // without layer norm
        # grad_l_wrt_Z0 = 2 * (Z0 - target.unsqueeze(-2))

        inner_loop_lr = torch.tril(inner_loop_lr.repeat(1, 1, mini_batch_size, 1))
        # grad_C = torch.einsum("bhkig,bhkji->bhkgj", basis, grad_l_wrt_Z0)
        grad_C = torch.einsum("bhkig,bhki->bhkig", basis, grad_l_wrt_Z0)  # // case2.
        # grad_C = (grad_l_wrt_Z0 @ basis).transpose(-2, -1)
        # grad_C = torch.einsum("bhnk,bhkgj->bhngj", inner_loop_lr, grad_C)  # batch, n_heads, mini_batch, G, dim
        grad_C = torch.einsum("bhnk,bhkig->bhnig", inner_loop_lr, grad_C).transpose(3, 4)  # // case2.

        grad_C += params.W_grad.unsqueeze(2)
        params.W_grad = grad_C[:, :, -1]

        coef = coef.unsqueeze(2) - gradient_scale.unsqueeze(-1) * grad_C
        params.W_states = coef[:, :, -1]

        # // b

        Z1 = self.basis_function(Q, self.grid, self.degree) @ coef  # // + b
        residual = self.base(Q)  # // case2.
        Z1 = torch.einsum("bhki,bhkij->bhki", residual, Z1)  # batch, n_heads, mini_batch, dim # // case2.
        Z1 = ln_fwd(Z1, ln_weight, ln_bias, eps=self.norm_eps)
        # Z1 = self.base(Q) + Z1.sum(-2)
        # Z1 = torch.einsum("bhkij,bhki->bhki", Z1, self.base(Q))  # ? or should there be "j" in the product?
        # Z1 = torch.einsum("bhkij,bhki->bhki", Z1, Q)  # ? or should there be "j" in the product?

        return Z1


class TTTBlock(nn.Module):
    def __init__(self, config: TTTArgs):  # ? layer_idx
        super().__init__()
        if config.ttt_type == "linear":
            self.ttt = T3Linear(config)
        elif config.ttt_type == "mlp":
            self.ttt = T3MLP(config)
        elif config.ttt_type == "kan":
            self.ttt = T3KAN(config)

        self.ffn = FeedForward(config)
        self.input_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, pos_frequencies: torch.Tensor):
        residual = x
        x = self.input_norm(x)
        x = self.ttt(x, pos_frequencies) + residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + residual
        return x


class T3(nn.Module):
    def __init__(self, config: TTTArgs):
        super().__init__()
        type(self).__name__ = f"T3{config.ttt_type.capitalize()}"
        self.config = config
        pos_freqs = precompute_theta_pos_frequencies(config.dim // config.n_heads, config.max_seq_len * 2)  # FIXME ?
        self.register_buffer("pos_freqs", pos_freqs, persistent=False)

        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([TTTBlock(config) for _ in range(config.depth)])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output_projection = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def __str__(self):
        base_str = super().__str__()
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{base_str}\nNumber of parameters: {params:,}  "

    def _init_weights(self):
        generator = self.config.generator
        std = self.config.initializer_range

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std, generator=generator)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std, generator=generator)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, x: torch.Tensor):
        hidden_states = self.emb(x)

        for block in self.blocks:
            hidden_states = block(hidden_states, self.pos_freqs)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_projection(hidden_states)
        return hidden_states


if __name__ == "__main__":
    config = TTTArgs(
        depth=4,
        dim=128,
        vocab_size=128,
        max_seq_len=32,
        ttt_bias=True,
    )
    model = T3(config)
    x = torch.randint(0, 128, (20, 74))  # torch.randint(0, 128, (20, 30))

    with torch.no_grad():
        output = model(x)
    print(output.shape)
