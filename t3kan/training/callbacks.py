import glob
import os
import random
import uuid
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, Tuple
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from t3kan.utils import TrainingOutput


def set_seed(seed: int = 0, deterministic: bool = True) -> torch.Generator:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def generate_training_uuid(force: Optional[str | int] = None) -> str:
    if force:
        return str(force)

    return str(uuid.uuid4()).split("-")[0]


class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 0.0, verbose_fn: Callable = print, *args, **kwargs):
        """
        Args:
           patience (int): How long to wait after last time validation loss improved. Default: 8
           delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
           verbose_fn: (Callable) Function to call when early stopping is triggered.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose_fn

        self.counter = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter == self.patience // 2:
                self.verbose("EarlyStopping detected weak improvement in validation loss... ")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter -= 1

        return False


class OptimizerWithCyclicLR:
    """
    Combines an optimizer and a cyclic learning rate scheduler into a single class with a unified `.step()` method.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use.
        base_lr (float | list[float]): Initial learning rate.
        max_lr (float | list[float]): Maximum learning rate.
        epoch_steps (int): Number of training steps in each epoch.
        step_size_up (int): Number of training iterations in the increasing half of a cycle.
        step_size_down (int | None): Number of training iterations in the decreasing half of a cycle. Default is None.
        mode (Literal["triangular", "triangular2", "exp_range"]): The mode for the learning rate schedule. Default is "triangular".
        cycle_momentum (bool): If True, momentum is cycled inversely to learning rate. Default is False.
        gamma (float): Multiplicative factor for learning rate decay.
        gamma_half_life (int | None): Half-life of the learning rate decay, represented in epochs.
    """

    name = "OptimizerWithCyclicLR"

    def __init__(
        self,
        optimizer: Any,
        base_lr: float | list[float],
        max_lr: float | list[float],
        epoch_steps: int,
        step_size_up: int | None = None,
        step_size_down: int | None = None,
        mode: Literal["triangular", "triangular2", "exp_range"] = "triangular",
        cycle_momentum: bool = False,
        gamma: float = 1.0,
        gamma_half_life: int | None = None,
        initial_visualize: bool = False,
    ) -> None:
        self._optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.epoch_steps = epoch_steps
        self.mode = mode
        self.cycle_momentum = cycle_momentum
        self.gamma = gamma
        if gamma_half_life and mode == "exp_range":
            self.calculate_cyclic_lr_gamma(gamma_half_life)
        self.initial_visualize = initial_visualize

        if step_size_up is None:
            if mode == "exp_range" and gamma_half_life:
                step_size_up = epoch_steps * gamma_half_life
            else:
                step_size_up = epoch_steps * 2

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up

        self.metadata = {
            "optimizer": optimizer.__name__,
            "base_lr": base_lr,
            "max_lr": max_lr,
            "step_size_up": step_size_up,
            "step_size_down": step_size_down,
            "mode": mode,
            "cycle_momentum": cycle_momentum,
            "gamma": gamma,
            "lr_records": [],
        }
        self._steps = 0

        filterwarnings(
            "ignore",
            message="Seems like `optimizer.step()`*",
        )

    def __call__(self, *args, **kwargs) -> "OptimizerWithCyclicLR":
        """
        Initializes the optimizer and scheduler with the provided parameters.
        """

        self.optimizer = self._optimizer(*args, **kwargs)

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            step_size_up=self.step_size_up,
            step_size_down=self.step_size_down,
            mode=self.mode,
            gamma=self.gamma,
            cycle_momentum=self.cycle_momentum,
        )
        self.metadata["lr_records"] = []

        if self.initial_visualize:
            self.initial_visualize = False
            self.visualize_lr(self.step_size_up * 6)

        return self

    def step(self) -> None:
        """
        Perform both optimizer step and scheduler step.
        """
        self.optimizer.step()
        self.scheduler.step()
        self._steps += 1
        if self._steps % self.epoch_steps == 0:
            self.metadata["lr_records"].append(self.scheduler.get_last_lr()[0])

    def zero_grad(self) -> None:
        """
        Reset gradients for all parameters in the optimizer.
        """
        self.optimizer.zero_grad()

    def get_lr(self) -> list[float]:
        """
        Returns the current learning rate.
        """
        return self.scheduler.get_last_lr()

    def calculate_cyclic_lr_gamma(self, half_life: int) -> None:
        total_steps = half_life * self.epoch_steps
        gamma = 0.5 ** (1 / total_steps)
        self.gamma = gamma

    def visualize_lr(self, n_steps=100) -> None:
        """
        Visualize the learning rate over `n_steps`.

        Args:
            n_steps: Number of steps to visualize.
        """
        if not hasattr(self, "scheduler"):
            raise ValueError("Optimizer has not been initialized. Please call the optimizer first.")

        scheduler = deepcopy(self.scheduler)
        lrs = []
        for step in range(n_steps):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        plt.plot(range(n_steps), lrs, label="Learning Rate")
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Cyclic Learning Rate Schedule")
        plt.grid(True, axis="y")

        # Add dotted vertical lines at every self.epoch_steps
        # for i in range(0, n_steps, self.epoch_steps * 2):
        #     plt.axvline(x=i, color="orange", linestyle="--", linewidth=0.5)
        # plt.axvline(x=0, color="orange", linestyle="--", linewidth=0.5, label="Epoch Boundary")

        plt.legend()
        plt.show()


def visualize_training(
    include: Tuple[str, ...], *args: TrainingOutput, save_path: Optional[str] = None, file_name: Optional[str] = None
) -> None:
    n = len(args)
    features = len(include)
    mapping = {
        "train_loss": {
            "label": "Train Loss",
            "title": "Training Loss for {model}",
            "xlabel": "Epoch",
            "ylabel": "Loss",
        },
        "validation_loss": {
            "label": "Validation Loss",
            "title": "Validation Loss for {model}",
            "xlabel": "Epoch",
            "ylabel": "Loss",
        },
        "duration": {
            "label": "Duration",
            "title": "Training Duration for {model}",
            "xlabel": "Epoch",
            "ylabel": "Duration [s]",
        },
    }
    assert all(key in mapping.keys() for key in include), "Not all keys are present in the TrainingOutput."

    fig, axes = plt.subplots(n, features, figsize=(features * 5, n * 5))
    for i, output in enumerate(args):
        for j, key in enumerate(include):
            ax = axes[i, j] if n > 1 else axes[j]
            data = getattr(output, key)
            ax.plot(data, label=mapping[key]["label"])
            ax.set_title(mapping[key]["title"].format(model=output.model))
            ax.set_xlabel(mapping[key]["xlabel"])
            ax.set_ylabel(mapping[key]["ylabel"])
            ax.legend()

    if file_name:
        title = f"Training Visualization for {file_name}"
    elif n == 1:
        file_name = f"{args[0].model.lower()}_training"
        title = f"Training Visualization for {args[0].model}"
    else:
        file_name = f"training_{args[0].training_uuid}"
        title = f"Training Visualization for {','.join([output.model for output in args])} Models"

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(pad=1.25)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        existing_images = glob.glob(os.path.join(save_path, f"{file_name}*.png"))
        if existing_images:
            if len(existing_images) == 1:
                file_path = os.path.join(save_path, f"{file_name}_1.png")
            else:
                image_subscript = max(int(f.split("_")[-1].split(".")[0]) for f in existing_images[1:]) + 1
                file_path = os.path.join(save_path, f"{file_name}_{image_subscript}.png")
        else:
            file_path = os.path.join(save_path, f"{file_name}.png")

        plt.savefig(file_path)
    else:
        plt.show()
