import datetime
import glob
import json
import os
from dataclasses import dataclass, fields
from typing import Callable, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn.modules.loss import _Loss

from .app_logger import get_logger

logger = get_logger()


@dataclass
class ModelArgs:
    depth: int = 12
    dim: int = 4096
    intermediate_size: int = 16384
    n_heads: int = 8

    vocab_size: int = -1  # Auto-set
    max_seq_len: int = -1
    pad_id: int = 0

    dropout: float = 0.1
    bias: bool = False
    ffn_post_act_ln: bool = False
    norm_eps: float = 1e-8


@dataclass
class TransformerArgs(ModelArgs):
    kv_heads: int = 4


@dataclass
class TTTArgs(ModelArgs):
    ttt_type: Literal["linear", "mlp", "kan"] = "linear"
    mini_batch_size: int = 16
    ttt_bias: bool = False
    ttt_loss_fn: Type[_Loss] = torch.nn.MSELoss  # TODO: Not configurable from hydra
    ttt_base_lr: float = 1e-3
    ttt_mlp_activation: Literal["tanh", "silu"] = "silu"
    initializer_range: float = 0.02
    generator: Optional[torch.Generator] = None  # TODO: -||-

    kan_init_method: Union[Literal["sin", "sigmoid", "silu", "noise"], Callable] = "noise"
    kan_grid_range: Tuple[float, float] = (-1.0, 1.0)
    kan_degree: int = 3
    kan_control_points: int = 5
    kan_learnable_grid: bool = False  # in outer loop!


@dataclass
class TrainingHyperparameters:
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    test_split: float
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    cycle_lr_optimizer: bool = False
    max_lr: float = 8e-3
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4


@dataclass
class LoggingConfig:
    logger_name: str = "training_logger"
    log_dir: str = "logs"
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None


@dataclass
class TrainingConfig:
    model: TTTArgs
    logging: LoggingConfig
    training: TrainingHyperparameters
    visualize: bool = True
    device: str = "cpu"
    seed: int = 42
    dataset_id: int = 1


@dataclass
class TrainingOutput:
    model: str
    seed: int
    training_uuid: str
    start_time: datetime.datetime
    hyperparameters: dict
    train_loss: List[float]
    validation_loss: List[float]
    dataset: Optional[dict] = None
    end_time: Optional[datetime.datetime] = None
    duration: Optional[float] = None
    stopped_at_epoch: Optional[int] = None
    stop_reason: Optional[str] = None

    def end_training(self, **kwargs):
        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])

    def save(self, path: str = "training_results") -> Optional[str]:
        # TODO: change the default path to absolute path in root directory
        os.makedirs(path, exist_ok=True)
        try:
            existing_files = glob.glob(os.path.join(path, "training_*"))
            file_subscript = 0
            if existing_files:
                max_file_subscript = max(int(f.split("_")[-1].split(".")[0]) for f in existing_files)

                # Check if the last training isn't the same as the current one
                json_files = glob.glob(os.path.join(path, f"training_{max_file_subscript}", "*.json"))
                for json_file in json_files:
                    with open(json_file, "r") as f:
                        last_training = json.load(f)
                        if last_training.get("training_uuid", -1) == self.training_uuid:
                            file_subscript = max_file_subscript
                            break

                    file_subscript = max_file_subscript + 1
                    os.makedirs(os.path.join(path, f"training_{file_subscript}"), exist_ok=False)
                    break
            else:
                os.makedirs(os.path.join(path, f"training_{file_subscript}"), exist_ok=False)

            logger.info(f"Saving training to 'training_{file_subscript}'. ")

            model = self.model.lower().replace(" ", "_")
            model_files = glob.glob(os.path.join(path, f"training_{file_subscript}", f"{model}_training*.json"))
            if not model_files:
                model_file = f"{model}_training.json"
            else:
                if len(model_files) == 1:
                    model_file = f"{model}_training_1.json"
                else:
                    max_model_subscript = max(int(f.split("_")[-1].split(".")[0]) for f in model_files[1:])
                    model_file = f"{model}_training_{max_model_subscript + 1}.json"

            # Save the training output as JSON
            with open(os.path.join(path, f"training_{file_subscript}", model_file), "w") as f:
                data = self.__dict__
                data["start_time"] = data["start_time"].strftime("%Y-%m-%d %H:%M:%S")
                if data["end_time"]:
                    data["end_time"] = data["end_time"].strftime("%Y-%m-%d %H:%M:%S")
                f.write(json.dumps(data, indent=4))

            logger.info(f"Training output saved to 'training_{file_subscript}/{model_file}'. ")
            return os.path.join(path, f"training_{file_subscript}")

        except Exception as e:
            logger.error(f"An error occurred while saving training: `{e}`. ")
            return None
