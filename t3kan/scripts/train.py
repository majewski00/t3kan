from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
import torch
from torch.utils.data import DataLoader, random_split

from t3kan.data import SequenceDataset
from t3kan.models import T3
from t3kan.training.callbacks import generate_training_uuid, set_seed, OptimizerWithCyclicLR, visualize_training
from t3kan.training.trainer import Trainer
from t3kan.utils import TTTArgs, TrainingConfig
from t3kan.utils import get_logger

logger = get_logger()

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainingConfig)

OPTIMIZER_REGISTRY = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


@hydra.main(version_base=None, config_path="../config/model", config_name="train")
def main(cfg: TrainingConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    dataset = SequenceDataset(cfg.dataset_id, device=cfg.device)

    train_dataset, val_dataset, _ = random_split(
        dataset,
        [
            1 - cfg.training.validation_split - cfg.training.test_split,
            cfg.training.validation_split,
            cfg.training.test_split,
        ],
        generator=set_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=set_seed(cfg.seed),
    )

    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    optimizer: Any = None
    if cfg.training.cycle_lr_optimizer:
        optimizer = OptimizerWithCyclicLR(
            optimizer=OPTIMIZER_REGISTRY[cfg.training.optimizer],
            base_lr=cfg.training.learning_rate,
            max_lr=cfg.training.max_lr,
            epoch_steps=len(train_loader),
            initial_visualize=False,
            mode="exp_range",
            gamma_half_life=4,
        )
    else:
        optimizer = OPTIMIZER_REGISTRY[cfg.training.optimizer]

    # TODO: Add support for baseline transformer model
    model = T3(
        TTTArgs(
            depth=cfg.model.depth,
            dim=cfg.model.dim,
            intermediate_size=cfg.model.intermediate_size,
            n_heads=cfg.model.n_heads,
            vocab_size=dataset.vocab_size,
            max_seq_len=dataset.seq_len,
            ttt_bias=cfg.model.ttt_bias,
            ttt_base_lr=cfg.model.ttt_base_lr,
            ttt_type=cfg.model.ttt_type,
            generator=set_seed(cfg.seed),
        )
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        learning_rate=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        device=cfg.device,
        early_stopping=cfg.training.early_stopping,
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta,
        cycle_lr_optimizer=cfg.training.cycle_lr_optimizer,
    )

    training_uuid = generate_training_uuid()
    try:
        results = trainer.run(training_uuid, cfg.seed)
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        results.dataset = {
            "dataset_id": cfg.dataset_id,
            "hyperparameters": {
                "validation_split": cfg.training.validation_split,
                "test_split": cfg.training.test_split,
            },
            "metadata": dataset.metadata,
        }

        path = results.save()
        if cfg.visualize:
            logger.info("Visualizing training...")
            visualize_training(("train_loss", "validation_loss"), results, save_path=path)
