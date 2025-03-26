import time
from datetime import datetime
from typing import Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from t3kan.training import EarlyStopping, OptimizerWithCyclicLR
from t3kan.utils import TrainingOutput, get_logger

logger = get_logger()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Any,
        learning_rate: float,
        epochs: int,
        device: str,
        early_stopping: bool = False,
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.early_stopping = early_stopping
        self.kwargs = kwargs
        self.criterion = nn.CrossEntropyLoss()
        if early_stopping:
            self.early_stopping_obj = EarlyStopping(verbose_fn=logger.debug, **kwargs)

        # self._train_dataset_length = len(train_loader.dataset)

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            outputs = self.model(inputs)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(self.train_loader.dataset)  # TODO: Check if this is correct
        val_loss = self.validate()
        return train_loss, val_loss

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item() * inputs.size(0)

        return val_loss / len(self.val_loader.dataset)

    def run(self, training_uuid: str, seed: int) -> TrainingOutput:
        logger.info(f"Training '{self.model.__class__.__name__}' model on {self.device}")

        start_time = time.perf_counter()
        results = TrainingOutput(
            train_loss=[],
            validation_loss=[],
            start_time=datetime.now(),
            model=self.model.__class__.__name__,
            seed=seed,
            training_uuid=training_uuid,
            hyperparameters={
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.train_loader.batch_size,
                "optimizer": self.optimizer.__class__.__name__,
            },
        )

        try:
            for epoch in range(self.epochs):
                train_loss, val_loss = self.train_epoch()
                results.train_loss.append(train_loss)
                results.validation_loss.append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - " f"Train Loss: {train_loss:.4f} - " f"Val Loss: {val_loss:.4f}"
                )

                if self.early_stopping and self.early_stopping_obj(val_loss):
                    logger.warning("Early stopping activated.")
                    results.end_training(
                        end_time=datetime.now(),
                        duration=time.perf_counter() - start_time,
                        stopped_at_epoch=epoch + 1,
                        stop_reason="Early stopping activated.",
                    )
                    return results

            results.end_training(
                end_time=datetime.now(),
                duration=time.perf_counter() - start_time,
                stopped_at_epoch=self.epochs,
                stop_reason="Epochs completed.",
            )

        except Exception as e:
            results.end_training(
                end_time=datetime.now(),
                duration=time.perf_counter() - start_time,
                stopped_at_epoch=epoch + 1,
                stop_reason=f"Error: {str(e)}",
            )
            logger.error(f"Training error: {e}")
        finally:
            if isinstance(self.optimizer, OptimizerWithCyclicLR):
                results.hyperparameters["optimizer"] = self.optimizer.metadata

        return results
