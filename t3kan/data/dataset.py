import json
import os
from typing import Any, Dict, List, Tuple

import torch

from t3kan.utils import get_logger

logger = get_logger()


def _load_dataset(subscript: int) -> Tuple[List[List[int]], List[List[int]], Dict[str, Any]]:
    dataset_path = os.path.join("seq_dataset", f"dataset_{subscript}")
    config_file = os.path.join(dataset_path, "config.json")
    data_file = os.path.join(dataset_path, "data.csv")
    target_file = os.path.join(dataset_path, "target.csv")

    with open(config_file, "r") as f:
        config = json.load(f)

    logger.debug(f"Loading dataset: {config['name']}...")
    config.pop("dataset_stats")

    data = []
    target = []

    with open(data_file, "r") as f:
        for line in f:
            data.append(list(map(int, line.strip().split(","))))

    with open(target_file, "r") as f:
        for line in f:
            target.append(list(map(int, line.strip().split(","))))

    logger.debug(f"Loaded {len(data)} samples from {config['name']} dataset.")
    return data, target, config


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_subscript: int, device: str = "cpu"):
        data, target, self._config = _load_dataset(dataset_subscript)
        self.data = torch.tensor(data).to(device)
        self.target = torch.tensor(target).to(device)

    @property
    def seq_len(self):
        return self._config["max_sequence_len"]

    @property
    def vocab_size(self):
        return self._config["vocab_size"]

    @property
    def name(self):
        return self._config["name"]

    @property
    def metadata(self):
        return self._config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
