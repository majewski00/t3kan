import argparse
import glob
import json
import os
import random
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig


def _create_train_data(
    samples: int = 1000,
    vocab_size: int = 1024,
    seq_step: int = 4,
    allow_descending: bool = False,
    components_number: int = 1,
    pad_id: Optional[int] = 0,
    seq_components_weights: int = 2,
    enhance: bool = True,
    name: Optional[str] = None,
) -> None:
    """
    Crete composite sequential training data that will imitate regular pattern and relations in sequences.
    Create a dataset in /seq_dataset folder with training data, targets, and the configuration file.
    Maximum **length of the sequence** will be determined by vocab_size / seq_step.

    Args:
        samples: Number of samples to generate.
        vocab_size: Size of the vocabulary. Assumes that the vocabulary starts from 0.
        seq_step: Maximum difference between regular patterns in the sequence.
        allow_descending: Allow descending patterns in the sequence.
        components_number: Maximum number of joined sequences with different patterns.
        pad_id: Padding id to use for the dataset. If None, no padding will be used and the dataset will be of variable length.
        seq_components_weights: Exponential base for the probability of the number of components in the sequence. The higher the weight, the less probable to have more components.
        enhance: Enhance the dataset by removing the last component in the sequence for each additional component. Less predictable data.
        name: Name of the dataset. If None, the dataset will be saved as dataset_{subscript}.
    Returns:
        None

    """
    if pad_id is not None:
        assert 0 <= pad_id < vocab_size, "Padding id should be within the vocabulary size."
    seq_len = vocab_size // seq_step

    output: dict[str, list] = {"data": [], "target": [], "statistics": []}

    skip = 0
    for _ in range(samples):
        if skip > 0:
            skip -= 1
            continue
        data = []

        _seq_components = random.choices(
            range(1, components_number + 1),
            weights=[seq_components_weights ** (n - 1) for n in range(components_number, 0, -1)],
            k=1,
        )[0]
        _seq_len = random.randint(
            int(vocab_size // _seq_components // seq_step * 0.2), vocab_size // _seq_components // seq_step - 1
        )  # each component will have the same length

        components = []
        for _ in range(_seq_components):
            if allow_descending and random.random() < 0.5:
                _seq_step = random.randint(-seq_step, -1)
                _seq_start = random.randint(_seq_len * abs(_seq_step) + 1, vocab_size - 1)
            else:
                _seq_step = random.randint(1, seq_step)
                _seq_start = random.randint(1, vocab_size - 1 - _seq_len * _seq_step)

            _seq = [i for i in range(_seq_start, _seq_start + _seq_len * _seq_step, _seq_step)]
            components.append(_seq)

        for s in range(_seq_len):
            for c in range(_seq_components):
                data += [components[c][s]]

        target = data[1:]
        data = data[:-1]

        for c in range(_seq_components):
            target = target[: len(target) - c]
            data = data[: len(data) - c]
            if pad_id is not None:
                padded_data = data + [pad_id] * (seq_len - len(data))
                padded_target = target + [pad_id] * (seq_len - len(target))
            else:
                padded_data = data
                padded_target = target

            output["data"].append(padded_data)
            output["target"].append(padded_target)
            output["statistics"].append(
                {
                    "enhanced": enhance and c > 0,
                    "seq_components": _seq_components,
                    "seq_len": _seq_len,
                    "components": torch.tensor(components),
                }
            )
            if not enhance:
                break
            skip += 1 if c > 0 else 0

    os.makedirs("seq_dataset", exist_ok=True)
    existing_files = glob.glob("seq_dataset/dataset_*")
    if existing_files:
        max_subscript = max(int(f.split("_")[-1].split(".")[0]) for f in existing_files)
    else:
        max_subscript = 0
    new_subscript = max_subscript + 1
    os.makedirs(os.path.join("seq_dataset", f"dataset_{new_subscript}"), exist_ok=False)

    for file in ["data", "target"]:
        with open(os.path.join("seq_dataset", f"dataset_{new_subscript}", f"{file}.csv"), "w") as f:
            for line in output[file]:
                f.write(",".join(map(str, line)) + "\n")

    config: dict = {
        "name": name or f"Dataset_{new_subscript}",
        "samples": samples,
        "max_sequence_len": seq_len,
        "vocab_size": vocab_size,
        "max_sequence_step": seq_step,
        "allow_descending": allow_descending,
        "max_number_components": components_number,
        "pad_id": pad_id,
        "seq_components_weights": seq_components_weights,
        "dataset_stats": {},
    }
    config["dataset_stats"] = {}
    for i, out in enumerate(output["statistics"] if isinstance(output["statistics"], list) else []):
        config["dataset_stats"][i] = {
            "enhanced": out["enhanced"],
            "total_sequence_len": out["seq_len"] * out["seq_components"],
            "number_components": out["seq_components"],
            "component_sequence_len": out["seq_len"],
            "components": {},
        }
        for j, comp in enumerate(out["components"]):
            comp = {
                "max": comp.max().item(),
                "min": comp.min().item(),
                "mean": comp.float().mean().item(),
                "step_size": (comp[1] - comp[0]).item(),
            }
            config["dataset_stats"][i]["components"][j] = comp

    with open(os.path.join("seq_dataset", f"dataset_{new_subscript}", "config.json"), "w") as f:
        json.dump(config, f, indent=4)


@hydra.main(
    version_base=None,
    config_path="../config/dataset",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """Entry point for CLI data generation with config override support"""

    parser = argparse.ArgumentParser(
        description="Generate training data. Can use config/dataset/generate.yaml to configure generation."
    )
    parser.add_argument("--samples", type=int, help="Number of samples to generate")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size")
    parser.add_argument("--seq_step", type=int, help="Sequence step size")
    parser.add_argument("--allow_descending", action="store_true", help="Allow descending sequences")
    parser.add_argument("--components_number", type=int, help="Number of components")
    parser.add_argument("--pad_id", type=int, help="Padding ID")
    parser.add_argument("--seq_components_weights", type=int, help="Component weights exponent")
    parser.add_argument("--no_enhance", action="store_true", help="Disable dataset enhancement")
    parser.add_argument("--name", type=str, help="Dataset name")

    args = parser.parse_args()

    params = {
        "samples": args.samples if args.samples is not None else cfg.base.samples,
        "vocab_size": args.vocab_size if args.vocab_size is not None else cfg.base.vocab_size,
        "seq_step": args.seq_step if args.seq_step is not None else cfg.base.seq_step,
        "allow_descending": args.allow_descending if args.allow_descending else cfg.base.allow_descending,
        "components_number": (
            args.components_number if args.components_number is not None else cfg.base.components_number
        ),
        "pad_id": args.pad_id if args.pad_id is not None else cfg.base.pad_id,
        "seq_components_weights": (
            args.seq_components_weights if args.seq_components_weights is not None else cfg.base.seq_components_weights
        ),
        "enhance": not args.no_enhance if args.no_enhance else cfg.base.enhance,
        "name": args.name if args.name is not None else cfg.base.name,
    }

    _create_train_data(**params)
