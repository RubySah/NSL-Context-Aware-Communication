"""Load project configuration."""

from pathlib import Path

import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    path = ROOT_DIR / "configs" / "config.yaml"

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["device"] = (
        torch.device("cuda")
        if config["device"]["use_cuda"] and torch.cuda.is_available()
        else torch.device("cpu")
    )

    for key, value in config["paths"].items():
        config["paths"][key] = ROOT_DIR / value

    return config
