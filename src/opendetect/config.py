"""Configuration helpers for OpenDetect."""

import os
from pathlib import Path

import torch

DEFAULT_OUTPUT_DIR = os.path.join(Path.home(), ".opendetect")
ENV_OUTPUT_DIR = "OPENDETECT_OUTPUT_DIR"


def get_output_dir() -> Path:
    """Return the output directory, creating it if needed."""
    output_dir = Path(os.environ.get(ENV_OUTPUT_DIR, DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
