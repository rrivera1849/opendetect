"""Dataset loading utilities for OpenDetect.

Supports:
- Local JSONL / JSON files
- Local CSV files
- HuggingFace dataset identifiers
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(
    source: str,
    text_field: str = "text",
    label_field: str = "label",
    split: str | None = None,
) -> pd.DataFrame:
    """Load a dataset and return a DataFrame with at least *text_field*.

    Parameters
    ----------
    source:
        Either a path to a local JSONL/JSON/CSV file, or a HuggingFace dataset
        identifier (e.g. ``"username/dataset_name"``).
    text_field:
        Name of the column containing the text to classify.
    label_field:
        Name of the column containing the label (used for few-shot extraction).
    split:
        HuggingFace dataset split to load (e.g. ``"test"``).  Ignored for
        local files.  If *None*, defaults to ``"test"`` when loading from
        HuggingFace.

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to have a *text_field* column.
    """
    path = Path(source)

    if path.suffix == ".csv":
        logger.info("Loading local CSV file: %s", source)
        df = pd.read_csv(source)
    elif path.suffix in (".jsonl", ".json") or path.exists():
        logger.info("Loading local file: %s", source)
        lines = path.suffix == ".jsonl" or _looks_like_jsonl(path)
        df = pd.read_json(source, lines=lines)
    else:
        logger.info("Loading HuggingFace dataset: %s (split=%s)", source, split)
        df = _load_from_huggingface(source, split=split or "test")

    if text_field not in df.columns:
        raise ValueError(
            f"Column {text_field!r} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def _load_from_huggingface(name: str, split: str) -> pd.DataFrame:
    """Load a HuggingFace dataset and convert to DataFrame."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for loading HuggingFace datasets. "
            "Install it with:  pip install datasets"
        ) from exc

    ds = hf_load(name, split=split)
    return ds.to_pandas()


def _looks_like_jsonl(path: Path) -> bool:
    """Heuristic: check if the first line is valid JSON (not an array)."""
    try:
        with open(path) as f:
            first = f.readline().strip()
        return first.startswith("{")
    except Exception:
        return False
