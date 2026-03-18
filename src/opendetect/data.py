"""Dataset loading utilities for OpenDetect.

Supports:
- Local JSONL / JSON files
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
        Either a path to a local JSONL/JSON file, or a HuggingFace dataset
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

    if path.suffix in (".jsonl", ".json") or path.exists():
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


def extract_fewshot(
    df: pd.DataFrame,
    num_fewshot: int,
    label_field: str = "label",
    machine_label: int = 1,
    text_field: str = "text",
) -> tuple[pd.DataFrame, list[str]]:
    """Extract few-shot examples from the dataset.

    Selects *num_fewshot* rows whose *label_field* equals *machine_label*,
    removes them from the main dataframe, and returns both the trimmed
    dataframe and the few-shot texts.

    Parameters
    ----------
    df:
        The full dataset.
    num_fewshot:
        Number of machine-text examples to extract for few-shot.
    label_field:
        Column name for the label.
    machine_label:
        Value in *label_field* that indicates machine-generated text.
    text_field:
        Column name for the text.

    Returns
    -------
    (remaining_df, fewshot_texts)
    """
    if num_fewshot <= 0:
        return df, []

    if label_field not in df.columns:
        raise ValueError(
            f"Column {label_field!r} not found in dataset — "
            f"required for few-shot extraction.  "
            f"Available columns: {list(df.columns)}"
        )

    machine_mask = df[label_field] == machine_label
    machine_rows = df[machine_mask]

    if len(machine_rows) < num_fewshot:
        logger.warning(
            "Requested %d few-shot examples but only %d machine texts available. "
            "Using all %d.",
            num_fewshot,
            len(machine_rows),
            len(machine_rows),
        )
        num_fewshot = len(machine_rows)

    fewshot_indices = machine_rows.index[:num_fewshot]
    fewshot_texts = df.loc[fewshot_indices, text_field].tolist()
    remaining = df.drop(fewshot_indices).reset_index(drop=True)

    logger.info("Extracted %d few-shot examples.", len(fewshot_texts))
    return remaining, fewshot_texts


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
