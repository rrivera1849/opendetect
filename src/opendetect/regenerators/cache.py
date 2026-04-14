"""Persistent cache for K-continuation regenerations.

Sharded by regenerator identity; keyed by
``sha256(prefix || truncate_ratio || K)`` so changing ``K`` or the
truncation ratio invalidates the cache correctly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from opendetect.config import get_output_dir

logger = logging.getLogger(__name__)


def _hash_entry(prefix: str, truncate_ratio: float, K: int) -> str:
    """Return a stable SHA-256 digest for a (prefix, γ, K) triple."""
    payload = f"{prefix}\x00{truncate_ratio}\x00{K}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_shard_id(regenerator_id: str) -> str:
    """Sanitize a regenerator identifier for filesystem use."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", regenerator_id)


class ContinuationCache:
    """Append-only JSONL cache of K continuations per prefix."""

    def __init__(
        self,
        regenerator_id: str,
        cache_dir: Path | None = None,
    ) -> None:
        """Open (or create) the cache file for ``regenerator_id``.

        Parameters
        ----------
        regenerator_id:
            Regenerator-specific shard key (e.g. ``"gpt-3.5-turbo-instruct"``).
        cache_dir:
            Directory to hold cache shards.  Defaults to
            ``<OPENDETECT_OUTPUT_DIR>/dna_gpt_continuations/``.
        """
        base = (
            cache_dir
            if cache_dir is not None
            else get_output_dir() / "dna_gpt_continuations"
        )
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{_safe_shard_id(regenerator_id)}.jsonl"
        self._entries: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        """Slurp existing entries from disk."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Malformed cache line in {self.path}: {line}",
                    )
                self._entries[rec["hash"]] = rec["continuations"]
        logger.info(
            "Loaded %d cached regenerations from %s",
            len(self._entries),
            self.path,
        )

    def get(
        self,
        prefix: str,
        truncate_ratio: float,
        K: int,
    ) -> list[str] | None:
        """Return cached continuations for ``(prefix, γ, K)`` or ``None``."""
        cached = self._entries.get(_hash_entry(prefix, truncate_ratio, K))
        if cached is None:
            return None
        # Guard against under-filled cache entries.
        if len(cached) < K:
            return None
        return cached[:K]

    def put_many(
        self,
        entries: list[tuple[str, float, int, list[str]]],
    ) -> None:
        """Batch-insert ``(prefix, γ, K, continuations)`` tuples."""
        with self.path.open("a", encoding="utf-8") as fh:
            for prefix, truncate_ratio, K, continuations in entries:
                key = _hash_entry(prefix, truncate_ratio, K)
                if key in self._entries:
                    continue
                self._entries[key] = continuations
                fh.write(
                    json.dumps(
                        {
                            "hash": key,
                            "continuations": continuations,
                        },
                    )
                    + "\n",
                )

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._entries)
