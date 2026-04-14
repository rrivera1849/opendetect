"""Persistent cache for text revisions, sharded by reviser identity.

Revisions are expensive (API calls or local generation), so we
persist them in JSONL at ``~/.opendetect/revisions/<safe_id>.jsonl``.
Each line is ``{"hash": sha256(text), "revised": "..."}``.  The file
is loaded at init and appended after each batch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from opendetect.config import get_output_dir

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    """Return a stable SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_shard_id(reviser_id: str) -> str:
    """Sanitize a reviser identifier for filesystem use."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", reviser_id)


class RevisionCache:
    """Append-only JSONL cache keyed by SHA-256 of the input text."""

    def __init__(
        self,
        reviser_id: str,
        cache_dir: Path | None = None,
    ) -> None:
        """Open (or create) the cache file for ``reviser_id``.

        Parameters
        ----------
        reviser_id:
            Reviser-specific shard key (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
        cache_dir:
            Directory to hold cache shards.  Defaults to
            ``<OPENDETECT_OUTPUT_DIR>/revisions/``.
        """
        base = cache_dir if cache_dir is not None else get_output_dir() / "revisions"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{_safe_shard_id(reviser_id)}.jsonl"
        self._entries: dict[str, str] = {}
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
                    raise ValueError(f"Malformed cache line in {self.path}: {line}")
                self._entries[rec["hash"]] = rec["revised"]
        logger.info(
            "Loaded %d cached revisions from %s",
            len(self._entries),
            self.path,
        )

    def get(self, text: str) -> str | None:
        """Return the cached revision for ``text`` if present, else ``None``."""
        return self._entries.get(_hash_text(text))

    def put(self, text: str, revised: str) -> None:
        """Store ``revised`` for ``text`` in memory and on disk."""
        key = _hash_text(text)
        if key in self._entries:
            return
        self._entries[key] = revised
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"hash": key, "revised": revised}) + "\n",
            )

    def put_many(
        self,
        pairs: list[tuple[str, str]],
    ) -> None:
        """Batch-insert ``(text, revised)`` pairs."""
        with self.path.open("a", encoding="utf-8") as fh:
            for text, revised in pairs:
                key = _hash_text(text)
                if key in self._entries:
                    continue
                self._entries[key] = revised
                fh.write(
                    json.dumps({"hash": key, "revised": revised}) + "\n",
                )

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._entries)
