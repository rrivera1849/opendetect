"""OpenAI completions-API regenerator.

One ``completions.create(n=K)`` call per prefix, run concurrently with
``asyncio`` for throughput.  Default model is ``gpt-3.5-turbo-instruct``,
the live successor to the paper's deprecated ``text-davinci-003``.
"""

from __future__ import annotations

import asyncio
import logging
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_REGENERATOR = "gpt-3.5-turbo-instruct"


class OpenAIRegenerator:
    """Regenerate continuations via OpenAI's completions API."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_REGENERATOR,
        concurrency: int = 16,
    ) -> None:
        """Build an async OpenAI client.

        Parameters
        ----------
        model_id:
            OpenAI completion-style model identifier.
        concurrency:
            Maximum number of in-flight requests.

        Raises
        ------
        RuntimeError
            If ``OPENAI_API_KEY`` is not set or the ``openai`` package
            is missing.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set.  Either export the key or "
                "switch to --dna-gpt-regenerator hf for a local model.",
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for --dna-gpt-regenerator "
                "openai.  Install it with `pip install openai`.",
            ) from exc

        self.id = model_id
        self.concurrency = concurrency
        self._client = AsyncOpenAI()

    async def _regen_one(
        self,
        prefix: str,
        K: int,
        max_new_tokens: int,
        temperature: float,
        sem: asyncio.Semaphore,
    ) -> list[str]:
        """Issue one API call under the concurrency semaphore."""
        async with sem:
            resp = await self._client.completions.create(
                model=self.id,
                prompt=prefix,
                max_tokens=max_new_tokens,
                temperature=temperature,
                n=K,
            )
            return [(c.text or "").strip() for c in resp.choices]

    async def _regen_all(
        self,
        prefixes: list[str],
        K: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[list[str]]:
        """Launch all requests concurrently."""
        sem = asyncio.Semaphore(self.concurrency)
        pbar = tqdm(total=len(prefixes), desc=f"Regenerating ({self.id})")

        async def run_and_tick(prefix: str) -> list[str]:
            out = await self._regen_one(
                prefix, K, max_new_tokens, temperature, sem,
            )
            pbar.update(1)
            return out

        try:
            return await asyncio.gather(
                *(run_and_tick(p) for p in prefixes),
            )
        finally:
            pbar.close()

    def regenerate(
        self,
        prefixes: list[str],
        K: int,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        batch_size: int = 8,  # noqa: ARG002 — kept for interface parity
    ) -> list[list[str]]:
        """Return ``K`` continuations per prefix, preserving order."""
        return asyncio.run(
            self._regen_all(prefixes, K, max_new_tokens, temperature),
        )

    def close(self) -> None:
        """No-op: OpenAI holds no local GPU state."""
