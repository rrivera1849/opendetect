"""OpenAI chat API reviser.

Applies the paper's prompt via the OpenAI Python SDK.  The paper's
original model, ``gpt-3.5-turbo-0301``, is deprecated; ``gpt-4o-mini``
is used as the default for a reasonable quality/cost trade-off.
Requests are run concurrently with ``asyncio`` for throughput.
"""

from __future__ import annotations

import asyncio
import logging
import os

from tqdm import tqdm

from opendetect.revisers import REVISE_PROMPT_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_REVISER = "gpt-4o-mini"


class OpenAIReviser:
    """Revise texts with an OpenAI chat completion model."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_REVISER,
        concurrency: int = 16,
        temperature: float = 0.0,
    ) -> None:
        """Build an async OpenAI client.

        Parameters
        ----------
        model_id:
            OpenAI chat model identifier.
        concurrency:
            Maximum number of in-flight requests.
        temperature:
            Sampling temperature.  Default 0.0 for reproducibility.

        Raises
        ------
        RuntimeError
            If ``OPENAI_API_KEY`` is not set.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set.  Either export the key or "
                "switch to --reviser hf for a local model.",
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for --reviser openai. "
                "Install it with `pip install openai`.",
            ) from exc

        self.id = model_id
        self.concurrency = concurrency
        self.temperature = temperature
        self._client = AsyncOpenAI()

    async def _revise_one(
        self,
        text: str,
        sem: asyncio.Semaphore,
    ) -> str:
        """Issue one API call under the concurrency semaphore."""
        async with sem:
            resp = await self._client.chat.completions.create(
                model=self.id,
                messages=[
                    {
                        "role": "user",
                        "content": REVISE_PROMPT_PREFIX + text,
                    },
                ],
                temperature=self.temperature,
            )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                logger.warning(
                    "OpenAI reviser returned empty output; "
                    "falling back to original text.",
                )
                return text
            return out

    async def _revise_all(self, texts: list[str]) -> list[str]:
        """Launch all requests concurrently, bounded by ``concurrency``."""
        sem = asyncio.Semaphore(self.concurrency)
        pbar = tqdm(total=len(texts), desc=f"Revising ({self.id})")

        async def run_and_tick(text: str) -> str:
            result = await self._revise_one(text, sem)
            pbar.update(1)
            return result

        try:
            return await asyncio.gather(*(run_and_tick(t) for t in texts))
        finally:
            pbar.close()

    def revise(
        self,
        texts: list[str],
        batch_size: int = 8,  # noqa: ARG002 — kept for interface parity
    ) -> list[str]:
        """Return one revised text per input, preserving order."""
        return asyncio.run(self._revise_all(texts))

    def close(self) -> None:
        """No-op: OpenAI holds no local GPU state."""
