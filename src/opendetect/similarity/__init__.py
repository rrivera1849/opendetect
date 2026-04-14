"""Similarity metrics for text-vs-text scoring.

Currently ships one metric, :class:`BARTScorer`, the best-performing
similarity in Zhu et al. (2023).  Future metrics (BLEU/ROUGE/BERTScore)
can be added here and selected by name.
"""

from opendetect.similarity.bartscore import BARTScorer

__all__ = ["BARTScorer"]
