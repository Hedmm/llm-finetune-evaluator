"""Paired-statistics evaluation of fine-tuned LLMs against a baseline.

The package answers one question: is a fine-tuned model actually better than the
base model or another checkpoint, and where does it get worse? It runs every model
on the same eval set, scores each answer, and compares the models item by item
with paired tests, confidence intervals and per-slice regression checks.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
