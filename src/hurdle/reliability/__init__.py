"""Reliability metrics for the HURDLE multimodal fusion pipeline.

Reuses the intraclass-correlation (ICC) framing from psychometrics for
inter-modality agreement: each modality is a noisy 'rater' of the same hidden
latent risk, and ICC measures how much they corroborate each other.
"""
from hurdle.reliability.icc import ICCResult, icc

__all__ = ["ICCResult", "icc"]
