"""Reliability metrics for the HURDLE multimodal fusion pipeline.

Re-uses the intraclass-correlation (ICC) reliability framing from psychometrics
(inter-observer agreement on ratings) and reapplies it to inter-MODALITY
agreement: each modality is treated as a noisy 'rater' of the same hidden latent
risk, and ICC measures how much the modalities corroborate each other.
"""
from hurdle.reliability.icc import ICCResult, icc

__all__ = ["ICCResult", "icc"]
