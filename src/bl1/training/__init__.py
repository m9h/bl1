"""Differentiable training utilities for cortical culture models.

Provides multi-objective loss functions for gradient-based optimisation
of synaptic weights, using differentiable proxies for burst detection,
synchrony, and firing rate regulation.

All loss functions are pure JAX and compatible with ``jax.grad``.
"""

from bl1.training.loss import (
    burst_rate_loss,
    culture_loss,
    firing_rate_loss,
    make_gaussian_kernel,
    synchrony_loss,
    weight_regularization,
)

__all__ = [
    "firing_rate_loss",
    "burst_rate_loss",
    "synchrony_loss",
    "weight_regularization",
    "culture_loss",
    "make_gaussian_kernel",
]
