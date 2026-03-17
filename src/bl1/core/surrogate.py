"""Surrogate gradient functions for differentiable spiking networks.

The spike threshold v >= V_peak has zero gradient almost everywhere.
Surrogate gradients replace the Heaviside step's derivative with a
smooth function during the backward pass, enabling gradient-based
optimization through spiking dynamics.

Supported surrogates (from Neftci et al. 2019):
- SuperSpike: sigma'(x) = 1 / (1 + beta|x|)^2
- SigmoidSurrogate: sigma'(x) = sigmoid(beta*x) * (1 - sigmoid(beta*x))
- FastSigmoid: sigma'(x) = 1 / (1 + beta|x|)  (cheaper than SuperSpike)
- ArcTan: sigma'(x) = 1 / (1 + (beta*x)^2)

Usage:
    from bl1.core.surrogate import superspike_threshold

    # Replace hard threshold in Izhikevich step:
    # OLD: spiked = v_new >= V_PEAK
    # NEW: spiked = superspike_threshold(v_new, V_PEAK, beta=10.0)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, custom_jvp


@custom_jvp
def superspike_threshold(v: Array, threshold: float, beta: float = 10.0) -> Array:
    """Spike threshold with SuperSpike surrogate gradient.

    Forward: binary spikes (v >= threshold).astype(float)
    Backward: d/dv = 1 / (1 + beta * |v - threshold|)^2

    Args:
        v: membrane potential array
        threshold: spike threshold (e.g., 30.0 for Izhikevich)
        beta: sharpness of surrogate (higher = closer to true gradient, but noisier)

    Returns:
        Binary spike indicator (float32), differentiable via surrogate.
    """
    return (v >= threshold).astype(jnp.float32)


@superspike_threshold.defjvp
def superspike_threshold_jvp(primals, tangents):
    v, threshold, beta = primals
    dv, d_threshold, d_beta = tangents
    # Forward pass: hard threshold
    spikes = (v >= threshold).astype(jnp.float32)
    # Surrogate gradient: 1 / (1 + beta * |v - threshold|)^2
    surrogate_grad = 1.0 / (1.0 + beta * jnp.abs(v - threshold)) ** 2
    # Chain rule
    tangent_out = surrogate_grad * dv
    return spikes, tangent_out


@custom_jvp
def sigmoid_threshold(v: Array, threshold: float, beta: float = 5.0) -> Array:
    """Spike threshold with sigmoid surrogate gradient.

    Forward: binary spikes (v >= threshold).astype(float)
    Backward: d/dv = beta * sigmoid(beta*(v - threshold)) * (1 - sigmoid(beta*(v - threshold)))

    Args:
        v: membrane potential array
        threshold: spike threshold
        beta: sharpness of surrogate

    Returns:
        Binary spike indicator (float32), differentiable via surrogate.
    """
    return (v >= threshold).astype(jnp.float32)


@sigmoid_threshold.defjvp
def sigmoid_threshold_jvp(primals, tangents):
    v, threshold, beta = primals
    dv, d_threshold, d_beta = tangents
    spikes = (v >= threshold).astype(jnp.float32)
    x = beta * (v - threshold)
    sig = jax.nn.sigmoid(x)
    surrogate_grad = beta * sig * (1.0 - sig)
    return spikes, surrogate_grad * dv


@custom_jvp
def fast_sigmoid_threshold(v: Array, threshold: float, beta: float = 10.0) -> Array:
    """Spike threshold with fast sigmoid surrogate (cheapest).

    Forward: binary spikes (v >= threshold).astype(float)
    Backward: d/dv = 1 / (1 + beta * |v - threshold|)

    Args:
        v: membrane potential array
        threshold: spike threshold
        beta: sharpness of surrogate

    Returns:
        Binary spike indicator (float32), differentiable via surrogate.
    """
    return (v >= threshold).astype(jnp.float32)


@fast_sigmoid_threshold.defjvp
def fast_sigmoid_threshold_jvp(primals, tangents):
    v, threshold, beta = primals
    dv, d_threshold, d_beta = tangents
    spikes = (v >= threshold).astype(jnp.float32)
    surrogate_grad = 1.0 / (1.0 + beta * jnp.abs(v - threshold))
    return spikes, surrogate_grad * dv


@custom_jvp
def arctan_threshold(v: Array, threshold: float, beta: float = 10.0) -> Array:
    """Spike threshold with arctan surrogate gradient.

    Forward: binary spikes (v >= threshold).astype(float)
    Backward: d/dv = 1 / (1 + (beta * (v - threshold))^2)

    Args:
        v: membrane potential array
        threshold: spike threshold
        beta: sharpness of surrogate

    Returns:
        Binary spike indicator (float32), differentiable via surrogate.
    """
    return (v >= threshold).astype(jnp.float32)


@arctan_threshold.defjvp
def arctan_threshold_jvp(primals, tangents):
    v, threshold, beta = primals
    dv, d_threshold, d_beta = tangents
    spikes = (v >= threshold).astype(jnp.float32)
    surrogate_grad = 1.0 / (1.0 + (beta * (v - threshold)) ** 2)
    return spikes, surrogate_grad * dv
