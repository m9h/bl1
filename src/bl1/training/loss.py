"""Multi-objective loss functions for differentiable cortical culture training.

Every function in this module is pure JAX (``jax.numpy``, no NumPy) and fully
differentiable, suitable for use with ``jax.grad`` / ``jax.value_and_grad``.

The key challenge is that classical burst detection (see
``bl1.analysis.bursts``) relies on hard threshold crossings and Python loops,
which are non-differentiable.  This module provides smooth, differentiable
proxies for burst-related metrics using Gaussian-smoothed population rates
and soft sigmoid threshold crossings.

Biological targets follow Wagenaar et al. 2006 (dense cortical cultures):
  - Firing rate: ~1.4 Hz per neuron
  - Network burst rate: ~8 bursts/min
  - Fano factor (synchrony): ~1.5 in 10 ms windows

Weight priors follow Song et al. 2005 (log-normal PSP distributions).

Usage::

    import jax
    from bl1.training.loss import culture_loss

    def loss_fn(W_exc, W_inh, spike_history):
        total, components = culture_loss(spike_history, W_exc, W_inh)
        return total

    grad_fn = jax.grad(lambda w_e, w_i, s: culture_loss(s, w_e, w_i)[0],
                        argnums=(0, 1))
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from bl1.core.regularization import (
    firing_rate_penalty,
    silence_penalty,
    sparsity_penalty,
)

# ---------------------------------------------------------------------------
# Gaussian kernel (constructed once, passed as a static parameter)
# ---------------------------------------------------------------------------


def make_gaussian_kernel(sigma_ms: float = 25.0, dt: float = 0.5) -> Array:
    """Create a normalised 1-D Gaussian kernel for smoothing population rate.

    The kernel is truncated at +/- 3 sigma and normalised to sum to 1.
    Construct this **once** outside the loss function and pass it in to
    avoid recomputation on every call.

    Args:
        sigma_ms: Standard deviation of the Gaussian in milliseconds.
        dt: Simulation timestep in milliseconds.

    Returns:
        1-D ``jnp.ndarray`` of shape ``(kernel_len,)`` summing to 1.
    """
    sigma_steps = sigma_ms / dt
    # Truncate at 3 sigma on each side
    half_width = int(3.0 * sigma_steps)
    # Ensure at least 1 element on each side
    half_width = max(half_width, 1)
    t = jnp.arange(-half_width, half_width + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (t / sigma_steps) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return kernel


# Module-level default kernel (sigma=10 ms, dt=0.5 ms).
# A 10 ms sigma reliably resolves individual bursts without merging
# adjacent ones.  Users working with different dt or who need wider
# smoothing should call ``make_gaussian_kernel`` and pass the result
# to ``burst_rate_loss`` via the ``kernel`` parameter.
_DEFAULT_KERNEL = make_gaussian_kernel(sigma_ms=10.0, dt=0.5)


# ---------------------------------------------------------------------------
# 1. Differentiable firing rate loss
# ---------------------------------------------------------------------------


def firing_rate_loss(
    spike_history: Array,
    target_hz: float = 1.4,
    dt: float = 0.5,
) -> Array:
    """L2 loss on mean per-neuron firing rate versus target.

    This is a thin wrapper around :func:`bl1.core.regularization.firing_rate_penalty`
    that uses the Wagenaar 2006 default target of 1.4 Hz.

    Args:
        spike_history: ``(T, N)`` float array where 1.0 = spike.
        target_hz: Target mean firing rate in Hz (default 1.4, Wagenaar 2006).
        dt: Simulation timestep in ms.

    Returns:
        Scalar L2 penalty, zero when all neurons fire at ``target_hz``.
    """
    return firing_rate_penalty(spike_history, target_rate_hz=target_hz, dt_ms=dt)


# ---------------------------------------------------------------------------
# 2. Differentiable burst rate proxy
# ---------------------------------------------------------------------------


def burst_rate_loss(
    spike_history: Array,
    target_bursts_per_min: float = 8.0,
    dt: float = 0.5,
    window_ms: float = 50.0,
    threshold_factor: float = 2.0,
    sigmoid_beta: float = 10.0,
    kernel: Array | None = None,
) -> Array:
    """Differentiable proxy for network burst rate.

    Classical burst detection (``bl1.analysis.bursts.detect_bursts``) uses
    hard threshold crossings and Python loops, making it non-differentiable.
    This function provides a smooth alternative using ``tanh``-based
    derivative sign-change detection:

    1. Compute the instantaneous population spike count ``pop(t)``.
    2. Smooth ``pop(t)`` with a Gaussian kernel (differentiable convolution).
    3. Compute the adaptive threshold ``mu + factor * sigma``.
    4. Compute the discrete derivative of the smoothed signal and apply
       ``tanh(beta * deriv / sigma)`` to get a smooth +1/-1 sign indicator
       that is exactly 0 for zero derivative (flat/silent regions).
    5. Detect peaks via sign change: ``tanh_deriv[t-1] - tanh_deriv[t]``.
       At a true peak this is ~2 (rising then falling); in flat regions
       it is ~0 (both derivatives near zero).  Normalise to [0, 1] and
       clamp negatives (which correspond to valleys).
    6. Gate by a soft above-threshold indicator and sum to get a
       differentiable burst count.  Compare to target.

    The ``tanh`` formulation avoids the ``sigmoid(0) = 0.5`` floor that
    plagues naive sigmoid-based peak counters, because ``tanh(0) = 0``.

    Args:
        spike_history: ``(T, N)`` float spike array.
        target_bursts_per_min: Target burst rate (default 8.0, Wagenaar 2006).
        dt: Timestep in ms.
        window_ms: Not used directly in computation (retained for API
            compatibility).  The smoothing width is controlled by the
            ``kernel`` parameter.
        threshold_factor: Number of standard deviations above the mean for
            burst onset detection (default 2.0, matching ``detect_bursts``).
        sigmoid_beta: Sharpness of the soft threshold / peak detection,
            in units of standard deviations.  Higher values approximate the
            hard threshold more closely; lower values yield smoother
            gradients.  Default 10.0.
        kernel: Pre-computed Gaussian smoothing kernel from
            :func:`make_gaussian_kernel`.  If ``None``, uses the module-level
            default (sigma=10 ms, dt=0.5 ms).

    Returns:
        Scalar L2 loss: ``(estimated_bursts_per_min - target)^2``.
    """
    if kernel is None:
        kernel = _DEFAULT_KERNEL

    T = spike_history.shape[0]
    duration_min = (T * dt) / 60_000.0  # total duration in minutes

    # Step 1: population spike count per timestep
    pop_count = jnp.sum(spike_history, axis=1)  # (T,)

    # Step 2: smooth with Gaussian kernel via 1-D convolution
    # jnp.convolve pads with zeros; 'same' keeps the length equal to T
    smoothed = jnp.convolve(pop_count, kernel, mode="same")  # (T,)

    # Step 3: adaptive threshold = mean + factor * std
    mu = jnp.mean(smoothed)
    sigma = jnp.std(smoothed)
    sigma_safe = sigma + 1e-6
    threshold = mu + threshold_factor * sigma_safe

    # Step 4: tanh-based derivative sign detection
    deriv = smoothed[1:] - smoothed[:-1]  # (T-1,)
    tanh_deriv = jnp.tanh(sigmoid_beta * deriv / sigma_safe)

    # Step 5: sign change from positive to negative = peak
    # At a peak: tanh_deriv[t-1] ~ +1, tanh_deriv[t] ~ -1 => diff ~ 2
    # In flat regions: both ~ 0 => diff ~ 0
    # In valleys: diff ~ -2 (clamped to 0)
    sign_change = tanh_deriv[:-1] - tanh_deriv[1:]  # (T-2,)
    soft_peak = jnp.maximum(0.0, sign_change) / 2.0  # normalise to [0, 1]

    # Step 6: gate by above-threshold and sum
    above_thr = jax_sigmoid(sigmoid_beta * (smoothed[1:-1] - threshold) / sigma_safe)
    burst_peaks = soft_peak * above_thr  # (T-2,)
    burst_count = jnp.sum(burst_peaks)

    # Convert to bursts per minute
    duration_min_safe = jnp.maximum(duration_min, 1e-6)
    bursts_per_min = burst_count / duration_min_safe

    # L2 loss vs target
    return (bursts_per_min - target_bursts_per_min) ** 2


def jax_sigmoid(x: Array) -> Array:
    """Numerically stable sigmoid, pure JAX."""
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# 3. Differentiable synchrony loss (Fano factor proxy)
# ---------------------------------------------------------------------------


def synchrony_loss(
    spike_history: Array,
    target_fano: float = 1.0,
    window_ms: float = 10.0,
    dt: float = 0.5,
) -> Array:
    """Penalise deviation of population synchrony from a target Fano factor.

    The Fano factor (variance / mean) of the population spike count in
    sliding windows is a standard measure of synchrony.  Values near 1
    indicate Poisson-like (asynchronous) firing; values >> 1 indicate
    synchronous bursting.

    This function computes the Fano factor using non-overlapping windows
    of ``window_ms`` duration and returns the squared deviation from
    ``target_fano``.  All operations are naturally differentiable.

    The window size is computed from ``window_ms`` and ``dt`` at Python
    level (not traced), so this function is JIT-compatible as long as
    ``window_ms`` and ``dt`` are static (not JAX arrays).

    Args:
        spike_history: ``(T, N)`` float spike array.
        target_fano: Target Fano factor (default 1.0 = Poisson).
            Wagenaar cultures are ~1.5; use that for realistic bursting.
        window_ms: Window width in ms for computing population counts.
        dt: Timestep in ms.

    Returns:
        Scalar L2 loss: ``(fano_factor - target_fano)^2``.
    """
    T = spike_history.shape[0]
    # Compute window size as a Python int so slicing is static under JIT
    window_steps = max(int(window_ms / dt), 1)

    # Population count per timestep
    pop_count = jnp.sum(spike_history, axis=1)  # (T,)

    # Number of complete non-overlapping windows
    n_windows = T // window_steps
    usable = n_windows * window_steps

    # Truncate to exact multiple and reshape (all static shapes under JIT)
    windowed = pop_count[:usable].reshape(n_windows, window_steps)

    # Sum spikes in each window
    window_counts = jnp.sum(windowed, axis=1)  # (n_windows,)

    # Fano factor = var / mean (add epsilon for numerical stability)
    mean_count = jnp.mean(window_counts)
    var_count = jnp.var(window_counts)
    fano = var_count / (mean_count + 1e-8)

    return (fano - target_fano) ** 2


# ---------------------------------------------------------------------------
# 4. Weight regularization (log-normal prior)
# ---------------------------------------------------------------------------


def weight_regularization(
    W_exc: Array,
    W_inh: Array,
    log_normal_mu: float = -0.702,
    log_normal_sigma: float = 0.9355,
) -> Array:
    """Penalise deviation of synaptic weights from a log-normal distribution.

    Song et al. 2005 found that excitatory PSP amplitudes follow a
    log-normal distribution.  This regulariser uses moment matching on the
    log-transformed non-zero excitatory weights to encourage the learned
    weight distribution to stay close to the biological prior.

    For inhibitory weights (stored as positive magnitudes), a separate
    simpler L2 penalty on the mean log-weight is applied to prevent runaway
    inhibition.

    The loss is the sum of:

    * Squared error on the mean of log(W_exc) vs ``log_normal_mu``.
    * Squared error on the std of log(W_exc) vs ``log_normal_sigma``.
    * Squared error on the mean of log(|W_inh|) vs ``log_normal_mu``
      (inhibitory weights also tend toward log-normal; Buzsaki & Mizuseki 2014).

    All operations handle zero weights gracefully by adding a small epsilon
    before taking logarithms.

    Args:
        W_exc: Excitatory weight array (any shape).  Non-negative values.
        W_inh: Inhibitory weight array (any shape).  Stored as positive
            magnitudes (absolute values).
        log_normal_mu: Target mean of log-weights (default -0.702, from
            Song et al. 2005 fit to cortical data).
        log_normal_sigma: Target std of log-weights (default 0.9355).

    Returns:
        Scalar regularisation loss.
    """
    eps = 1e-8

    # --- Excitatory weights ---
    # Flatten and use a soft mask for non-zero weights so that gradients
    # flow smoothly.  Weights very close to zero contribute negligibly.
    w_exc_flat = jnp.ravel(W_exc)
    # Soft mask: sigmoid ramps from 0 to 1 around a small threshold
    exc_mask = jax_sigmoid(20.0 * (w_exc_flat - eps))
    # Weighted log-weights (masked, so near-zero entries contribute ~0)
    log_w_exc = jnp.log(w_exc_flat + eps)
    n_exc_eff = jnp.sum(exc_mask) + eps  # effective count of nonzero weights

    exc_mean = jnp.sum(exc_mask * log_w_exc) / n_exc_eff
    exc_var = jnp.sum(exc_mask * (log_w_exc - exc_mean) ** 2) / n_exc_eff
    exc_std = jnp.sqrt(exc_var + eps)

    loss_exc_mean = (exc_mean - log_normal_mu) ** 2
    loss_exc_std = (exc_std - log_normal_sigma) ** 2

    # --- Inhibitory weights ---
    w_inh_flat = jnp.ravel(jnp.abs(W_inh))
    inh_mask = jax_sigmoid(20.0 * (w_inh_flat - eps))
    log_w_inh = jnp.log(w_inh_flat + eps)
    n_inh_eff = jnp.sum(inh_mask) + eps

    inh_mean = jnp.sum(inh_mask * log_w_inh) / n_inh_eff
    inh_var = jnp.sum(inh_mask * (log_w_inh - inh_mean) ** 2) / n_inh_eff
    inh_std = jnp.sqrt(inh_var + eps)

    loss_inh_mean = (inh_mean - log_normal_mu) ** 2
    loss_inh_std = (inh_std - log_normal_sigma) ** 2

    return loss_exc_mean + loss_exc_std + loss_inh_mean + loss_inh_std


# ---------------------------------------------------------------------------
# 5. Combined multi-objective loss
# ---------------------------------------------------------------------------


def culture_loss(
    spike_history: Array,
    W_exc: Array,
    W_inh: Array,
    dt: float = 0.5,
    # Wagenaar 2006 targets
    target_firing_rate_hz: float = 1.4,
    target_burst_rate_per_min: float = 8.0,
    target_fano: float = 1.5,
    # Loss weights
    w_firing_rate: float = 1.0,
    w_burst_rate: float = 1.0,
    w_synchrony: float = 0.5,
    w_silence: float = 0.1,
    w_sparsity: float = 0.1,
    w_weight_reg: float = 0.01,
    # Optional pre-computed kernel
    kernel: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Combined multi-objective loss for cortical culture training.

    Aggregates six differentiable loss components targeting biologically
    realistic cortical culture dynamics (Wagenaar et al. 2006):

    1. **Firing rate**: L2 penalty on per-neuron mean rate vs target.
    2. **Burst rate**: Differentiable proxy for network burst frequency.
    3. **Synchrony**: Fano factor deviation from target.
    4. **Silence**: Penalty for neurons below 0.5 Hz (from ``regularization.py``).
    5. **Sparsity**: Penalty for neurons above 20 Hz (from ``regularization.py``).
    6. **Weight regularisation**: Log-normal distribution prior on weights.

    Args:
        spike_history: ``(T, N)`` float spike array.
        W_exc: Excitatory weight matrix (any shape).
        W_inh: Inhibitory weight matrix (any shape).
        dt: Simulation timestep in ms.
        target_firing_rate_hz: Wagenaar 2006 target (~1.4 Hz).
        target_burst_rate_per_min: Wagenaar 2006 target (~8 bursts/min).
        target_fano: Target Fano factor (~1.5 for cultured networks).
        w_firing_rate: Weight for firing rate loss.
        w_burst_rate: Weight for burst rate loss.
        w_synchrony: Weight for synchrony loss.
        w_silence: Weight for silence penalty.
        w_sparsity: Weight for sparsity penalty.
        w_weight_reg: Weight for weight regularisation.
        kernel: Pre-computed Gaussian kernel for burst smoothing.
            If ``None``, uses default (sigma=10 ms, dt=0.5 ms).

    Returns:
        ``(total_loss, loss_dict)`` where ``loss_dict`` maps component
        names to their *unweighted* scalar values (useful for logging /
        TensorBoard).
    """
    l_firing = firing_rate_loss(spike_history, target_hz=target_firing_rate_hz, dt=dt)
    l_burst = burst_rate_loss(
        spike_history,
        target_bursts_per_min=target_burst_rate_per_min,
        dt=dt,
        kernel=kernel,
    )
    l_sync = synchrony_loss(
        spike_history,
        target_fano=target_fano,
        dt=dt,
    )
    l_silence = silence_penalty(spike_history, min_rate_hz=0.5, dt_ms=dt)
    l_sparsity = sparsity_penalty(spike_history, max_rate_hz=20.0, dt_ms=dt)
    l_weight = weight_regularization(W_exc, W_inh)

    total = (
        w_firing_rate * l_firing
        + w_burst_rate * l_burst
        + w_synchrony * l_sync
        + w_silence * l_silence
        + w_sparsity * l_sparsity
        + w_weight_reg * l_weight
    )

    loss_dict = {
        "firing_rate": l_firing,
        "burst_rate": l_burst,
        "synchrony": l_sync,
        "silence": l_silence,
        "sparsity": l_sparsity,
        "weight_reg": l_weight,
        "total": total,
    }

    return total, loss_dict
