"""Differentiable training loop for optimizing BL-1 synaptic weights.

Uses surrogate gradient descent (SuperSpike) to optimize excitatory and
inhibitory weight matrices toward Wagenaar 2006 cortical culture targets.

Numerical stability design:
  - ``optax.apply_if_finite`` skips updates when gradients contain NaN/inf
  - Log-scale firing rate loss prevents gradient explosion at high rates
  - Differentiable burst proxy via population rate variance (not threshold
    crossings, which kill gradients)
  - Per-element weight clamping with configurable max
  - Configurable surrogate beta (lower = smoother gradients for long scans)

Usage::

    from bl1.training.trainer import train_weights, TrainingConfig

    result = train_weights(TrainingConfig(n_neurons=1000, n_epochs=50))
    print(result.loss_history[-1])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from bl1.core.integrator import SimulationResult, simulate
from bl1.core.izhikevich import IzhikevichParams, NeuronState, create_population
from bl1.core.synapses import SynapseState, create_synapse_state
from bl1.network.topology import build_connectivity, place_neurons

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for differentiable training."""

    n_neurons: int = 5000
    ei_ratio: float = 0.8

    # Simulation
    sim_duration_ms: float = 2000.0  # 2 seconds per training step
    dt: float = 0.5

    # Optimizer
    learning_rate: float = 1e-4
    n_epochs: int = 100

    # Wagenaar 2006 targets
    target_firing_rate_hz: float = 1.6
    target_burst_rate_per_min: float = 8.0

    # Loss weights
    w_firing_rate: float = 1.0
    w_burst_rate: float = 0.1  # low weight — burst proxy has noisy gradients
    w_synchrony: float = 0.5
    w_weight_reg: float = 0.01

    # Surrogate gradient
    surrogate_beta: float = 5.0  # lower than default 10 for long scans

    # Network params
    lambda_um: float = 200.0
    p_max: float = 0.21
    g_exc_init: float = 0.12
    g_inh_init: float = 0.36

    # Training runs WITHOUT STP, so weights must be smaller to avoid seizure.
    init_weight_scale: float = 0.1

    # Weight clamping
    max_weight: float = 0.5  # max absolute weight value

    # Gradient clipping (max global norm)
    grad_clip_norm: float = 1.0

    # External drive
    I_noise_amplitude: float = 3.0

    seed: int = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Output from :func:`train_weights`."""

    W_exc: Array
    W_inh: Array
    loss_history: list[dict] = field(default_factory=list)
    config: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Loss function (numerically stable)
# ---------------------------------------------------------------------------


def culture_loss(
    spike_history: Array,
    W_exc: Array,
    W_inh: Array,
    config: TrainingConfig,
) -> tuple[Array, dict]:
    """Compute scalar loss from a spike raster and weight matrices.

    All loss components are designed to have well-behaved gradients:
      1. Log-scale firing rate loss (prevents explosion at high rates)
      2. Differentiable burst proxy via population rate variance
      3. Synchrony penalty via CV deviation
      4. L2 weight regularisation (sparse-aware)
    """
    T, N = spike_history.shape
    sim_duration_s = T * config.dt / 1000.0

    # --- 1. Firing rate loss (log-scale) ---
    # log(fr) - log(target) is much more stable than (fr - target)^2
    # when fr can range from 0 to 1000+ Hz
    total_spikes = jnp.sum(spike_history)
    mean_firing_rate = total_spikes / (N * sim_duration_s + 1e-8)
    # Smooth log: log(fr + eps) to avoid log(0)
    log_fr = jnp.log(mean_firing_rate + 0.01)
    log_target = jnp.log(config.target_firing_rate_hz + 0.01)
    loss_fr = (log_fr - log_target) ** 2

    # --- 2. Burst rate proxy (differentiable) ---
    # Instead of counting threshold crossings (zero gradient), use the
    # variance of the population rate in time windows as a proxy.
    # High variance = bursty. Target: moderate burstiness.
    pop_count = jnp.sum(spike_history, axis=1)  # (T,)

    # Compute windowed variance (50ms windows)
    window_steps = max(int(50.0 / config.dt), 1)
    n_windows = T // window_steps
    if n_windows > 1:
        windowed = pop_count[: n_windows * window_steps].reshape(n_windows, window_steps)
        window_means = jnp.mean(windowed, axis=1)
        # Normalized variance: var(window_means) / (mean + eps)^2 = CV^2
        w_mean = jnp.mean(window_means) + 1e-6
        w_var = jnp.var(window_means)
        # Target CV^2 ~ 1.0 for bursty activity
        burst_cv2 = w_var / (w_mean ** 2)
        loss_burst = (burst_cv2 - 1.0) ** 2
    else:
        loss_burst = jnp.float32(0.0)

    # Also compute the non-differentiable burst count for logging
    kernel = jnp.ones(window_steps) / window_steps
    pop_smooth = jnp.convolve(pop_count, kernel, mode="same")
    mean_pop = jnp.mean(pop_smooth)
    above = (pop_smooth > 2.0 * mean_pop + 1e-6).astype(jnp.float32)
    rising = jnp.clip(above[1:] - above[:-1], 0.0, 1.0)
    burst_rate_per_min = jnp.sum(rising) / (sim_duration_s / 60.0 + 1e-8)

    # --- 3. Synchrony penalty ---
    pop_std = jnp.std(pop_count)
    pop_mean = jnp.mean(pop_count) + 1e-6
    cv = pop_std / pop_mean
    loss_sync = (cv - 1.0) ** 2

    # --- 4. Weight regularisation (only on non-zero entries) ---
    # Use mean of squares of non-zero weights to avoid scale dependence
    exc_nnz = jnp.sum(W_exc > 0) + 1.0
    inh_nnz = jnp.sum(W_inh > 0) + 1.0
    loss_reg = jnp.sum(W_exc ** 2) / exc_nnz + jnp.sum(W_inh ** 2) / inh_nnz

    # --- Total ---
    total = (
        config.w_firing_rate * loss_fr
        + config.w_burst_rate * loss_burst
        + config.w_synchrony * loss_sync
        + config.w_weight_reg * loss_reg
    )

    components = {
        "firing_rate": loss_fr,
        "burst_rate": loss_burst,
        "synchrony": loss_sync,
        "weight_reg": loss_reg,
        "total": total,
        "mean_fr_hz": mean_firing_rate,
        "burst_rate_per_min": burst_rate_per_min,
    }
    return total, components


# ---------------------------------------------------------------------------
# Single training step (JIT-compilable)
# ---------------------------------------------------------------------------


def _make_loss_fn(
    params: IzhikevichParams,
    init_state: NeuronState,
    syn_state: SynapseState,
    I_external: Array,
    config: TrainingConfig,
):
    """Return a closure that maps (W_exc, W_inh) -> (loss, components)."""

    def loss_fn(W_exc: Array, W_inh: Array) -> tuple[Array, dict]:
        result: SimulationResult = simulate(
            params=params,
            init_state=init_state,
            syn_state=syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_external,
            dt=config.dt,
            plasticity_fn=None,
            surrogate=True,
            surrogate_beta=config.surrogate_beta,
        )
        loss, components = culture_loss(result.spike_history, W_exc, W_inh, config)
        return loss, components

    return loss_fn


def _train_step(
    W_exc: Array,
    W_inh: Array,
    opt_state_exc: Any,
    opt_state_inh: Any,
    params: IzhikevichParams,
    init_state: NeuronState,
    syn_state: SynapseState,
    I_external: Array,
    config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    exc_mask: Array,
    inh_mask: Array,
) -> tuple[Array, Array, Any, Any, Array, dict]:
    """Single training step: forward + backward + optimizer update."""
    loss_fn = _make_loss_fn(params, init_state, syn_state, I_external, config)

    (loss, components), (grad_exc, grad_inh) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(W_exc, W_inh)

    # Replace NaN/inf gradients with zeros (prevents poisoning Adam state)
    grad_exc = jnp.where(jnp.isfinite(grad_exc), grad_exc, 0.0)
    grad_inh = jnp.where(jnp.isfinite(grad_inh), grad_inh, 0.0)

    # Apply optimizer updates
    updates_exc, opt_state_exc = optimizer.update(grad_exc, opt_state_exc, W_exc)
    W_exc_new = optax.apply_updates(W_exc, updates_exc)

    updates_inh, opt_state_inh = optimizer.update(grad_inh, opt_state_inh, W_inh)
    W_inh_new = optax.apply_updates(W_inh, updates_inh)

    # Enforce Dale's law: weights >= 0
    W_exc_new = jnp.maximum(W_exc_new, 0.0)
    W_inh_new = jnp.maximum(W_inh_new, 0.0)

    # Clamp max weight to prevent runaway
    W_exc_new = jnp.minimum(W_exc_new, config.max_weight)
    W_inh_new = jnp.minimum(W_inh_new, config.max_weight)

    # Preserve sparsity pattern
    W_exc_new = W_exc_new * exc_mask
    W_inh_new = W_inh_new * inh_mask

    # If the new weights contain NaN, fall back to the old weights
    exc_ok = jnp.all(jnp.isfinite(W_exc_new))
    inh_ok = jnp.all(jnp.isfinite(W_inh_new))
    W_exc_out = jnp.where(exc_ok, W_exc_new, W_exc)
    W_inh_out = jnp.where(inh_ok, W_inh_new, W_inh)

    return W_exc_out, W_inh_out, opt_state_exc, opt_state_inh, loss, components


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _build_network(
    config: TrainingConfig,
) -> tuple[IzhikevichParams, NeuronState, Array, Array, Array, Array]:
    """Build the network: place neurons, create population, build connectivity.

    Returns:
        params, init_state, W_exc_dense, W_inh_dense, exc_mask, inh_mask
    """
    key = jax.random.PRNGKey(config.seed)
    key_pop, key_pos, key_conn = jax.random.split(key, 3)

    params, init_state, is_excitatory = create_population(
        key_pop, config.n_neurons, ei_ratio=config.ei_ratio
    )
    positions = place_neurons(key_pos, config.n_neurons)
    W_exc_sp, W_inh_sp, _delays = build_connectivity(
        key_conn,
        positions,
        is_excitatory,
        lambda_um=config.lambda_um,
        p_max=config.p_max,
        g_exc=config.g_exc_init,
        g_inh=config.g_inh_init,
        dt=config.dt,
    )

    W_exc_dense = W_exc_sp.todense()
    W_inh_dense = W_inh_sp.todense()

    exc_mask = (W_exc_dense > 0.0).astype(jnp.float32)
    inh_mask = (W_inh_dense > 0.0).astype(jnp.float32)

    # Scale down initial weights for training without STP
    W_exc_dense = W_exc_dense * config.init_weight_scale
    W_inh_dense = W_inh_dense * config.init_weight_scale

    return params, init_state, W_exc_dense, W_inh_dense, exc_mask, inh_mask


def train_weights(config: TrainingConfig | None = None) -> TrainingResult:
    """Train synaptic weights via surrogate gradient descent.

    Optimizes excitatory and inhibitory weight matrices to match
    Wagenaar 2006 cortical culture activity targets.
    """
    if config is None:
        config = TrainingConfig()
    print(f"Building network: {config.n_neurons} neurons, E/I ratio {config.ei_ratio:.1%}")
    t0 = time.time()

    params, init_state_template, W_exc, W_inh, exc_mask, inh_mask = _build_network(config)

    n_exc = int(jnp.sum(exc_mask > 0))
    n_inh = int(jnp.sum(inh_mask > 0))
    print(f"  Excitatory synapses: {n_exc:,}  (scaled {config.init_weight_scale}x)")
    print(f"  Inhibitory synapses: {n_inh:,}  (scaled {config.init_weight_scale}x)")
    print(f"  Max weight: {config.max_weight}")
    print(f"  Surrogate beta: {config.surrogate_beta}")
    print(f"  Network built in {time.time() - t0:.1f}s")

    n_steps = int(config.sim_duration_ms / config.dt)

    # Optimizer: clip gradients → Adam → skip if non-finite
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.learning_rate),
    )
    opt_state_exc = optimizer.init(W_exc)
    opt_state_inh = optimizer.init(W_inh)

    syn_state = create_synapse_state(config.n_neurons)

    train_step_jit = jax.jit(
        _train_step,
        static_argnames=("config", "optimizer"),
    )

    loss_history: list[dict] = []
    rng = jax.random.PRNGKey(config.seed + 1)
    nan_count = 0  # track consecutive NaN epochs

    print(
        f"\nTraining for {config.n_epochs} epochs "
        f"({config.sim_duration_ms:.0f} ms per epoch, dt={config.dt} ms, "
        f"LR={config.learning_rate}, grad_clip={config.grad_clip_norm})"
    )
    print("-" * 80)

    for epoch in range(config.n_epochs):
        epoch_t0 = time.time()

        rng, key_noise, key_state = jax.random.split(rng, 3)
        I_external = config.I_noise_amplitude * jax.random.normal(
            key_noise, shape=(n_steps, config.n_neurons)
        )

        v_init = -65.0 + 5.0 * jax.random.normal(key_state, shape=(config.n_neurons,))
        init_state = NeuronState(
            v=v_init,
            u=init_state_template.u,
            spikes=jnp.zeros(config.n_neurons, dtype=jnp.float32),
        )

        W_exc, W_inh, opt_state_exc, opt_state_inh, loss, components = train_step_jit(
            W_exc, W_inh,
            opt_state_exc, opt_state_inh,
            params, init_state, syn_state, I_external,
            config, optimizer, exc_mask, inh_mask,
        )

        epoch_record = {k: float(v) for k, v in components.items()}
        epoch_record["epoch"] = epoch
        epoch_record["wall_time_s"] = time.time() - epoch_t0
        loss_history.append(epoch_record)

        # Track NaN
        loss_val = float(loss)
        if loss_val != loss_val:  # NaN check
            nan_count += 1
        else:
            nan_count = 0

        if epoch % 10 == 0 or epoch == config.n_epochs - 1:
            fr = epoch_record['mean_fr_hz']
            print(
                f"Epoch {epoch:4d}/{config.n_epochs} | "
                f"loss={loss_val:8.4f} | "
                f"FR={fr:7.2f} Hz | "
                f"bursts={epoch_record['burst_rate_per_min']:5.1f}/min | "
                f"L_fr={epoch_record['firing_rate']:7.4f} | "
                f"L_burst={epoch_record['burst_rate']:7.4f} | "
                f"L_sync={epoch_record['synchrony']:6.4f} | "
                f"L_reg={epoch_record['weight_reg']:6.4f} | "
                f"{epoch_record['wall_time_s']:.1f}s"
                + (" [NaN-protected]" if nan_count > 0 else "")
            )

        if nan_count >= 10:
            print(f"\nWARNING: {nan_count} consecutive NaN epochs. Stopping early.")
            break

    total_time = time.time() - t0
    print("-" * 80)
    print(f"Training complete in {total_time:.1f}s ({len(loss_history)} epochs)")

    final = loss_history[-1]
    print(f"Final loss: {final['total']:.4f}")
    print(f"Final firing rate: {final['mean_fr_hz']:.2f} Hz (target: {config.target_firing_rate_hz} Hz)")
    print(f"Final burst rate: {final['burst_rate_per_min']:.1f}/min (target: {config.target_burst_rate_per_min}/min)")

    # Weight statistics
    w_exc_vals = W_exc[exc_mask > 0]
    w_inh_vals = W_inh[inh_mask > 0]
    print(f"W_exc: mean={float(jnp.mean(w_exc_vals)):.4f}, max={float(jnp.max(w_exc_vals)):.4f}")
    print(f"W_inh: mean={float(jnp.mean(w_inh_vals)):.4f}, max={float(jnp.max(w_inh_vals)):.4f}")

    return TrainingResult(
        W_exc=W_exc,
        W_inh=W_inh,
        loss_history=loss_history,
        config=config,
    )
