"""Differentiable training loop for optimizing BL-1 synaptic weights.

Uses surrogate gradient descent (SuperSpike) to optimize excitatory and
inhibitory weight matrices toward Wagenaar 2006 cortical culture targets:
  - Firing rate ~1.4 Hz
  - Burst rate ~8.0 bursts/min

The optimizer (optax.adam) acts on dense copies of the weight matrices.
After each gradient step, Dale's law is enforced (W_exc >= 0, W_inh >= 0)
and the original sparsity pattern is preserved (no new connections created).

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
    learning_rate: float = 1e-3
    n_epochs: int = 100

    # Wagenaar 2006 targets
    target_firing_rate_hz: float = 1.4
    target_burst_rate_per_min: float = 8.0

    # Loss weights
    w_firing_rate: float = 1.0
    w_burst_rate: float = 1.0
    w_synchrony: float = 0.5
    w_weight_reg: float = 0.01

    # Surrogate gradient
    surrogate_beta: float = 10.0

    # Network params
    lambda_um: float = 200.0
    p_max: float = 0.02
    g_exc_init: float = 0.50
    g_inh_init: float = 2.00

    # External drive
    I_noise_amplitude: float = 5.0

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
# Loss function
# ---------------------------------------------------------------------------


def culture_loss(
    spike_history: Array,
    W_exc: Array,
    W_inh: Array,
    config: TrainingConfig,
) -> tuple[Array, dict]:
    """Compute scalar loss from a spike raster and weight matrices.

    Loss components:
        1. Firing rate MSE vs Wagenaar 2006 target (1.4 Hz).
        2. Burst rate proxy: penalise deviation from target temporal
           clustering of spikes (8 bursts/min).
        3. Synchrony penalty: discourage hyper-synchronous activity
           (all neurons firing simultaneously) via coefficient of
           variation of the population spike count over time.
        4. L2 weight regularisation to prevent runaway growth.

    Args:
        spike_history: Float spike raster, shape (T, N), values 0.0/1.0.
        W_exc: Excitatory weight matrix, shape (N, N).
        W_inh: Inhibitory weight matrix, shape (N, N).
        config: Training configuration.

    Returns:
        Tuple of (scalar_loss, component_dict) where component_dict has
        keys ``"firing_rate"``, ``"burst_rate"``, ``"synchrony"``,
        ``"weight_reg"``, and ``"total"``.
    """
    T, N = spike_history.shape
    sim_duration_s = T * config.dt / 1000.0

    # --- 1. Firing rate loss --------------------------------------------------
    # Mean firing rate across all neurons (Hz)
    mean_spike_count = jnp.sum(spike_history) / N
    mean_firing_rate = mean_spike_count / sim_duration_s
    loss_fr = (mean_firing_rate - config.target_firing_rate_hz) ** 2

    # --- 2. Burst rate proxy --------------------------------------------------
    # Population spike count per timestep
    pop_count = jnp.sum(spike_history, axis=1)  # (T,)

    # Smooth with a boxcar filter (~50 ms window) to detect bursts
    window_steps = max(int(50.0 / config.dt), 1)
    kernel = jnp.ones(window_steps) / window_steps
    # Use 1-D convolution (valid mode via lax.conv)
    pop_smooth = jnp.convolve(pop_count, kernel, mode="same")

    # A "burst" is defined as a time bin where smoothed population count
    # exceeds 2x the mean.  Count threshold crossings (rising edges).
    mean_pop = jnp.mean(pop_smooth)
    burst_threshold = 2.0 * mean_pop + 1e-6  # +eps to avoid div-by-zero
    above = (pop_smooth > burst_threshold).astype(jnp.float32)
    # Rising edges: diff > 0
    rising = jnp.clip(above[1:] - above[:-1], 0.0, 1.0)
    n_bursts = jnp.sum(rising)
    sim_duration_min = sim_duration_s / 60.0
    burst_rate_per_min = n_bursts / jnp.maximum(sim_duration_min, 1e-6)
    loss_burst = (burst_rate_per_min - config.target_burst_rate_per_min) ** 2

    # --- 3. Synchrony penalty ------------------------------------------------
    # CV of the population spike count: high CV = bursty (good), but
    # very high synchrony (single massive burst) is pathological.
    # Penalise absolute deviation of CV from a moderate target (~1.0).
    pop_std = jnp.std(pop_count)
    pop_mean = jnp.mean(pop_count) + 1e-6
    cv = pop_std / pop_mean
    loss_sync = (cv - 1.0) ** 2

    # --- 4. Weight regularisation --------------------------------------------
    loss_reg = jnp.mean(W_exc**2) + jnp.mean(W_inh**2)

    # --- Total ----------------------------------------------------------------
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
    """Return a closure that maps (W_exc, W_inh) -> (loss, components).

    The closure captures all non-weight arguments so that
    ``jax.value_and_grad`` only differentiates w.r.t. the weight
    matrices.
    """

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
    """Single training step: forward + backward + optimizer update.

    Args:
        W_exc: Current excitatory weight matrix (N, N) dense.
        W_inh: Current inhibitory weight matrix (N, N) dense.
        opt_state_exc: Optax optimizer state for W_exc.
        opt_state_inh: Optax optimizer state for W_inh.
        params: Fixed Izhikevich neuron parameters.
        init_state: Initial neuron state for this epoch.
        syn_state: Initial synapse state (zeroed).
        I_external: External current array (T, N).
        config: Training configuration.
        optimizer: Optax optimizer instance.
        exc_mask: Binary mask of original excitatory connectivity (N, N).
        inh_mask: Binary mask of original inhibitory connectivity (N, N).

    Returns:
        Tuple of (W_exc, W_inh, opt_state_exc, opt_state_inh, loss, components).
    """
    loss_fn = _make_loss_fn(params, init_state, syn_state, I_external, config)

    # Compute loss and gradients w.r.t. both weight matrices.
    # Use has_aux=True because loss_fn returns (loss, components).
    # Differentiate w.r.t. argnums (0, 1) = (W_exc, W_inh).
    (loss, components), (grad_exc, grad_inh) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(W_exc, W_inh)

    # Apply optimizer updates independently for E and I
    updates_exc, opt_state_exc = optimizer.update(grad_exc, opt_state_exc, W_exc)
    W_exc = optax.apply_updates(W_exc, updates_exc)

    updates_inh, opt_state_inh = optimizer.update(grad_inh, opt_state_inh, W_inh)
    W_inh = optax.apply_updates(W_inh, updates_inh)

    # Enforce Dale's law: excitatory weights >= 0, inhibitory weights >= 0
    W_exc = jnp.maximum(W_exc, 0.0)
    W_inh = jnp.maximum(W_inh, 0.0)

    # Preserve sparsity: zero out any weights where original connectivity was absent
    W_exc = W_exc * exc_mask
    W_inh = W_inh * inh_mask

    return W_exc, W_inh, opt_state_exc, opt_state_inh, loss, components


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

    # Create neuron population
    params, init_state, is_excitatory = create_population(
        key_pop, config.n_neurons, ei_ratio=config.ei_ratio
    )

    # Place neurons on substrate
    positions = place_neurons(key_pos, config.n_neurons)

    # Build distance-dependent connectivity (returns BCOO sparse)
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

    # Convert to dense for differentiable training
    # (surrogate gradient path requires dense weight matrices)
    W_exc_dense = W_exc_sp.todense()
    W_inh_dense = W_inh_sp.todense()

    # Store sparsity masks (binary: 1 where connection exists, 0 otherwise)
    exc_mask = (W_exc_dense > 0.0).astype(jnp.float32)
    inh_mask = (W_inh_dense > 0.0).astype(jnp.float32)

    return params, init_state, W_exc_dense, W_inh_dense, exc_mask, inh_mask


def train_weights(config: TrainingConfig | None = None) -> TrainingResult:
    """Train synaptic weights via surrogate gradient descent.

    Optimizes excitatory and inhibitory weight matrices to match
    Wagenaar 2006 cortical culture activity targets using the
    SuperSpike surrogate gradient through ``bl1.core.integrator.simulate``.

    Steps:
        1. Build network (place_neurons, create_population, build_connectivity).
        2. Initialize optimizer (optax.adam).
        3. For each epoch:
           a. Generate external noise current.
           b. Forward pass: simulate with surrogate=True.
           c. Backward pass: compute gradients of culture_loss w.r.t. weights.
           d. Apply Adam optimizer update.
           e. Clamp weights (Dale's law) and preserve sparsity.
           f. Log loss components.
        4. Return final weights + training history.

    Args:
        config: Training configuration. See :class:`TrainingConfig`.

    Returns:
        :class:`TrainingResult` with optimized W_exc, W_inh, and loss history.
    """
    if config is None:
        config = TrainingConfig()
    print(f"Building network: {config.n_neurons} neurons, E/I ratio {config.ei_ratio:.1%}")
    t0 = time.time()

    # 1. Build network
    params, init_state_template, W_exc, W_inh, exc_mask, inh_mask = _build_network(config)

    n_exc = int(jnp.sum(exc_mask > 0))
    n_inh = int(jnp.sum(inh_mask > 0))
    print(f"  Excitatory synapses: {n_exc:,}")
    print(f"  Inhibitory synapses: {n_inh:,}")
    print(f"  Network built in {time.time() - t0:.1f}s")

    # Number of simulation timesteps per epoch
    n_steps = int(config.sim_duration_ms / config.dt)

    # 2. Initialize optimizer (separate states for E and I)
    optimizer = optax.adam(config.learning_rate)
    opt_state_exc = optimizer.init(W_exc)
    opt_state_inh = optimizer.init(W_inh)

    # Create synapse state template
    syn_state = create_synapse_state(config.n_neurons)

    # 3. JIT-compile the training step
    train_step_jit = jax.jit(
        _train_step,
        static_argnames=("config", "optimizer"),
    )

    # 4. Training loop
    loss_history: list[dict] = []
    rng = jax.random.PRNGKey(config.seed + 1)

    print(
        f"\nTraining for {config.n_epochs} epochs "
        f"({config.sim_duration_ms:.0f} ms per epoch, dt={config.dt} ms)"
    )
    print("-" * 72)

    for epoch in range(config.n_epochs):
        epoch_t0 = time.time()

        # Generate fresh noise current for this epoch
        rng, key_noise, key_state = jax.random.split(rng, 3)
        I_external = config.I_noise_amplitude * jax.random.normal(
            key_noise, shape=(n_steps, config.n_neurons)
        )

        # Reset neuron state each epoch (start from rest)
        # Small random perturbation to membrane potential to break symmetry
        v_init = -65.0 + 5.0 * jax.random.normal(key_state, shape=(config.n_neurons,))
        init_state = NeuronState(
            v=v_init,
            u=init_state_template.u,
            spikes=jnp.zeros(config.n_neurons, dtype=jnp.float32),
        )

        # Single training step
        W_exc, W_inh, opt_state_exc, opt_state_inh, loss, components = train_step_jit(
            W_exc,
            W_inh,
            opt_state_exc,
            opt_state_inh,
            params,
            init_state,
            syn_state,
            I_external,
            config,
            optimizer,
            exc_mask,
            inh_mask,
        )

        # Log (convert JAX arrays to Python floats for serialisation)
        epoch_record = {k: float(v) for k, v in components.items()}
        epoch_record["epoch"] = epoch
        epoch_record["wall_time_s"] = time.time() - epoch_t0
        loss_history.append(epoch_record)

        # Print progress every 10 epochs (and first + last)
        if epoch % 10 == 0 or epoch == config.n_epochs - 1:
            print(
                f"Epoch {epoch:4d}/{config.n_epochs} | "
                f"loss={float(loss):8.4f} | "
                f"FR={epoch_record['mean_fr_hz']:5.2f} Hz | "
                f"bursts={epoch_record['burst_rate_per_min']:5.1f}/min | "
                f"L_fr={epoch_record['firing_rate']:7.3f} | "
                f"L_burst={epoch_record['burst_rate']:7.3f} | "
                f"L_sync={epoch_record['synchrony']:6.3f} | "
                f"L_reg={epoch_record['weight_reg']:6.4f} | "
                f"{epoch_record['wall_time_s']:.1f}s"
            )

    total_time = time.time() - t0
    print("-" * 72)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Final loss: {loss_history[-1]['total']:.4f}")
    print(
        f"Final firing rate: {loss_history[-1]['mean_fr_hz']:.2f} Hz "
        f"(target: {config.target_firing_rate_hz} Hz)"
    )
    print(
        f"Final burst rate: {loss_history[-1]['burst_rate_per_min']:.1f}/min "
        f"(target: {config.target_burst_rate_per_min}/min)"
    )

    return TrainingResult(
        W_exc=W_exc,
        W_inh=W_inh,
        loss_history=loss_history,
        config=config,
    )
