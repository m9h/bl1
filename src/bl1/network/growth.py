"""Network growth model — simplified NETMORPH-inspired maturation.

Simulates the development of cortical culture connectivity over days in vitro (DIV):

- DIV 1-7:   Axon/dendrite outgrowth, random sparse connections forming
- DIV 7-14:  Connection density increasing, first global bursts
- DIV 14-28: Activity-dependent refinement, small-world emergence
- DIV 28+:   Semi-mature topology with rich-club hubs

This can be bypassed by directly instantiating a mature network via
build_connectivity() in topology.py. The growth model provides a more
biologically realistic trajectory for studying development.

Reference: Van Ooyen et al. (2003), Koene et al. (2009) NETMORPH
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class GrowthParams(NamedTuple):
    """Parameters controlling network growth trajectory."""

    # Connection probability envelope
    p_max_initial: float = 0.01       # Initial max connection prob (DIV 1)
    p_max_mature: float = 0.21        # Mature max connection prob (DIV 28)
    lambda_initial_um: float = 100.0  # Initial length constant (short range)
    lambda_mature_um: float = 200.0   # Mature length constant

    # Growth rate
    growth_rate: float = 0.15         # Logistic growth rate per DIV
    midpoint_div: float = 10.0        # Midpoint of connectivity growth curve

    # Weights
    g_exc_mature: float = 0.05        # Mature excitatory weight
    g_inh_mature: float = 0.20        # Mature inhibitory weight

    # Network refinement (DIV 14+)
    hub_fraction: float = 0.2         # Fraction of neurons that become hubs
    hub_weight_boost: float = 2.0     # Weight multiplier for hub connections


class GrowthState(NamedTuple):
    """State of the growing network at a particular DIV."""

    div: float                    # Current days in vitro
    W_exc: Array                  # Current excitatory weights
    W_inh: Array                  # Current inhibitory weights
    connectivity_fraction: float  # Current fraction of possible connections


def init_growth(
    key: Array,
    positions: Array,
    is_excitatory: Array,
    params: GrowthParams = GrowthParams(),
) -> GrowthState:
    """Initialize a growth state at DIV 0 with no connections."""
    N = positions.shape[0]
    W_exc = jnp.zeros((N, N))
    W_inh = jnp.zeros((N, N))
    return GrowthState(div=0.0, W_exc=W_exc, W_inh=W_inh, connectivity_fraction=0.0)


def _logistic(div: float, growth_rate: float, midpoint: float) -> float:
    """Logistic growth factor in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-growth_rate * (div - midpoint)))


def grow_to_div(
    key: Array,
    positions: Array,
    is_excitatory: Array,
    target_div: float,
    params: GrowthParams = GrowthParams(),
    current_state: GrowthState | None = None,
) -> GrowthState:
    """Grow the network to the specified DIV.

    Uses a logistic growth curve for connection probability and
    distance-dependent connectivity at each stage.

    Args:
        key: JAX PRNG key
        positions: (N, 2) neuron positions in um
        is_excitatory: (N,) boolean mask
        target_div: Target days in vitro (e.g., 28)
        params: Growth parameters
        current_state: Optional starting state (if None, starts from DIV 0)

    Returns:
        GrowthState at the target DIV
    """
    # Move to numpy for efficient connectivity construction
    pos_np = np.asarray(positions)
    is_exc_np = np.asarray(is_excitatory).astype(bool)
    N = pos_np.shape[0]

    # --- Logistic growth factor -------------------------------------------
    f = _logistic(target_div, params.growth_rate, params.midpoint_div)

    # --- Interpolate parameters -------------------------------------------
    p_max = params.p_max_initial + f * (params.p_max_mature - params.p_max_initial)
    lambda_um = params.lambda_initial_um + f * (params.lambda_mature_um - params.lambda_initial_um)
    g_exc = f * params.g_exc_mature
    g_inh = f * params.g_inh_mature

    # --- Pairwise distances (N, N) ----------------------------------------
    diff = pos_np[:, None, :] - pos_np[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    # --- Connection probabilities -----------------------------------------
    prob = p_max * np.exp(-dist / lambda_um)
    np.fill_diagonal(prob, 0.0)  # no self-connections

    # --- Draw connections using numpy RNG seeded from JAX key -------------
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    connected = rng.random((N, N)) < prob

    # --- Split into E / I masks -------------------------------------------
    exc_mask = connected & is_exc_np[:, None]   # pre-synaptic is excitatory
    inh_mask = connected & (~is_exc_np)[:, None]  # pre-synaptic is inhibitory

    # --- Weights with +/-10% uniform noise --------------------------------
    noise_exc = 1.0 + 0.1 * (2.0 * rng.random((N, N)) - 1.0)
    noise_inh = 1.0 + 0.1 * (2.0 * rng.random((N, N)) - 1.0)

    W_exc_np = np.where(exc_mask, g_exc * noise_exc, 0.0)
    W_inh_np = np.where(inh_mask, g_inh * noise_inh, 0.0)

    # --- Hub formation for DIV > 14 ---------------------------------------
    if target_div > 14.0:
        n_hubs = max(1, int(N * params.hub_fraction))
        hub_indices = rng.choice(N, size=n_hubs, replace=False)

        # Boost outgoing weights from hub neurons
        W_exc_np[hub_indices, :] *= params.hub_weight_boost
        W_inh_np[hub_indices, :] *= params.hub_weight_boost

    # --- Compute connectivity fraction ------------------------------------
    total_possible = N * (N - 1)  # exclude diagonal
    n_connections = int(connected.sum())
    connectivity_fraction = n_connections / total_possible if total_possible > 0 else 0.0

    # --- Convert to JAX arrays --------------------------------------------
    W_exc = jnp.array(W_exc_np, dtype=jnp.float32)
    W_inh = jnp.array(W_inh_np, dtype=jnp.float32)

    return GrowthState(
        div=target_div,
        W_exc=W_exc,
        W_inh=W_inh,
        connectivity_fraction=connectivity_fraction,
    )


def mature_network(
    key: Array,
    positions: Array,
    is_excitatory: Array,
    target_div: float = 28.0,
    accelerated: bool = True,
    params: GrowthParams = GrowthParams(),
) -> tuple[Array, Array]:
    """Convenience: grow network and return (W_exc, W_inh) ready for simulation.

    Args:
        key: JAX PRNG key
        positions: (N, 2) neuron positions in um
        is_excitatory: (N,) boolean mask
        target_div: Target days in vitro (default 28.0)
        accelerated: If True, jump directly to target_div (no intermediate steps)
        params: Growth parameters

    Returns:
        (W_exc, W_inh) dense weight matrices as JAX arrays
    """
    state = grow_to_div(key, positions, is_excitatory, target_div, params)
    return state.W_exc, state.W_inh
