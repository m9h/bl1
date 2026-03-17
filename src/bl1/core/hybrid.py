"""Hybrid multi-model neuron simulation.

Allows mixing different neuron models (Izhikevich, AdEx) in the same
network. Each population is stepped with its own model, but they share
a common synapse model and connectivity.

This enables:
- Running mostly Izhikevich neurons for speed, with a small AdEx subset
  for detailed biophysics in a region of interest
- Validating that point-neuron results hold with more detailed models
- Gradual migration between model types

Usage:
    from bl1.core.hybrid import HybridPopulation, hybrid_step

    pop = HybridPopulation.create(
        key=key,
        n_neurons=1000,
        model_assignments=["izhikevich"] * 800 + ["adex"] * 200,
        ei_ratio=0.8,
    )
    new_state, spikes = hybrid_step(pop, I_ext, dt=0.5)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    izhikevich_step,
    create_population,
)
from bl1.core.adex import (
    AdExParams,
    AdExState,
    adex_step,
    create_adex_population,
)


class HybridState(NamedTuple):
    """Combined state for a mixed-model population."""

    # Izhikevich neurons
    izh_v: Array  # (N_izh,) membrane potential
    izh_u: Array  # (N_izh,) recovery variable
    # AdEx neurons
    adex_v: Array  # (N_adex,) membrane potential
    adex_w: Array  # (N_adex,) adaptation current
    # Combined
    spikes: Array  # (N_total,) boolean spikes for ALL neurons


class HybridParams(NamedTuple):
    """Combined parameters for a mixed-model population."""

    # Izhikevich parameters (for neurons using that model)
    izh_params: IzhikevichParams
    # AdEx parameters (for neurons using that model)
    adex_params: AdExParams
    # Indexing
    izh_indices: Array  # (N_izh,) indices into the full population
    adex_indices: Array  # (N_adex,) indices into the full population
    n_total: int  # Total neuron count
    is_excitatory: Array  # (N_total,) boolean


class HybridPopulation:
    """Factory for creating hybrid neuron populations."""

    @staticmethod
    def create(
        key: Array,
        n_neurons: int,
        model_assignments: list[str] | None = None,
        n_adex: int = 0,
        ei_ratio: float = 0.8,
    ) -> tuple[HybridParams, HybridState]:
        """Create a hybrid population with mixed neuron models.

        Args:
            key: JAX PRNG key
            n_neurons: Total number of neurons
            model_assignments: Optional list of "izhikevich" or "adex" per neuron.
                If None, uses n_adex to determine the split.
            n_adex: Number of AdEx neurons (used if model_assignments is None).
                AdEx neurons are placed at the end of the population.
            ei_ratio: Fraction excitatory

        Returns:
            (HybridParams, HybridState) ready for simulation
        """
        if model_assignments is None:
            n_izh = n_neurons - n_adex
            model_assignments = ["izhikevich"] * n_izh + ["adex"] * n_adex

        assert len(model_assignments) == n_neurons, (
            f"model_assignments length ({len(model_assignments)}) "
            f"must match n_neurons ({n_neurons})"
        )

        izh_mask = [m == "izhikevich" for m in model_assignments]
        adex_mask = [m == "adex" for m in model_assignments]

        # Validate assignments
        for i, m in enumerate(model_assignments):
            if m not in ("izhikevich", "adex"):
                raise ValueError(
                    f"Unknown model '{m}' at index {i}. "
                    f"Supported: 'izhikevich', 'adex'"
                )

        n_izh = sum(izh_mask)
        n_adex_actual = sum(adex_mask)

        # Build index arrays -- use Python ints for the `size` argument
        # so JAX knows array shapes at trace time.
        izh_indices = jnp.where(jnp.array(izh_mask), size=n_izh)[0]
        adex_indices = jnp.where(jnp.array(adex_mask), size=n_adex_actual)[0]

        # Create sub-populations
        k1, k2 = jax.random.split(key)

        if n_izh > 0:
            izh_params, izh_state, izh_exc = create_population(
                k1, n_izh, ei_ratio
            )
        else:
            # Empty Izhikevich sub-population
            izh_params = IzhikevichParams(
                a=jnp.array([]),
                b=jnp.array([]),
                c=jnp.array([]),
                d=jnp.array([]),
            )
            izh_state = NeuronState(
                v=jnp.array([]),
                u=jnp.array([]),
                spikes=jnp.array([], dtype=jnp.bool_),
            )
            izh_exc = jnp.array([], dtype=jnp.bool_)

        if n_adex_actual > 0:
            adex_params, adex_state, adex_exc = create_adex_population(
                k2, n_adex_actual, ei_ratio
            )
        else:
            # Empty AdEx sub-population
            adex_params = AdExParams(
                C=jnp.array([]),
                g_L=jnp.array([]),
                E_L=jnp.array([]),
                delta_T=jnp.array([]),
                V_T=jnp.array([]),
                V_reset=jnp.array([]),
                V_peak=jnp.array([]),
                a=jnp.array([]),
                b=jnp.array([]),
                tau_w=jnp.array([]),
            )
            adex_state = AdExState(
                v=jnp.array([]),
                w=jnp.array([]),
                spikes=jnp.array([], dtype=jnp.bool_),
            )
            adex_exc = jnp.array([], dtype=jnp.bool_)

        # Combined excitatory mask
        is_excitatory = jnp.zeros(n_neurons, dtype=jnp.bool_)
        if n_izh > 0:
            is_excitatory = is_excitatory.at[izh_indices].set(izh_exc)
        if n_adex_actual > 0:
            is_excitatory = is_excitatory.at[adex_indices].set(adex_exc)

        params = HybridParams(
            izh_params=izh_params,
            adex_params=adex_params,
            izh_indices=izh_indices,
            adex_indices=adex_indices,
            n_total=n_neurons,
            is_excitatory=is_excitatory,
        )

        state = HybridState(
            izh_v=izh_state.v,
            izh_u=izh_state.u,
            adex_v=adex_state.v,
            adex_w=adex_state.w,
            spikes=jnp.zeros(n_neurons, dtype=jnp.bool_),
        )

        return params, state


def _izh_substep(
    izh_v: Array,
    izh_u: Array,
    izh_params: IzhikevichParams,
    I_izh: Array,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Step the Izhikevich sub-population. Works for empty arrays too."""
    izh_state_in = NeuronState(
        v=izh_v,
        u=izh_u,
        spikes=jnp.zeros_like(izh_v, dtype=jnp.bool_),
    )
    new_izh = izhikevich_step(izh_state_in, izh_params, I_izh, dt)
    return new_izh.v, new_izh.u, new_izh.spikes


def _adex_substep(
    adex_v: Array,
    adex_w: Array,
    adex_params: AdExParams,
    I_adex: Array,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Step the AdEx sub-population. Works for empty arrays too."""
    adex_state_in = AdExState(
        v=adex_v,
        w=adex_w,
        spikes=jnp.zeros_like(adex_v, dtype=jnp.bool_),
    )
    new_adex = adex_step(adex_state_in, adex_params, I_adex, dt)
    return new_adex.v, new_adex.w, new_adex.spikes


@jax.jit
def hybrid_step(
    state: HybridState,
    params: HybridParams,
    I_ext: Array,
    dt: float = 0.5,
) -> HybridState:
    """Step a hybrid population forward by one timestep.

    Each sub-population is advanced with its own model, then spikes
    are gathered into the combined spike vector.

    This function is JIT-compatible. Empty sub-populations (zero-length
    index arrays) produce zero-length outputs from gather/scatter, which
    are no-ops in JAX.

    Args:
        state: Current HybridState
        params: HybridParams with sub-population parameters and indices
        I_ext: (N_total,) external current for all neurons
        dt: Timestep in ms

    Returns:
        Updated HybridState
    """
    # Extract per-model currents via gather
    I_izh = I_ext[params.izh_indices]
    I_adex = I_ext[params.adex_indices]

    # Step Izhikevich sub-population
    new_izh_v, new_izh_u, izh_spikes = _izh_substep(
        state.izh_v, state.izh_u, params.izh_params, I_izh, dt
    )

    # Step AdEx sub-population
    new_adex_v, new_adex_w, adex_spikes = _adex_substep(
        state.adex_v, state.adex_w, params.adex_params, I_adex, dt
    )

    # Gather spikes into combined vector via scatter (no-op for empty indices).
    # Use zeros_like(state.spikes) instead of zeros(n_total) because the shape
    # of state.spikes is statically known during JIT tracing, whereas n_total
    # stored as an int in HybridParams would be traced as a dynamic value.
    combined_spikes = jnp.zeros_like(state.spikes)
    combined_spikes = combined_spikes.at[params.izh_indices].set(izh_spikes)
    combined_spikes = combined_spikes.at[params.adex_indices].set(adex_spikes)

    return HybridState(
        izh_v=new_izh_v,
        izh_u=new_izh_u,
        adex_v=new_adex_v,
        adex_w=new_adex_w,
        spikes=combined_spikes,
    )


def get_membrane_potential(state: HybridState, params: HybridParams) -> Array:
    """Reconstruct the full (N_total,) membrane potential vector.

    Useful for diagnostics and plotting. Combines Izhikevich v and AdEx v
    back into their original positions in the population.

    Args:
        state: Current HybridState
        params: HybridParams with index mappings

    Returns:
        (N_total,) membrane potential for all neurons.
    """
    # Use zeros_like(state.spikes) to get the right shape without relying
    # on n_total, which may be a traced value inside JIT.
    v = jnp.zeros_like(state.spikes, dtype=jnp.float32)
    v = v.at[params.izh_indices].set(state.izh_v)
    v = v.at[params.adex_indices].set(state.adex_v)
    return v
