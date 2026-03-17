"""Core data types for the BL-1 cortical culture simulator.

All state containers are JAX-compatible :class:`~typing.NamedTuple`
instances so they can be passed through ``jax.jit``, ``jax.lax.scan``,
and related transforms without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


# ---------------------------------------------------------------------------
# Static network parameters (immutable across a simulation run)
# ---------------------------------------------------------------------------

class NetworkParams(NamedTuple):
    """Static (non-plastic) network parameters."""

    positions: jnp.ndarray
    """(N, 2) or (N, 3) neuron soma positions in μm."""

    is_excitatory: jnp.ndarray
    """(N,) boolean mask — True for excitatory neurons."""

    W_exc: jnp.ndarray
    """(N, N) excitatory weight matrix (BCOO sparse)."""

    W_inh: jnp.ndarray
    """(N, N) inhibitory weight matrix (BCOO sparse)."""

    delays: jnp.ndarray
    """(N, N) axonal delays in timestep multiples (BCOO sparse,
    same sparsity pattern as the union of W_exc and W_inh)."""


# ---------------------------------------------------------------------------
# Mutable culture state (updated every timestep)
# ---------------------------------------------------------------------------

class CultureState(NamedTuple):
    """Full mutable state of a simulated cortical culture."""

    v: jnp.ndarray
    """(N,) membrane potential (mV)."""

    u: jnp.ndarray
    """(N,) Izhikevich recovery variable."""

    spikes: jnp.ndarray
    """(N,) boolean — True where a spike occurred this timestep."""

    g_ampa: jnp.ndarray
    """(N,) AMPA (excitatory) synaptic conductance."""

    g_gaba_a: jnp.ndarray
    """(N,) GABA_A (inhibitory) synaptic conductance."""

    stdp_pre_trace: jnp.ndarray
    """(N,) STDP pre-synaptic eligibility trace."""

    stdp_post_trace: jnp.ndarray
    """(N,) STDP post-synaptic eligibility trace."""

    W_exc: jnp.ndarray
    """(N, N) current excitatory weight matrix (plastic — updated by STDP)."""

    t: jnp.ndarray
    """Scalar — current simulation time in ms."""


# ---------------------------------------------------------------------------
# IzhikevichParams forward reference
# ---------------------------------------------------------------------------
# ``bl1.core.izhikevich`` defines the canonical ``IzhikevichParams``
# NamedTuple.  Because that module may not exist yet during early
# development we provide a local stand-in used only by ``Culture.create``
# when the core module is unavailable.

try:
    from bl1.core.izhikevich import IzhikevichParams, create_population  # type: ignore[import-untyped]
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

    class IzhikevichParams(NamedTuple):  # type: ignore[no-redef]
        """Minimal stand-in for ``bl1.core.izhikevich.IzhikevichParams``.

        Used only when the core module has not been implemented yet.
        Matches the standard Izhikevich (2003) parameterisation.
        """

        a: jnp.ndarray
        """(N,) time-scale of recovery variable."""

        b: jnp.ndarray
        """(N,) sensitivity of recovery variable to sub-threshold v."""

        c: jnp.ndarray
        """(N,) after-spike reset value of v (mV)."""

        d: jnp.ndarray
        """(N,) after-spike increment of u."""

    class _FallbackNeuronState(NamedTuple):
        v: jnp.ndarray
        u: jnp.ndarray
        spikes: jnp.ndarray

    def create_population(  # type: ignore[misc]
        key: jax.Array,
        n_neurons: int,
        ei_ratio: float = 0.8,
    ) -> Tuple["IzhikevichParams", "_FallbackNeuronState", jnp.ndarray]:
        """Fallback population factory (matches core.izhikevich signature).

        Returns (params, state, is_excitatory).
        """
        n_exc = int(n_neurons * ei_ratio)
        is_exc = jnp.concatenate([
            jnp.ones(n_exc, dtype=jnp.bool_),
            jnp.zeros(n_neurons - n_exc, dtype=jnp.bool_),
        ])

        key_a, key_b = jax.random.split(key)
        r_exc = jax.random.uniform(key_a, (n_exc,))
        a_exc = 0.02 * jnp.ones(n_exc)
        b_exc = 0.2 * jnp.ones(n_exc)
        c_exc = -65.0 + 15.0 * r_exc ** 2
        d_exc = 8.0 - 6.0 * r_exc ** 2

        n_inh = n_neurons - n_exc
        r_inh = jax.random.uniform(key_b, (n_inh,))
        a_inh = 0.02 + 0.08 * r_inh
        b_inh = 0.25 - 0.05 * r_inh
        c_inh = -65.0 * jnp.ones(n_inh)
        d_inh = 2.0 * jnp.ones(n_inh)

        params = IzhikevichParams(
            a=jnp.concatenate([a_exc, a_inh]),
            b=jnp.concatenate([b_exc, b_inh]),
            c=jnp.concatenate([c_exc, c_inh]),
            d=jnp.concatenate([d_exc, d_inh]),
        )
        b_all = params.b
        v0 = -65.0 * jnp.ones(n_neurons)
        state = _FallbackNeuronState(v=v0, u=b_all * v0, spikes=jnp.zeros(n_neurons, dtype=jnp.bool_))
        return params, state, is_exc


# ---------------------------------------------------------------------------
# High-level culture factory
# ---------------------------------------------------------------------------

from bl1.network.topology import (
    build_connectivity,
    place_neurons,
    place_neurons_layered,
    place_neurons_spheroid,
)


@dataclass
class Culture:
    """Convenience factory for creating fully-initialised cultures.

    This is **not** a NamedTuple — it is a plain dataclass / namespace that
    provides static helper methods.  The actual simulation state is held in
    :class:`NetworkParams` and :class:`CultureState`.
    """

    @staticmethod
    def create(
        key: jax.Array,
        n_neurons: int,
        ei_ratio: float = 0.8,
        neuron_model: str = "izhikevich",
        substrate_um: Tuple[float, float] = (3000.0, 3000.0),
        substrate_3d: Optional[Tuple[float, float, float]] = None,
        placement: str = "uniform",
        connectivity: str = "distance_dependent",
        lambda_um: float = 200.0,
        p_max: float = 0.21,
        g_exc: float = 0.05,
        g_inh: float = 0.20,
        dt: float = 0.5,
        spheroid_radius_um: float = 500.0,
        spheroid_center_um: Optional[Tuple[float, float, float]] = None,
        layer_depths_um: Optional[tuple] = None,
        layer_densities: Optional[tuple] = None,
    ) -> Tuple[NetworkParams, CultureState, IzhikevichParams]:
        """Create a new cortical culture from scratch.

        Supports 2D (flat substrate) and 3D (organoid / spheroid / layered
        cortical) cultures.  The default behaviour is identical to the
        original 2D-only implementation for full backward compatibility.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        n_neurons : int
            Total number of neurons.
        ei_ratio : float
            Fraction of neurons that are excitatory (default 0.8).
        neuron_model : str
            Currently only ``"izhikevich"`` is supported.
        substrate_um : tuple of float
            (width, height) of the MEA substrate in μm.
        substrate_3d : tuple of float or None
            (width, height, depth) in μm for 3-D uniform placement.
            Only used when *placement* is ``"uniform"``.
        placement : str
            Placement strategy:

            - ``"uniform"`` (default) — uniform random placement.  Uses
              *substrate_3d* if provided, otherwise *substrate_um* for 2D.
            - ``"spheroid"`` — uniform within a 3D sphere (organoid model).
            - ``"layered"`` — layered 3D cortical structure with six layers.

        connectivity : str
            Connectivity model — ``"distance_dependent"`` (default).
        lambda_um : float
            Spatial length constant for connection probability.
        p_max : float
            Maximum connection probability at zero distance.
        g_exc : float
            Baseline excitatory synaptic weight.
        g_inh : float
            Baseline inhibitory synaptic weight.
        dt : float
            Simulation timestep in ms.
        spheroid_radius_um : float
            Radius of the spheroid in μm (only for ``placement="spheroid"``).
        spheroid_center_um : tuple of float or None
            Centre of the spheroid (only for ``placement="spheroid"``).
        layer_depths_um : tuple of float or None
            Per-layer thicknesses (only for ``placement="layered"``).
        layer_densities : tuple of float or None
            Per-layer relative densities (only for ``placement="layered"``).

        Returns
        -------
        net_params : NetworkParams
        state : CultureState
        izh_params : IzhikevichParams
        """
        if neuron_model != "izhikevich":
            raise ValueError(f"Unsupported neuron model: {neuron_model!r}")
        if connectivity != "distance_dependent":
            raise ValueError(f"Unsupported connectivity: {connectivity!r}")

        # --- PRNG key splitting ------------------------------------------
        k_place, k_pop, k_conn = jax.random.split(key, 3)

        # 1. Place neurons on the substrate or in a 3D volume
        if placement == "uniform":
            positions = place_neurons(
                k_place, n_neurons, substrate_um, substrate_3d=substrate_3d,
            )
        elif placement == "spheroid":
            positions = place_neurons_spheroid(
                k_place, n_neurons,
                radius_um=spheroid_radius_um,
                center_um=spheroid_center_um,
            )
        elif placement == "layered":
            kwargs_layered: dict = dict(
                substrate_um=substrate_um,
            )
            if layer_depths_um is not None:
                kwargs_layered["layer_depths_um"] = layer_depths_um
            if layer_densities is not None:
                kwargs_layered["layer_densities"] = layer_densities
            positions = place_neurons_layered(
                k_place, n_neurons, **kwargs_layered,
            )
        else:
            raise ValueError(
                f"Unsupported placement: {placement!r}. "
                f"Expected 'uniform', 'spheroid', or 'layered'."
            )

        # 2. Create neuron population (E/I assignment + Izhikevich params)
        izh_params, _init_state, is_excitatory = create_population(k_pop, n_neurons, ei_ratio)

        # 3. Build distance-dependent connectivity
        W_exc, W_inh, delays = build_connectivity(
            k_conn,
            positions,
            is_excitatory,
            lambda_um=lambda_um,
            p_max=p_max,
            g_exc=g_exc,
            g_inh=g_inh,
            dt=dt,
        )

        # --- assemble NetworkParams --------------------------------------
        net_params = NetworkParams(
            positions=positions,
            is_excitatory=is_excitatory,
            W_exc=W_exc,
            W_inh=W_inh,
            delays=delays,
        )

        # --- initialise CultureState to biologically-sensible defaults ---
        N = n_neurons
        v0 = -65.0 * jnp.ones(N)
        u0 = izh_params.b * v0
        state = CultureState(
            v=v0,
            u=u0,
            spikes=jnp.zeros(N, dtype=jnp.bool_),
            g_ampa=jnp.zeros(N),
            g_gaba_a=jnp.zeros(N),
            stdp_pre_trace=jnp.zeros(N),
            stdp_post_trace=jnp.zeros(N),
            W_exc=W_exc,
            t=jnp.array(0.0),
        )

        return net_params, state, izh_params
