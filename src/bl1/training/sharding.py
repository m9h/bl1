"""Multi-GPU sharding primitives for the BL-1 differentiable trainer.

Scaffold for the >50K-neuron path: at 50K neurons the GPU performance
table shows the single-device simulation drops to 0.24x realtime, and at
100K it drops to 0.06x.  Custom kernels in :mod:`bl1.core.pallas_ops`
are correct but not faster than BCOO/cuSPARSE in practice, so the path
forward is data-parallel sharding across multiple devices, not better
single-device kernels.

This module provides the *construction* primitives -- mesh, partition
specs, and helpers to shard each of the four arrays the scan loop
actually depends on:

  * ``W_exc`` and ``W_inh`` -- (N_post, N_pre) dense or sparse weights
  * ``I_external``          -- (T, N) per-step external drive
  * ``NeuronState`` fields  -- v, u, spikes; each is (N,)

The natural mesh for cortical-culture simulation is **neuron-parallel**:
shard along the ``N`` axis (= output / post-synaptic neurons) and
replicate everything else.  Each device owns a contiguous slab of
neurons and computes their dynamics independently; synaptic input from
other devices arrives via the all-reduce JAX inserts automatically when
a sharded matmul is encountered inside ``jax.lax.scan``.

The scaffold is correctness-only.  Real GPU validation -- and any
tuning of axis assignment, prefetching, or all-reduce placement --
happens on DGX.  On a Mac / single-device CPU the helpers degrade to
the trivial single-device case; tests use the
``--xla_force_host_platform_device_count`` XLA flag to fake a
multi-device mesh.

Example::

    from bl1.training.sharding import make_neuron_mesh, shard_network

    mesh = make_neuron_mesh()                # uses all jax.devices()
    sharded = shard_network(
        mesh,
        W_exc=W_exc, W_inh=W_inh,
        I_external=I_external,
        init_state=init_state,
    )
    # then call simulate(...) with the sharded arrays
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Axis name used throughout: the contiguous slab of *neurons* a device owns.
NEURON_AXIS = "neuron"


def make_neuron_mesh(devices: list | None = None) -> Mesh:
    """Build a 1-D neuron-parallel mesh.

    Args:
        devices: Optional explicit list of devices.  Defaults to all
            devices visible to JAX (``jax.devices()``).

    Returns:
        ``jax.sharding.Mesh`` with one axis ``NEURON_AXIS`` of length
        ``len(devices)``.

    Raises:
        ValueError: if no devices are available.
    """
    if devices is None:
        devices = jax.devices()
    if not devices:
        raise ValueError("make_neuron_mesh: no JAX devices available")
    # jax.sharding.Mesh accepts a plain list/ndarray of Device objects.
    # Do NOT pass these through jnp -- Device objects aren't JAX arrays.
    return Mesh(list(devices), axis_names=(NEURON_AXIS,))


def neuron_spec_2d() -> PartitionSpec:
    """Spec for (N_post, N_pre)-shaped weight matrices.

    Shards the *row* (post-synaptic) axis across the neuron mesh;
    replicates the *column* (pre-synaptic) axis so every device sees
    the same input-spike vector.  This is the standard SPMD pattern
    for row-parallel matrix-vector products.
    """
    return PartitionSpec(NEURON_AXIS, None)


def neuron_spec_1d() -> PartitionSpec:
    """Spec for (N,)-shaped per-neuron state vectors."""
    return PartitionSpec(NEURON_AXIS)


def external_current_spec() -> PartitionSpec:
    """Spec for (T, N)-shaped per-step external drive.

    Time axis replicated, neuron axis sharded -- so each device owns
    its slab of neurons across the entire simulation.
    """
    return PartitionSpec(None, NEURON_AXIS)


def replicated_spec() -> PartitionSpec:
    """Spec for fully replicated arrays (e.g. scalar config tensors)."""
    return PartitionSpec()


def _put(arr: Array, mesh: Mesh, spec: PartitionSpec) -> Array:
    """Place ``arr`` on the mesh with the given partition spec."""
    return jax.device_put(arr, NamedSharding(mesh, spec))


class ShardedNetwork(NamedTuple):
    """Bundle of all the sharded arrays the trainer's scan body touches.

    Fields with ``None`` are not sharded (caller passed ``None`` for that
    array).  Keep this class light -- it carries arrays, not the model.
    """

    mesh: Mesh
    W_exc: Array | None
    W_inh: Array | None
    I_external: Array | None
    v: Array | None
    u: Array | None
    spikes: Array | None


def shard_network(
    mesh: Mesh,
    *,
    W_exc: Array | None = None,
    W_inh: Array | None = None,
    I_external: Array | None = None,
    init_state: Any = None,
) -> ShardedNetwork:
    """Apply neuron-axis sharding to every array the scan loop reads.

    Args:
        mesh: Output of :func:`make_neuron_mesh`.
        W_exc: ``(N, N)`` dense excitatory weights.
        W_inh: ``(N, N)`` dense inhibitory weights.
        I_external: ``(T, N)`` per-step external drive.
        init_state: Optional ``bl1.core.izhikevich.NeuronState`` (or any
            object with ``v``, ``u``, ``spikes`` array attributes).

    Returns:
        :class:`ShardedNetwork` with each provided array placed on the
        mesh under its appropriate spec.  Arrays passed as ``None``
        remain ``None`` in the result.
    """
    W_exc_s = _put(W_exc, mesh, neuron_spec_2d()) if W_exc is not None else None
    W_inh_s = _put(W_inh, mesh, neuron_spec_2d()) if W_inh is not None else None
    I_s = _put(I_external, mesh, external_current_spec()) if I_external is not None else None

    v_s = u_s = spk_s = None
    if init_state is not None:
        v_s = _put(init_state.v, mesh, neuron_spec_1d())
        u_s = _put(init_state.u, mesh, neuron_spec_1d())
        spk_s = _put(init_state.spikes, mesh, neuron_spec_1d())

    return ShardedNetwork(
        mesh=mesh,
        W_exc=W_exc_s,
        W_inh=W_inh_s,
        I_external=I_s,
        v=v_s,
        u=u_s,
        spikes=spk_s,
    )


def is_sharded(arr: Array, mesh: Mesh, spec: PartitionSpec) -> bool:
    """Return True iff ``arr`` is placed on ``mesh`` with ``spec``.

    Useful as a precondition check inside the trainer: after building a
    :class:`ShardedNetwork`, assert the resulting matmul actually sees
    the partition we expect.
    """
    sharding = getattr(arr, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return False
    return sharding.mesh is mesh and sharding.spec == spec
