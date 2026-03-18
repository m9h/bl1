"""Event-driven sparse synaptic computation for large-scale networks.

Instead of computing ``W @ spikes`` (which iterates over all nonzeros via
BCOO sparse matmul), this module uses direct array operations —
multiply-and-scatter — that bypass BCOO's internal indexing overhead.

At typical cortical firing rates (1-10 Hz) fewer than 1% of neurons spike
per 0.5 ms timestep.  The multiply ``W_data * spikes[W_cols]`` naturally
zeros out contributions from non-spiking pre-synaptic neurons, so only the
spiking fraction produces meaningful floating-point work.  The subsequent
``segment_sum`` is a well-optimized GPU primitive.

Usage
-----
At network creation time, call :func:`bcoo_to_raw` once per weight matrix
to extract the three flat arrays.  Then, inside the simulation scan loop,
replace ``weights @ spikes`` calls with :func:`fast_sparse_input`.

Backward compatibility
----------------------
All existing code paths using BCOO ``@`` remain untouched.  The fast
sparse path is opt-in via the ``use_fast_sparse`` flag in the integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import Array
from jax.experimental.sparse import BCOO

# ---------------------------------------------------------------------------
# Raw weight arrays container
# ---------------------------------------------------------------------------


class RawSparseWeights(NamedTuple):
    """Pre-extracted COO arrays for :func:`fast_sparse_input`.

    Attributes
    ----------
    data : Array, shape (nnz,)
        Non-zero weight values.
    rows : Array, shape (nnz,)
        Post-synaptic (row) indices for each non-zero.
    cols : Array, shape (nnz,)
        Pre-synaptic (column) indices for each non-zero.
    n_post : int
        Number of post-synaptic neurons (= number of rows in the
        original weight matrix).
    """

    data: Array
    rows: Array
    cols: Array
    n_post: int


# ---------------------------------------------------------------------------
# BCOO -> raw conversion
# ---------------------------------------------------------------------------


def bcoo_to_raw(W: BCOO) -> RawSparseWeights:
    """Extract flat COO arrays from a BCOO sparse matrix.

    Parameters
    ----------
    W : BCOO
        A 2-D BCOO sparse matrix of shape ``(N_post, N_pre)``.

    Returns
    -------
    RawSparseWeights
        Named tuple with ``(data, rows, cols, n_post)`` suitable for
        :func:`fast_sparse_input`.
    """
    return RawSparseWeights(
        data=W.data,
        rows=W.indices[:, 0],
        cols=W.indices[:, 1],
        n_post=W.shape[0],
    )


# ---------------------------------------------------------------------------
# Fast sparse matmul replacement
# ---------------------------------------------------------------------------


def fast_sparse_input(
    W_data: Array,
    W_rows: Array,
    W_cols: Array,
    spikes: Array,
    n_post: int,
) -> Array:
    """Compute synaptic input equivalent to ``BCOO_matrix @ spikes``.

    This is faster than BCOO matmul because it avoids BCOO's internal
    indexing and dispatch overhead, using only two well-optimized
    primitives: a gather-multiply and a segment-sum scatter.

    The multiply ``W_data * spikes[W_cols]`` naturally zeros out
    contributions from non-spiking neurons, so the effective work
    scales with the number of active synapses.

    Parameters
    ----------
    W_data : Array, shape ``(nnz,)``
        Non-zero weight values.
    W_rows : Array, shape ``(nnz,)``
        Post-synaptic (row) indices.
    W_cols : Array, shape ``(nnz,)``
        Pre-synaptic (column) indices.
    spikes : Array, shape ``(N_pre,)``
        Pre-synaptic spike vector (float; 0.0 for no spike).
    n_post : int
        Number of post-synaptic neurons.

    Returns
    -------
    Array, shape ``(n_post,)``
        Synaptic input per post-synaptic neuron, identical to
        ``W @ spikes`` for the corresponding BCOO matrix.
    """
    # Gather pre-synaptic spike amplitudes and multiply by weights
    weighted = W_data * spikes[W_cols]
    # Scatter-add into post-synaptic neuron accumulators
    return jax.ops.segment_sum(weighted, W_rows, num_segments=n_post)


def fast_sparse_input_raw(raw: RawSparseWeights, spikes: Array) -> Array:
    """Convenience wrapper: call :func:`fast_sparse_input` from a
    :class:`RawSparseWeights` container.
    """
    return fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)
