"""Distance-dependent connectivity for cortical culture networks.

Implements spatial neuron placement and biologically-motivated connectivity
where connection probability decays exponentially with inter-somatic distance
(Lonardoni et al. 2017, Wagenaar et al. 2006).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO


# ---------------------------------------------------------------------------
# Neuron placement
# ---------------------------------------------------------------------------

def place_neurons(
    key: jax.Array,
    n_neurons: int,
    substrate_um: Tuple[float, float] = (3000.0, 3000.0),
) -> jnp.ndarray:
    """Uniformly distribute neurons on a 2-D rectangular substrate.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n_neurons : int
        Number of neurons to place.
    substrate_um : tuple of float
        (width, height) of the culture substrate in micrometres.

    Returns
    -------
    positions : jnp.ndarray, shape (n_neurons, 2)
        (x, y) positions in μm.
    """
    return jax.random.uniform(key, shape=(n_neurons, 2)) * jnp.array(substrate_um)


# ---------------------------------------------------------------------------
# Distance-dependent connectivity (dense path, <10 K neurons)
# ---------------------------------------------------------------------------

def _build_connectivity_dense(
    key: jax.Array,
    positions: jnp.ndarray,
    is_excitatory: jnp.ndarray,
    lambda_um: float = 200.0,
    p_max: float = 0.21,
    g_exc: float = 0.05,
    g_inh: float = 0.20,
    v_axon_um_per_ms: float = 300.0,
    dt: float = 0.5,
) -> Tuple[BCOO, BCOO, BCOO]:
    """Build connectivity for small networks using a full distance matrix.

    Suitable for N < ~10 000.  For larger networks use the spatial-hashing
    path (:func:`_build_connectivity_spatial`).

    Returns
    -------
    W_exc : BCOO  – excitatory weight matrix (N, N)
    W_inh : BCOO  – inhibitory weight matrix (N, N)
    delays : BCOO – axonal delay matrix (N, N), in timesteps
    """
    N = positions.shape[0]

    # --- distances (N, N) ------------------------------------------------
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))           # (N, N)

    # --- connection probabilities ----------------------------------------
    prob = p_max * jnp.exp(-dist / lambda_um)
    prob = prob.at[jnp.diag_indices(N)].set(0.0)           # no self-connections

    key_conn, key_w = jax.random.split(key)
    connected = jax.random.uniform(key_conn, shape=(N, N)) < prob  # bool

    # --- split into E / I masks -----------------------------------------
    exc_mask = connected & is_excitatory[:, None]           # pre is excitatory
    inh_mask = connected & ~is_excitatory[:, None]          # pre is inhibitory

    # --- weights (with ±10 % uniform noise) ------------------------------
    key_we, key_wi = jax.random.split(key_w)
    noise_exc = 1.0 + 0.1 * (2.0 * jax.random.uniform(key_we, (N, N)) - 1.0)
    noise_inh = 1.0 + 0.1 * (2.0 * jax.random.uniform(key_wi, (N, N)) - 1.0)

    W_exc_dense = jnp.where(exc_mask, g_exc * noise_exc, 0.0)
    W_inh_dense = jnp.where(inh_mask, g_inh * noise_inh, 0.0)

    # --- delays in timestep multiples ------------------------------------
    delay_ms = dist / v_axon_um_per_ms
    delay_steps = jnp.maximum(jnp.round(delay_ms / dt), 1.0)
    delay_dense = jnp.where(connected, delay_steps, 0.0)

    # --- convert to BCOO -------------------------------------------------
    W_exc_sp = BCOO.fromdense(W_exc_dense)
    W_inh_sp = BCOO.fromdense(W_inh_dense)
    delays_sp = BCOO.fromdense(delay_dense)

    return W_exc_sp, W_inh_sp, delays_sp


# ---------------------------------------------------------------------------
# Distance-dependent connectivity (spatial-hashing path, ≥10 K neurons)
# ---------------------------------------------------------------------------

def _build_connectivity_spatial(
    key: jax.Array,
    positions: jnp.ndarray,
    is_excitatory: jnp.ndarray,
    lambda_um: float = 200.0,
    p_max: float = 0.21,
    g_exc: float = 0.05,
    g_inh: float = 0.20,
    v_axon_um_per_ms: float = 300.0,
    dt: float = 0.5,
) -> Tuple[BCOO, BCOO, BCOO]:
    """Build connectivity for large networks via spatial binning in NumPy.

    Neurons are assigned to square bins of side ``lambda_um``.  For each
    neuron we only evaluate potential connections to neurons whose bins fall
    within a ``cutoff = 3 * lambda_um`` radius, making the algorithm
    O(N * k) with k ~ average neighbours inside the cutoff.

    All heavy lifting is done in NumPy; the result is converted to JAX
    BCOO at the very end.

    Returns
    -------
    W_exc, W_inh, delays : BCOO (N, N) each.
    """
    # Move to numpy for the spatial-hash loop.
    pos_np: np.ndarray = np.asarray(positions)
    is_exc_np: np.ndarray = np.asarray(is_excitatory).astype(bool)
    N = pos_np.shape[0]

    cutoff = 3.0 * lambda_um
    bin_size = lambda_um  # square bin side

    # --- build spatial hash ----------------------------------------------
    bin_idx = (pos_np / bin_size).astype(np.int32)           # (N, 2)
    from collections import defaultdict
    bins: dict[Tuple[int, int], list[int]] = defaultdict(list)
    for i in range(N):
        bins[(int(bin_idx[i, 0]), int(bin_idx[i, 1]))].append(i)

    # Pre-compute the set of relative bin offsets within cutoff
    max_offset = int(np.ceil(cutoff / bin_size))
    offsets = [
        (dx, dy)
        for dx in range(-max_offset, max_offset + 1)
        for dy in range(-max_offset, max_offset + 1)
        if np.hypot(dx, dy) * bin_size <= cutoff + bin_size  # generous
    ]

    # --- RNG setup (numpy for speed) -------------------------------------
    # Consume the JAX key to seed a numpy RNG.
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    # Pre-allocate COO lists
    row_exc: list[np.ndarray] = []
    col_exc: list[np.ndarray] = []
    val_exc: list[np.ndarray] = []

    row_inh: list[np.ndarray] = []
    col_inh: list[np.ndarray] = []
    val_inh: list[np.ndarray] = []

    row_del: list[np.ndarray] = []
    col_del: list[np.ndarray] = []
    val_del: list[np.ndarray] = []

    # --- iterate over bins -----------------------------------------------
    for (bx, by), src_neurons in bins.items():
        src_arr = np.array(src_neurons, dtype=np.int64)
        src_pos = pos_np[src_arr]  # (|src|, 2)

        # Gather candidate targets from neighbouring bins
        tgt_list: list[int] = []
        for dx, dy in offsets:
            nb = (bx + dx, by + dy)
            if nb in bins:
                tgt_list.extend(bins[nb])
        if not tgt_list:
            continue

        tgt_arr = np.array(tgt_list, dtype=np.int64)
        tgt_pos = pos_np[tgt_arr]  # (|tgt|, 2)

        # Pairwise distances between src and tgt  (|src|, |tgt|)
        diff = src_pos[:, None, :] - tgt_pos[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Connection probability
        prob = p_max * np.exp(-dist / lambda_um)

        # Zero out self-connections
        # src_arr[:, None] == tgt_arr[None, :] produces (|src|, |tgt|) bool
        self_mask = src_arr[:, None] == tgt_arr[None, :]
        prob[self_mask] = 0.0

        # Zero out pairs beyond cutoff (already low prob, but be explicit)
        prob[dist > cutoff] = 0.0

        # Draw connections
        rand = rng.random(prob.shape)
        connected = rand < prob

        if not np.any(connected):
            continue

        si, ti = np.nonzero(connected)
        rows = src_arr[si]
        cols = tgt_arr[ti]
        dists = dist[si, ti]

        # Delays (timesteps, minimum 1)
        delay_steps = np.maximum(np.round(dists / v_axon_um_per_ms / dt), 1.0)

        # Split E / I by pre-synaptic type
        pre_exc = is_exc_np[rows]

        # Weight noise (±10 %)
        noise = 1.0 + 0.1 * (2.0 * rng.random(rows.shape[0]) - 1.0)

        # -- Excitatory synapses
        if np.any(pre_exc):
            idx_e = np.nonzero(pre_exc)[0]
            row_exc.append(rows[idx_e])
            col_exc.append(cols[idx_e])
            val_exc.append(g_exc * noise[idx_e])
            row_del.append(rows[idx_e])
            col_del.append(cols[idx_e])
            val_del.append(delay_steps[idx_e])

        # -- Inhibitory synapses
        pre_inh = ~pre_exc
        if np.any(pre_inh):
            idx_i = np.nonzero(pre_inh)[0]
            row_inh.append(rows[idx_i])
            col_inh.append(cols[idx_i])
            val_inh.append(g_inh * noise[idx_i])
            row_del.append(rows[idx_i])
            col_del.append(cols[idx_i])
            val_del.append(delay_steps[idx_i])

    # --- assemble COO and convert to BCOO --------------------------------
    def _to_bcoo(
        row_parts: list[np.ndarray],
        col_parts: list[np.ndarray],
        val_parts: list[np.ndarray],
    ) -> BCOO:
        if row_parts:
            rows_np = np.concatenate(row_parts)
            cols_np = np.concatenate(col_parts)
            vals_np = np.concatenate(val_parts)
        else:
            rows_np = np.empty(0, dtype=np.int64)
            cols_np = np.empty(0, dtype=np.int64)
            vals_np = np.empty(0, dtype=np.float64)

        indices = jnp.stack(
            [jnp.array(rows_np, dtype=jnp.int32),
             jnp.array(cols_np, dtype=jnp.int32)],
            axis=-1,
        )
        data = jnp.array(vals_np, dtype=jnp.float32)
        return BCOO((data, indices), shape=(N, N))

    # Deduplicate: because a pair (i, j) can appear from *both* i's bin
    # iteration and j's bin iteration we must deduplicate.  We keep the
    # first occurrence.
    def _dedup(
        row_parts: list[np.ndarray],
        col_parts: list[np.ndarray],
        val_parts: list[np.ndarray],
    ) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        if not row_parts:
            return row_parts, col_parts, val_parts
        rows_np = np.concatenate(row_parts)
        cols_np = np.concatenate(col_parts)
        vals_np = np.concatenate(val_parts)
        # Encode each (row, col) as a single int64 for fast dedup
        pair_ids = rows_np.astype(np.int64) * N + cols_np.astype(np.int64)
        _, unique_idx = np.unique(pair_ids, return_index=True)
        return [rows_np[unique_idx]], [cols_np[unique_idx]], [vals_np[unique_idx]]

    row_exc, col_exc, val_exc = _dedup(row_exc, col_exc, val_exc)
    row_inh, col_inh, val_inh = _dedup(row_inh, col_inh, val_inh)
    row_del, col_del, val_del = _dedup(row_del, col_del, val_del)

    W_exc_sp = _to_bcoo(row_exc, col_exc, val_exc)
    W_inh_sp = _to_bcoo(row_inh, col_inh, val_inh)
    delays_sp = _to_bcoo(row_del, col_del, val_del)

    return W_exc_sp, W_inh_sp, delays_sp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DENSE_THRESHOLD = 10_000


def build_connectivity(
    key: jax.Array,
    positions: jnp.ndarray,
    is_excitatory: jnp.ndarray,
    *,
    lambda_um: float = 200.0,
    p_max: float = 0.21,
    g_exc: float = 0.05,
    g_inh: float = 0.20,
    v_axon_um_per_ms: float = 300.0,
    dt: float = 0.5,
) -> Tuple[BCOO, BCOO, BCOO]:
    """Build distance-dependent connectivity for a cortical culture.

    Dispatches to a dense implementation for small networks (N < 10 000) and
    a spatial-hashing implementation for larger ones.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    positions : jnp.ndarray, shape (N, 2)
        Neuron positions in μm.
    is_excitatory : jnp.ndarray, shape (N,)
        Boolean mask — True for excitatory neurons.
    lambda_um : float
        Length constant for connection probability decay (default 200 μm).
    p_max : float
        Maximum connection probability at distance 0.
    g_exc : float
        Base excitatory synaptic weight (nS-like units).
    g_inh : float
        Base inhibitory synaptic weight (~4× excitatory, Lonardoni 2012).
    v_axon_um_per_ms : float
        Axonal conduction velocity (default 300 μm/ms ≈ 0.3 m/s).
    dt : float
        Simulation timestep in ms, used to discretise delays.

    Returns
    -------
    W_exc : BCOO (N, N) – excitatory weight matrix
    W_inh : BCOO (N, N) – inhibitory weight matrix
    delays : BCOO (N, N) – axonal delays in timestep multiples
    """
    N = positions.shape[0]
    kwargs = dict(
        lambda_um=lambda_um,
        p_max=p_max,
        g_exc=g_exc,
        g_inh=g_inh,
        v_axon_um_per_ms=v_axon_um_per_ms,
        dt=dt,
    )
    if N < _DENSE_THRESHOLD:
        return _build_connectivity_dense(key, positions, is_excitatory, **kwargs)
    else:
        return _build_connectivity_spatial(key, positions, is_excitatory, **kwargs)
