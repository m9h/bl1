"""Distance-dependent connectivity for cortical culture networks.

Implements spatial neuron placement and biologically-motivated connectivity
where connection probability decays exponentially with inter-somatic distance
(Lonardoni et al. 2017, Wagenaar et al. 2006).

Supports both 2D (flat MEA substrate) and 3D (organoid / spheroid)
neuron placements.  The distance calculations work transparently for
any dimensionality thanks to ``axis=-1`` reductions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neuron placement
# ---------------------------------------------------------------------------

def place_neurons(
    key: jax.Array,
    n_neurons: int,
    substrate_um: tuple[float, float] = (3000.0, 3000.0),
    substrate_3d: tuple[float, float, float] | None = None,
) -> jnp.ndarray:
    """Uniformly distribute neurons on a 2-D substrate or in a 3-D volume.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n_neurons : int
        Number of neurons to place.
    substrate_um : tuple of float
        (width, height) of the culture substrate in micrometres.
        Used only when *substrate_3d* is ``None``.
    substrate_3d : tuple of float or None
        (width, height, depth) in micrometres for 3-D placement.
        If provided, returns (N, 3) positions; otherwise returns (N, 2).

    Returns
    -------
    positions : jnp.ndarray, shape (n_neurons, 2) or (n_neurons, 3)
        Neuron soma positions in μm.
    """
    if substrate_3d is not None:
        return jax.random.uniform(key, shape=(n_neurons, 3)) * jnp.array(substrate_3d)
    return jax.random.uniform(key, shape=(n_neurons, 2)) * jnp.array(substrate_um)


def place_neurons_spheroid(
    key: jax.Array,
    n_neurons: int,
    radius_um: float = 500.0,
    center_um: tuple[float, float, float] | None = None,
) -> jnp.ndarray:
    """Place neurons uniformly within a 3-D spheroid (organoid model).

    Uses rejection sampling to obtain a uniform distribution inside a
    sphere of the given radius.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n_neurons : int
        Number of neurons to place.
    radius_um : float
        Radius of the spheroid in micrometres (default 500).
    center_um : tuple of float or None
        (x, y, z) centre of the spheroid in μm.  Defaults to
        ``(radius_um, radius_um, radius_um)`` so the spheroid sits in
        positive coordinate space.

    Returns
    -------
    positions : jnp.ndarray, shape (n_neurons, 3)
        Neuron soma positions in μm.
    """
    if center_um is None:
        center_um = (radius_um, radius_um, radius_um)
    center = jnp.array(center_um)

    # Rejection sampling: generate points in the bounding cube, keep those
    # inside the sphere.  On average ~52 % of samples lie inside a sphere
    # inscribed in a cube, so we over-sample by ~3x per round.
    collected = []
    n_remaining = n_neurons
    oversample_factor = 3  # conservative

    while n_remaining > 0:
        key, subkey = jax.random.split(key)
        n_candidates = n_remaining * oversample_factor
        # Sample in [-1, 1]^3, then scale by radius
        candidates = 2.0 * jax.random.uniform(subkey, shape=(n_candidates, 3)) - 1.0
        # Keep points inside the unit sphere
        r2 = jnp.sum(candidates ** 2, axis=-1)
        inside = r2 < 1.0
        # Move to numpy for dynamic indexing
        candidates_np = np.asarray(candidates)
        inside_np = np.asarray(inside)
        accepted = candidates_np[inside_np]
        collected.append(accepted[:n_remaining])
        n_remaining -= accepted[:n_remaining].shape[0]

    points = jnp.array(np.concatenate(collected, axis=0)[:n_neurons])
    return points * radius_um + center


def place_neurons_layered(
    key: jax.Array,
    n_neurons: int,
    substrate_um: tuple[float, float] = (3000.0, 3000.0),
    layer_depths_um: Sequence[float] = (100.0, 200.0, 400.0, 300.0, 200.0, 100.0),
    layer_densities: Sequence[float] = (0.05, 0.15, 0.3, 0.25, 0.15, 0.1),
) -> jnp.ndarray:
    """Place neurons in a layered 3-D cortical structure.

    Models the six cortical layers with different neuron densities.
    The x and y coordinates are uniformly distributed over
    *substrate_um*; the z coordinate is drawn according to the relative
    density of each layer.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n_neurons : int
        Total number of neurons to place.
    substrate_um : tuple of float
        (width, height) of the substrate surface in μm.
    layer_depths_um : sequence of float
        Thickness of each cortical layer in μm, ordered from the
        surface (layer I) downward.
    layer_densities : sequence of float
        Relative neuron density in each layer.  Need not sum to 1
        (will be normalised internally).

    Returns
    -------
    positions : jnp.ndarray, shape (n_neurons, 3)
        Neuron soma positions in μm.  z=0 is the substrate surface.
    """
    layer_depths_um = list(layer_depths_um)
    layer_densities = list(layer_densities)
    assert len(layer_depths_um) == len(layer_densities), (
        "layer_depths_um and layer_densities must have the same length"
    )

    # Normalise densities to obtain per-layer neuron counts
    total_density = sum(layer_densities)
    fractions = [d / total_density for d in layer_densities]
    counts = [int(round(f * n_neurons)) for f in fractions]
    # Fix rounding so total == n_neurons
    diff = n_neurons - sum(counts)
    # Adjust the largest layer
    max_idx = counts.index(max(counts))
    counts[max_idx] += diff

    # Compute cumulative z-offsets for each layer
    z_offsets = [0.0]
    for depth in layer_depths_um:
        z_offsets.append(z_offsets[-1] + depth)

    all_positions = []
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        key, k_xy, k_z = jax.random.split(key, 3)
        xy = jax.random.uniform(k_xy, shape=(count, 2)) * jnp.array(substrate_um)
        z = z_offsets[i] + jax.random.uniform(k_z, shape=(count, 1)) * layer_depths_um[i]
        all_positions.append(jnp.concatenate([xy, z], axis=-1))

    return jnp.concatenate(all_positions, axis=0)


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
) -> tuple[BCOO, BCOO, BCOO]:
    """Build connectivity for small networks using a full distance matrix.

    Works transparently with (N, 2) or (N, 3) positions — the distance
    calculation ``jnp.sqrt(jnp.sum(diff ** 2, axis=-1))`` reduces over the
    last axis regardless of dimensionality.

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
) -> tuple[BCOO, BCOO, BCOO]:
    """Build connectivity for large networks via spatial binning in NumPy.

    Neurons are assigned to square (2D) or cubic (3D) bins of side
    ``lambda_um``.  For each neuron we only evaluate potential connections
    to neurons whose bins fall within a ``cutoff = 3 * lambda_um`` radius,
    making the algorithm O(N * k) with k ~ average neighbours inside the
    cutoff.

    Works transparently with (N, 2) or (N, 3) positions.

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
    n_dim = pos_np.shape[1]  # 2 or 3

    cutoff = 3.0 * lambda_um
    bin_size = lambda_um  # square/cubic bin side

    # --- build spatial hash ----------------------------------------------
    bin_idx = (pos_np / bin_size).astype(np.int32)           # (N, n_dim)
    from collections import defaultdict
    bins: dict[tuple, list[int]] = defaultdict(list)
    for i in range(N):
        bins[tuple(int(bin_idx[i, d]) for d in range(n_dim))].append(i)

    # Pre-compute the set of relative bin offsets within cutoff.
    # Works for both 2D and 3D by building offsets over n_dim axes.
    max_offset = int(np.ceil(cutoff / bin_size))
    axis_range = list(range(-max_offset, max_offset + 1))
    if n_dim == 2:
        offsets = [
            (dx, dy)
            for dx in axis_range
            for dy in axis_range
            if np.hypot(dx, dy) * bin_size <= cutoff + bin_size
        ]
    else:
        offsets = [
            (dx, dy, dz)
            for dx in axis_range
            for dy in axis_range
            for dz in axis_range
            if np.sqrt(dx**2 + dy**2 + dz**2) * bin_size <= cutoff + bin_size
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
    for bin_key, src_neurons in bins.items():
        src_arr = np.array(src_neurons, dtype=np.int64)
        src_pos = pos_np[src_arr]  # (|src|, n_dim)

        # Gather candidate targets from neighbouring bins
        tgt_list: list[int] = []
        for offset in offsets:
            nb = tuple(bin_key[d] + offset[d] for d in range(n_dim))
            if nb in bins:
                tgt_list.extend(bins[nb])
        if not tgt_list:
            continue

        tgt_arr = np.array(tgt_list, dtype=np.int64)
        tgt_pos = pos_np[tgt_arr]  # (|tgt|, n_dim)

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
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
# Optimized spatial connectivity (KD-tree + batched processing)
# ---------------------------------------------------------------------------

def _build_connectivity_spatial_fast(
    key: jax.Array,
    positions: jnp.ndarray,
    is_excitatory: jnp.ndarray,
    lambda_um: float = 200.0,
    p_max: float = 0.21,
    g_exc: float = 0.05,
    g_inh: float = 0.20,
    v_axon_um_per_ms: float = 300.0,
    dt: float = 0.5,
) -> tuple[BCOO, BCOO, BCOO]:
    """Build connectivity for large networks using scipy KD-tree.

    Faster than :func:`_build_connectivity_spatial` because:

    1. Uses ``scipy.spatial.cKDTree.query_pairs`` to find all candidate
       pairs within the distance cutoff in O(N log N) time — no Python
       loop over spatial bins.
    2. Processes all pairs in a single vectorised NumPy pass — no per-bin
       distance matrices.
    3. Produces directed (row, col) pairs directly — no deduplication
       step needed.

    Falls back to :func:`_build_connectivity_spatial` if scipy is not
    available (should never happen with a normal install).

    Returns
    -------
    W_exc, W_inh, delays : BCOO (N, N) each.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        logger.warning(
            "scipy not available; falling back to slower spatial-hash "
            "connectivity builder"
        )
        return _build_connectivity_spatial(
            key, positions, is_excitatory,
            lambda_um=lambda_um, p_max=p_max, g_exc=g_exc, g_inh=g_inh,
            v_axon_um_per_ms=v_axon_um_per_ms, dt=dt,
        )

    pos_np: np.ndarray = np.asarray(positions)
    is_exc_np: np.ndarray = np.asarray(is_excitatory).astype(bool)
    N = pos_np.shape[0]

    cutoff = 3.0 * lambda_um

    # --- KD-tree: find all pairs within cutoff ---------------------------
    tree = cKDTree(pos_np)
    # query_pairs returns a set of (i, j) with i < j and dist <= cutoff.
    # We need directed connections (i->j AND j->i are independent draws).
    pair_set = tree.query_pairs(r=cutoff, output_type="ndarray")  # (P, 2)

    if pair_set.shape[0] == 0:
        # No pairs within cutoff — return empty matrices
        empty = BCOO(
            (jnp.empty(0, dtype=jnp.float32),
             jnp.empty((0, 2), dtype=jnp.int32)),
            shape=(N, N),
        )
        return empty, empty, empty

    # Duplicate into both directions: (i,j) AND (j,i) as separate edges
    src_all = np.concatenate([pair_set[:, 0], pair_set[:, 1]])  # pre-synaptic
    tgt_all = np.concatenate([pair_set[:, 1], pair_set[:, 0]])  # post-synaptic

    # --- distances for all directed pairs --------------------------------
    diff = pos_np[src_all] - pos_np[tgt_all]                    # (2P, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))                  # (2P,)

    # --- connection probabilities ----------------------------------------
    prob = p_max * np.exp(-dists / lambda_um)

    # --- RNG setup -------------------------------------------------------
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    rand = rng.random(prob.shape)
    connected = rand < prob

    if not np.any(connected):
        empty = BCOO(
            (jnp.empty(0, dtype=jnp.float32),
             jnp.empty((0, 2), dtype=jnp.int32)),
            shape=(N, N),
        )
        return empty, empty, empty

    # Filter to connected edges only
    rows = src_all[connected]    # pre-synaptic (row in W[post, pre]? No.)
    cols = tgt_all[connected]    # post-synaptic
    conn_dists = dists[connected]

    # NOTE: Weight matrix convention is W[post, pre] so that
    # W @ spikes_pre gives input to each post-synaptic neuron.
    # In the code: rows = pre-synaptic index (column of W)
    #              cols = post-synaptic index (row of W)
    # But we need the BCOO matrix in (post, pre) layout, so:
    W_rows = cols   # post-synaptic = row index in weight matrix
    W_cols = rows   # pre-synaptic = column index in weight matrix

    # --- delays (timesteps, minimum 1) -----------------------------------
    delay_steps = np.maximum(np.round(conn_dists / v_axon_um_per_ms / dt), 1.0)

    # --- split E / I by pre-synaptic type --------------------------------
    pre_exc = is_exc_np[W_cols]  # pre-synaptic neuron is excitatory

    # --- weight noise (+-10 %) -------------------------------------------
    noise = 1.0 + 0.1 * (2.0 * rng.random(W_rows.shape[0]) - 1.0)

    # --- Excitatory synapses ---------------------------------------------
    exc_mask = pre_exc
    inh_mask = ~pre_exc

    def _to_bcoo_fast(mask, g_base):
        if not np.any(mask):
            return BCOO(
                (jnp.empty(0, dtype=jnp.float32),
                 jnp.empty((0, 2), dtype=jnp.int32)),
                shape=(N, N),
            )
        r = W_rows[mask]
        c = W_cols[mask]
        v = (g_base * noise[mask]).astype(np.float32)
        indices = jnp.stack(
            [jnp.array(r, dtype=jnp.int32),
             jnp.array(c, dtype=jnp.int32)],
            axis=-1,
        )
        return BCOO((jnp.array(v, dtype=jnp.float32), indices), shape=(N, N))

    W_exc_sp = _to_bcoo_fast(exc_mask, g_exc)
    W_inh_sp = _to_bcoo_fast(inh_mask, g_inh)

    # --- Delays (all connected synapses) ---------------------------------
    all_indices = jnp.stack(
        [jnp.array(W_rows, dtype=jnp.int32),
         jnp.array(W_cols, dtype=jnp.int32)],
        axis=-1,
    )
    delays_sp = BCOO(
        (jnp.array(delay_steps.astype(np.float32), dtype=jnp.float32),
         all_indices),
        shape=(N, N),
    )

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
) -> tuple[BCOO, BCOO, BCOO]:
    """Build distance-dependent connectivity for a cortical culture.

    Dispatches to a dense implementation for small networks (N < 10 000) and
    a spatial-hashing implementation for larger ones.  Works transparently
    with 2D ``(N, 2)`` or 3D ``(N, 3)`` neuron positions.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    positions : jnp.ndarray, shape (N, 2) or (N, 3)
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
        return _build_connectivity_spatial_fast(key, positions, is_excitatory, **kwargs)
