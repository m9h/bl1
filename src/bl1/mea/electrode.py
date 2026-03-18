"""Electrode configurations for virtual multi-electrode arrays.

Provides MEA hardware definitions and spatial mappings between neurons
and electrodes.  Two configurations are included:

- **cl1_64ch**: 8x8 grid, 200 um spacing — the default for BL-1 experiments.
- **maxone_hd**: 120x220 grid, 17.5 um spacing (26,400 electrodes) — Phase 3.

The ``build_neuron_electrode_map`` function precomputes a boolean mask
relating every electrode to the neurons within its detection radius,
which is reused throughout an experiment.

For HD-MEA configurations where the dense (E, N) map is prohibitively
large, ``build_neuron_electrode_map_sparse`` provides a BCOO sparse
alternative using spatial binning.  ``select_electrode_subset`` handles
the hardware constraint that HD-MEAs cannot record from all electrodes
simultaneously.  ``compute_lfp`` approximates extracellular local field
potentials at electrode positions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO

# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------


class MEAConfig(NamedTuple):
    """MEA hardware configuration."""

    name: str
    n_electrodes: int
    grid_shape: tuple  # (rows, cols)
    spacing_um: float  # inter-electrode distance in um
    positions: Array  # (n_electrodes, 2) electrode positions in um
    detection_radius_um: float = 100.0
    activation_radius_um: float = 75.0


# ---------------------------------------------------------------------------
# Built-in electrode layouts
# ---------------------------------------------------------------------------


def _make_cl1_64ch() -> MEAConfig:
    """CL1 64-channel MEA: 8x8 grid, 200 um spacing on a 3000x3000 um substrate."""
    rows, cols = 8, 8
    spacing = 200.0
    center = 1500.0  # substrate center

    # Grid offsets: -3.5, -2.5, ..., +3.5 times spacing, centered on substrate
    offsets = jnp.arange(8) - 3.5  # [-3.5, -2.5, ..., 3.5]
    xs = center + offsets * spacing  # 800, 1000, ..., 2200
    ys = center + offsets * spacing

    # Meshgrid — (rows, cols) then flatten to (n_electrodes, 2)
    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    positions = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (64, 2)

    return MEAConfig(
        name="cl1_64ch",
        n_electrodes=rows * cols,
        grid_shape=(rows, cols),
        spacing_um=spacing,
        positions=positions,
        detection_radius_um=100.0,
        activation_radius_um=75.0,
    )


def _make_maxone_hd() -> MEAConfig:
    """MaxOne HD MEA: 120x220 grid, 17.5 um spacing (26,400 electrodes).

    This is a Phase-3 configuration.  Positions are defined lazily here;
    full support (including routing constraints) will be added later.
    """
    rows, cols = 120, 220
    spacing = 17.5

    # Center the grid on the same 3000x3000 um substrate
    center_x = 1500.0
    center_y = 1500.0

    xs = center_x + (jnp.arange(cols) - (cols - 1) / 2.0) * spacing
    ys = center_y + (jnp.arange(rows) - (rows - 1) / 2.0) * spacing

    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    positions = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (26400, 2)

    return MEAConfig(
        name="maxone_hd",
        n_electrodes=rows * cols,
        grid_shape=(rows, cols),
        spacing_um=spacing,
        positions=positions,
        detection_radius_um=17.5,
        activation_radius_um=12.0,
    )


# ---------------------------------------------------------------------------
# MEA wrapper class
# ---------------------------------------------------------------------------


class MEA:
    """Virtual multi-electrode array.

    Instantiate with a config name and use ``positions`` / ``n_electrodes``
    to access the electrode geometry.

    Example::

        mea = MEA("cl1_64ch")
        print(mea.n_electrodes)  # 64
        print(mea.positions.shape)  # (64, 2)
    """

    def __init__(self, config: str = "cl1_64ch") -> None:
        if config == "cl1_64ch":
            self.config = _make_cl1_64ch()
        elif config == "maxone_hd":
            self.config = _make_maxone_hd()
        else:
            raise ValueError(f"Unknown MEA config: {config}")

    @property
    def positions(self) -> Array:
        """Electrode positions in um, shape (n_electrodes, 2)."""
        return self.config.positions

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes."""
        return self.config.n_electrodes

    @property
    def detection_radius_um(self) -> float:
        """Detection radius in um."""
        return self.config.detection_radius_um

    @property
    def activation_radius_um(self) -> float:
        """Activation radius in um for stimulation."""
        return self.config.activation_radius_um


# ---------------------------------------------------------------------------
# Neuron-electrode spatial mapping
# ---------------------------------------------------------------------------


def build_neuron_electrode_map(
    neuron_positions: Array,
    electrode_positions: Array,
    radius_um: float,
) -> Array:
    """Precompute a boolean mask mapping electrodes to nearby neurons.

    For each electrode, identifies which neurons fall within
    ``radius_um`` of the electrode centre.  This mask is computed once
    at experiment setup and reused for spike detection and stimulation.

    Handles both 2D and 3D neuron positions.  For 3D neurons the
    electrodes are assumed to sit on the z=0 surface; distance is
    computed as 3D Euclidean distance from each neuron to the electrode
    position at (x, y, z=0).

    Args:
        neuron_positions: Neuron positions in um, shape (N, 2) or (N, 3).
        electrode_positions: Electrode positions in um, shape (E, 2).
        radius_um: Detection or activation radius in um.

    Returns:
        Boolean mask of shape (E, N) where ``mask[e, n]`` is ``True``
        when neuron *n* is within ``radius_um`` of electrode *e*.
    """
    n_dim = neuron_positions.shape[1]
    if n_dim == 3 and electrode_positions.shape[1] == 2:
        # Pad electrode positions with z=0 for 3D distance calculation
        electrode_3d = jnp.pad(electrode_positions, ((0, 0), (0, 1)))
        diff = electrode_3d[:, None, :] - neuron_positions[None, :, :]
    else:
        diff = electrode_positions[:, None, :] - neuron_positions[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (E, N)
    return dist < radius_um


# ---------------------------------------------------------------------------
# Sparse neuron-electrode mapping (HD-MEA)
# ---------------------------------------------------------------------------


def build_neuron_electrode_map_sparse(
    neuron_positions: Array,
    electrode_positions: Array,
    radius_um: float,
) -> BCOO:
    """Build a sparse (E, N) neuron-electrode map for large MEAs.

    Uses spatial binning to efficiently find nearby neurons for each
    electrode, returning a BCOO sparse matrix instead of a dense boolean
    array.  For an HD-MEA with 26,400 electrodes and 100K neurons the
    dense map would require 2.6 billion entries; this function stores
    only the True entries.

    Handles both 2D and 3D neuron positions.  For 3D neurons the
    electrodes are assumed to sit on the z=0 surface; spatial binning
    uses only the x-y coordinates, and the full 3D Euclidean distance
    is used for the radius test.

    The algorithm:

    1. Bin neurons into square cells of side ``radius_um`` (using x-y
       coordinates only, even for 3D neurons).
    2. For each electrode, look up bins within one cell of the
       electrode's own bin (a 3x3 neighbourhood).
    3. Compute distances only to candidate neurons in those bins.
    4. Collect (electrode, neuron) pairs where distance < radius_um.
    5. Return as a BCOO matrix with boolean data.

    Args:
        neuron_positions: Neuron positions in um, shape (N, 2) or (N, 3).
        electrode_positions: Electrode positions in um, shape (E, 2).
        radius_um: Detection radius in um.

    Returns:
        BCOO sparse matrix of shape (E, N) with boolean (float32 1.0)
        data indicating which neurons fall within ``radius_um`` of
        each electrode.
    """
    pos_np = np.asarray(neuron_positions)
    elec_np = np.asarray(electrode_positions)
    N = pos_np.shape[0]
    E = elec_np.shape[0]
    neuron_3d = pos_np.shape[1] == 3

    # For 3D neurons, pad electrode positions with z=0 for distance calc
    if neuron_3d and elec_np.shape[1] == 2:
        elec_3d = np.pad(elec_np, ((0, 0), (0, 1)))
    else:
        elec_3d = elec_np

    bin_size = radius_um

    # --- bin neurons (always using x-y coordinates) ------------------------
    neuron_bins_xy = (pos_np[:, :2] / bin_size).astype(np.int32)  # (N, 2)
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(N):
        bins[(int(neuron_bins_xy[i, 0]), int(neuron_bins_xy[i, 1]))].append(i)

    # Precompute electrode bin indices (always 2D)
    elec_bins = (elec_np[:, :2] / bin_size).astype(np.int32)  # (E, 2)

    # Relative offsets: 3x3 neighbourhood
    offsets = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]

    # --- collect sparse entries --------------------------------------------
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    for e in range(E):
        bx, by = int(elec_bins[e, 0]), int(elec_bins[e, 1])
        candidates: list[int] = []
        for dx, dy in offsets:
            nb = (bx + dx, by + dy)
            if nb in bins:
                candidates.extend(bins[nb])
        if not candidates:
            continue

        cand_arr = np.array(candidates, dtype=np.int64)
        cand_pos = pos_np[cand_arr]  # (C, 2) or (C, 3)

        # Distance: use full dimensionality (2D or 3D)
        diff = cand_pos - elec_3d[e]  # (C, 2) or (C, 3)
        dist = np.sqrt(np.sum(diff**2, axis=-1))  # (C,)

        within = dist < radius_um
        if not np.any(within):
            continue

        matched = cand_arr[within]
        row_parts.append(np.full(matched.shape[0], e, dtype=np.int64))
        col_parts.append(matched)

    # --- assemble BCOO -----------------------------------------------------
    if row_parts:
        rows_np = np.concatenate(row_parts)
        cols_np = np.concatenate(col_parts)
    else:
        rows_np = np.empty(0, dtype=np.int64)
        cols_np = np.empty(0, dtype=np.int64)

    indices = jnp.stack(
        [jnp.array(rows_np, dtype=jnp.int32), jnp.array(cols_np, dtype=jnp.int32)],
        axis=-1,
    )
    data = jnp.ones(indices.shape[0], dtype=jnp.float32)
    return BCOO((data, indices), shape=(E, N))


# ---------------------------------------------------------------------------
# Electrode subset selection (HD-MEA routing constraint)
# ---------------------------------------------------------------------------


def select_electrode_subset(
    mea_config: MEAConfig,
    region: str = "center",
    n_electrodes: int = 1024,
    key: Array | None = None,
) -> Array:
    """Select a subset of electrodes from an HD-MEA.

    HD-MEAs such as the MaxOne cannot record from all electrodes
    simultaneously due to routing constraints.  This function selects
    a manageable subset according to a spatial strategy.

    Args:
        mea_config: Full MEA configuration.
        region: Selection strategy:

            - ``"center"`` — select the *n_electrodes* closest to the
              array centre.
            - ``"random"`` — uniformly random subset (requires *key*).
            - ``"grid"``  — regular sub-sampled grid covering the full
              array extent.

        n_electrodes: Number of electrodes to select.  Clamped to
            ``mea_config.n_electrodes`` if larger.
        key: JAX PRNG key, required when *region* is ``"random"``.

    Returns:
        Integer array of shape ``(n_electrodes,)`` containing the
        selected electrode indices.

    Raises:
        ValueError: If *region* is unknown or ``"random"`` is requested
            without a PRNG *key*.
    """
    total = mea_config.n_electrodes
    n_electrodes = min(n_electrodes, total)
    positions_np = np.asarray(mea_config.positions)

    if region == "center":
        # Distance of each electrode from the array centroid
        centroid = positions_np.mean(axis=0)  # (2,)
        dists = np.sqrt(np.sum((positions_np - centroid) ** 2, axis=-1))
        idx = np.argsort(dists)[:n_electrodes]
        return jnp.array(np.sort(idx), dtype=jnp.int32)

    if region == "random":
        if key is None:
            raise ValueError("A JAX PRNG key is required for region='random'.")
        chosen = jax.random.choice(key, total, shape=(n_electrodes,), replace=False)
        return jnp.sort(chosen)

    if region == "grid":
        rows, cols = mea_config.grid_shape
        # Determine a sub-sampling stride that yields ~n_electrodes
        # We want stride_r * stride_c ≈ (rows * cols) / n_electrodes
        ratio = (rows * cols) / n_electrodes
        stride = max(1, int(np.round(np.sqrt(ratio))))

        sub_rows = np.arange(0, rows, stride)
        sub_cols = np.arange(0, cols, stride)

        # Grid indices into the flattened electrode array (row-major)
        rr, cc = np.meshgrid(sub_rows, sub_cols, indexing="ij")
        flat_idx = (rr * cols + cc).ravel()

        # If we got more than requested, trim by distance to centre
        if flat_idx.shape[0] > n_electrodes:
            centroid = positions_np.mean(axis=0)
            dists = np.sqrt(np.sum((positions_np[flat_idx] - centroid) ** 2, axis=-1))
            keep = np.argsort(dists)[:n_electrodes]
            flat_idx = flat_idx[keep]

        return jnp.array(np.sort(flat_idx), dtype=jnp.int32)

    raise ValueError(f"Unknown region: {region!r}. Expected 'center', 'random', or 'grid'.")


# ---------------------------------------------------------------------------
# Local field potential approximation
# ---------------------------------------------------------------------------


def compute_lfp(
    neuron_positions: Array,
    electrode_positions: Array,
    membrane_currents: Array,
    sigma: float = 0.3,
) -> Array:
    """Compute simplified local field potential at electrode positions.

    Uses a point-source approximation in an infinite homogeneous
    extracellular medium:

        V_ext(r_e) = sum_n  (1 / 4*pi*sigma) * I_m(n) / |r_e - r_n|

    Distances are converted from micrometres to metres inside the
    calculation so that the conductivity (in S/m) is applied correctly,
    and the result is returned in microvolts.

    A minimum distance clamp (1 um) prevents divergence when a neuron
    sits exactly at an electrode.

    Handles both 2D and 3D neuron positions.  For 3D neurons the
    electrodes are assumed to sit on the z=0 surface.

    Args:
        neuron_positions: (N, 2) or (N, 3) neuron positions in um.
        electrode_positions: (E, 2) electrode positions in um.
        membrane_currents: (N,) total membrane current per neuron
            (in nA, with the convention that negative = inward current).
        sigma: Extracellular conductivity in S/m (default 0.3).

    Returns:
        (E,) LFP values at each electrode in uV.
    """
    n_dim = neuron_positions.shape[1]
    if n_dim == 3 and electrode_positions.shape[1] == 2:
        # Pad electrode positions with z=0 for 3D distance calculation
        electrode_positions = jnp.pad(electrode_positions, ((0, 0), (0, 1)))
    # (E, 1, D) - (1, N, D) -> (E, N, D) -> (E, N)
    diff = electrode_positions[:, None, :] - neuron_positions[None, :, :]
    dist_um = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (E, N)

    # Clamp minimum distance to 1 um to avoid singularity
    dist_um = jnp.maximum(dist_um, 1.0)

    # Convert um -> m for the physics formula
    dist_m = dist_um * 1e-6

    # Point-source coefficient: 1 / (4 * pi * sigma)
    coeff = 1.0 / (4.0 * jnp.pi * sigma)

    # membrane_currents in nA -> convert to A: * 1e-9
    I_A = membrane_currents * 1e-9  # (N,)

    # V = coeff * sum_n (I_n / d_n)  in volts
    # (E, N) * (N,) -> (E, N), then sum over N -> (E,)
    V_volts = coeff * (I_A[None, :] / dist_m).sum(axis=-1)

    # Convert V -> uV: * 1e6
    V_uv = V_volts * 1e6

    return V_uv
