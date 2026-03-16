"""Electrode configurations for virtual multi-electrode arrays.

Provides MEA hardware definitions and spatial mappings between neurons
and electrodes.  Two configurations are included:

- **cl1_64ch**: 8x8 grid, 200 um spacing — the default for BL-1 experiments.
- **maxone_hd**: 120x220 grid, 17.5 um spacing (26,400 electrodes) — Phase 3.

The ``build_neuron_electrode_map`` function precomputes a boolean mask
relating every electrode to the neurons within its detection radius,
which is reused throughout an experiment.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------

class MEAConfig(NamedTuple):
    """MEA hardware configuration."""
    name: str
    n_electrodes: int
    grid_shape: tuple          # (rows, cols)
    spacing_um: float          # inter-electrode distance in um
    positions: Array           # (n_electrodes, 2) electrode positions in um
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

    Args:
        neuron_positions: Neuron (x, y) positions in um, shape (N, 2).
        electrode_positions: Electrode (x, y) positions in um,
            shape (E, 2).
        radius_um: Detection or activation radius in um.

    Returns:
        Boolean mask of shape (E, N) where ``mask[e, n]`` is ``True``
        when neuron *n* is within ``radius_um`` of electrode *e*.
    """
    # (E, 1, 2) - (1, N, 2) -> (E, N, 2) -> (E, N)
    diff = electrode_positions[:, None, :] - neuron_positions[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (E, N)
    return dist < radius_um
