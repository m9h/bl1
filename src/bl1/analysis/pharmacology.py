"""Pharmacological modeling — simulate drug effects by modifying model parameters.

Supported drugs (matching DishBrain validation, Watmuff et al. 2025):

| Drug          | Target          | Model Change                       | Expected Effect                      |
|---------------|-----------------|------------------------------------|--------------------------------------|
| TTX           | Na+ channels    | Zero all synaptic weights          | Silences network                     |
| Carbamazepine | Partial Na+ block | Scale excitatory weights by 0.7  | Improves performance                 |
| Bicuculline   | GABA_A block    | Zero GABA_A conductance            | Disinhibition, excessive bursting    |
| APV           | NMDA block      | Zero NMDA conductance              | Impairs STDP-dependent learning      |
| CNQX          | AMPA block      | Zero AMPA conductance              | Silences fast excitation             |
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array
from jax.experimental.sparse import BCOO


class DrugEffect(NamedTuple):
    """Description of a pharmacological manipulation."""

    name: str
    description: str
    # Scaling factors for different conductance types (1.0 = no effect)
    ampa_scale: float = 1.0
    nmda_scale: float = 1.0
    gaba_a_scale: float = 1.0
    gaba_b_scale: float = 1.0
    exc_weight_scale: float = 1.0
    inh_weight_scale: float = 1.0


# ---------------------------------------------------------------------------
# Pre-defined drug profiles
# ---------------------------------------------------------------------------

TTX = DrugEffect(
    name="TTX",
    description="Tetrodotoxin — blocks Na+ channels, silences all activity",
    exc_weight_scale=0.0,
    inh_weight_scale=0.0,
)

CARBAMAZEPINE = DrugEffect(
    name="Carbamazepine",
    description="Partial Na+ channel block — reduces but doesn't eliminate excitation",
    exc_weight_scale=0.7,
)

BICUCULLINE = DrugEffect(
    name="Bicuculline",
    description="GABA_A antagonist — blocks inhibition, causes disinhibition",
    gaba_a_scale=0.0,
)

APV = DrugEffect(
    name="APV",
    description="NMDA antagonist — blocks NMDA receptors, impairs plasticity",
    nmda_scale=0.0,
)

CNQX = DrugEffect(
    name="CNQX",
    description="AMPA antagonist — blocks fast excitation",
    ampa_scale=0.0,
)


# ---------------------------------------------------------------------------
# Weight matrix manipulation
# ---------------------------------------------------------------------------

def _scale_matrix(mat: Array | BCOO, scale: float) -> Array | BCOO:
    """Scale a weight matrix (dense or BCOO sparse) by a scalar factor.

    For dense arrays this is a simple multiplication.  For BCOO sparse
    arrays we scale the ``.data`` values and reconstruct.
    """
    if scale == 1.0:
        return mat
    if isinstance(mat, BCOO):
        new_data = mat.data * scale
        return BCOO((new_data, mat.indices), shape=mat.shape)
    # Dense array
    return mat * scale


def apply_drug(
    W_exc: Array | BCOO,
    W_inh: Array | BCOO,
    drug: DrugEffect,
) -> tuple[Array | BCOO, Array | BCOO]:
    """Apply a drug effect to weight matrices.

    Supports both dense and BCOO sparse matrices.

    Args:
        W_exc: Excitatory weight matrix (N, N).
        W_inh: Inhibitory weight matrix (N, N).
        drug: DrugEffect specifying scaling factors.

    Returns:
        ``(new_W_exc, new_W_inh)`` with scaled weights.
    """
    new_W_exc = _scale_matrix(W_exc, drug.exc_weight_scale)
    new_W_inh = _scale_matrix(W_inh, drug.inh_weight_scale)
    return new_W_exc, new_W_inh


def apply_drug_to_synapses(
    syn_state: SynapseState,
    drug: DrugEffect,
) -> SynapseState:
    """Scale existing synaptic conductances by drug factors.

    Useful for applying a drug mid-simulation.  Operates on a
    :class:`~bl1.core.synapses.SynapseState` and returns a new one with
    conductance values scaled according to the drug profile.

    Parameters
    ----------
    syn_state : SynapseState
        Current synaptic conductance state.
    drug : DrugEffect
        Drug profile specifying per-receptor scaling factors.

    Returns
    -------
    SynapseState
        New state with conductances multiplied by the drug's scaling
        factors (``ampa_scale``, ``nmda_scale``, ``gaba_a_scale``,
        ``gaba_b_scale``).
    """
    return syn_state._replace(
        g_ampa=syn_state.g_ampa * drug.ampa_scale,
        g_gaba_a=syn_state.g_gaba_a * drug.gaba_a_scale,
        g_nmda_rise=syn_state.g_nmda_rise * drug.nmda_scale,
        g_nmda_decay=syn_state.g_nmda_decay * drug.nmda_scale,
        g_gaba_b_rise=syn_state.g_gaba_b_rise * drug.gaba_b_scale,
        g_gaba_b_decay=syn_state.g_gaba_b_decay * drug.gaba_b_scale,
    )


def wash_out(
    original_W_exc: Array | BCOO,
    original_W_inh: Array | BCOO,
) -> tuple[Array | BCOO, Array | BCOO]:
    """Return original (pre-drug) weight matrices — simulates drug washout.

    Args:
        original_W_exc: Excitatory weight matrix saved before drug application.
        original_W_inh: Inhibitory weight matrix saved before drug application.

    Returns:
        ``(original_W_exc, original_W_inh)`` unchanged.
    """
    return original_W_exc, original_W_inh
