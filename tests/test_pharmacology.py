"""Tests for pharmacological modeling (bl1.analysis.pharmacology)."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.experimental.sparse import BCOO

from bl1.analysis.pharmacology import (
    APV,
    BICUCULLINE,
    CARBAMAZEPINE,
    CNQX,
    TTX,
    DrugEffect,
    apply_drug,
    apply_drug_to_synapses,
    wash_out,
)
from bl1.core.synapses import SynapseState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dense_weights(n: int = 4, exc_val: float = 0.05, inh_val: float = 0.20):
    """Return simple dense excitatory and inhibitory weight matrices."""
    W_exc = jnp.ones((n, n)) * exc_val
    W_inh = jnp.ones((n, n)) * inh_val
    return W_exc, W_inh


def _make_sparse_weights(n: int = 4, exc_val: float = 0.05, inh_val: float = 0.20):
    """Return BCOO sparse excitatory and inhibitory weight matrices."""
    W_exc_dense = jnp.ones((n, n)) * exc_val
    W_inh_dense = jnp.ones((n, n)) * inh_val
    W_exc = BCOO.fromdense(W_exc_dense)
    W_inh = BCOO.fromdense(W_inh_dense)
    return W_exc, W_inh


def _make_synapse_state(n: int = 4, value: float = 1.0):
    """Return a SynapseState with all conductances set to *value*."""
    v = jnp.full(n, value)
    return SynapseState(
        g_ampa=v,
        g_gaba_a=v,
        g_nmda_rise=v,
        g_nmda_decay=v,
        g_gaba_b_rise=v,
        g_gaba_b_decay=v,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ttx_silences():
    """TTX should zero both excitatory and inhibitory weight matrices."""
    W_exc, W_inh = _make_dense_weights()
    new_exc, new_inh = apply_drug(W_exc, W_inh, TTX)

    npt.assert_allclose(np.asarray(new_exc), 0.0)
    npt.assert_allclose(np.asarray(new_inh), 0.0)


def test_carbamazepine_partial_block():
    """Carbamazepine should scale excitatory weights by 0.7, leave inhibitory unchanged."""
    W_exc, W_inh = _make_dense_weights()
    new_exc, new_inh = apply_drug(W_exc, W_inh, CARBAMAZEPINE)

    expected_exc = np.asarray(W_exc) * 0.7
    npt.assert_allclose(np.asarray(new_exc), expected_exc, rtol=1e-6)
    npt.assert_allclose(np.asarray(new_inh), np.asarray(W_inh))


def test_bicuculline_blocks_gaba_a():
    """Bicuculline should zero GABA_A conductances in synapse state."""
    ss = _make_synapse_state()
    new_ss = apply_drug_to_synapses(ss, BICUCULLINE)

    npt.assert_allclose(np.asarray(new_ss.g_gaba_a), 0.0)
    # AMPA and other conductances should remain unchanged
    npt.assert_allclose(np.asarray(new_ss.g_ampa), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_nmda_rise), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_nmda_decay), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_gaba_b_rise), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_gaba_b_decay), 1.0)


def test_apv_blocks_nmda():
    """APV should zero NMDA conductances (both rise and decay components)."""
    ss = _make_synapse_state()
    new_ss = apply_drug_to_synapses(ss, APV)

    npt.assert_allclose(np.asarray(new_ss.g_nmda_rise), 0.0)
    npt.assert_allclose(np.asarray(new_ss.g_nmda_decay), 0.0)
    # Other conductances should remain unchanged
    npt.assert_allclose(np.asarray(new_ss.g_ampa), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_gaba_a), 1.0)


def test_cnqx_blocks_ampa():
    """CNQX should zero AMPA conductances."""
    ss = _make_synapse_state()
    new_ss = apply_drug_to_synapses(ss, CNQX)

    npt.assert_allclose(np.asarray(new_ss.g_ampa), 0.0)
    # Other conductances should remain unchanged
    npt.assert_allclose(np.asarray(new_ss.g_gaba_a), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_nmda_rise), 1.0)
    npt.assert_allclose(np.asarray(new_ss.g_nmda_decay), 1.0)


def test_drug_effect_composable():
    """Applying two drugs sequentially should compose their effects."""
    W_exc, W_inh = _make_dense_weights(exc_val=1.0, inh_val=1.0)

    # First apply carbamazepine (exc *= 0.7), then TTX (exc *= 0.0, inh *= 0.0)
    W_exc, W_inh = apply_drug(W_exc, W_inh, CARBAMAZEPINE)
    npt.assert_allclose(np.asarray(W_exc), 0.7, rtol=1e-6)
    npt.assert_allclose(np.asarray(W_inh), 1.0)

    W_exc, W_inh = apply_drug(W_exc, W_inh, TTX)
    npt.assert_allclose(np.asarray(W_exc), 0.0)
    npt.assert_allclose(np.asarray(W_inh), 0.0)

    # Also test composing on synapse state: APV + CNQX should zero both NMDA and AMPA
    ss = _make_synapse_state()
    ss = apply_drug_to_synapses(ss, APV)
    ss = apply_drug_to_synapses(ss, CNQX)

    npt.assert_allclose(np.asarray(ss.g_ampa), 0.0)
    npt.assert_allclose(np.asarray(ss.g_nmda_rise), 0.0)
    npt.assert_allclose(np.asarray(ss.g_nmda_decay), 0.0)
    # GABA should still be intact
    npt.assert_allclose(np.asarray(ss.g_gaba_a), 1.0)


def test_washout_restores():
    """wash_out should return the original pre-drug weight matrices."""
    W_exc_orig, W_inh_orig = _make_dense_weights()

    # Apply a drug
    W_exc_drugged, W_inh_drugged = apply_drug(W_exc_orig, W_inh_orig, TTX)
    npt.assert_allclose(np.asarray(W_exc_drugged), 0.0)

    # Wash out — restores originals
    W_exc_restored, W_inh_restored = wash_out(W_exc_orig, W_inh_orig)

    npt.assert_allclose(np.asarray(W_exc_restored), np.asarray(W_exc_orig))
    npt.assert_allclose(np.asarray(W_inh_restored), np.asarray(W_inh_orig))


def test_sparse_support():
    """Drug application should work correctly with BCOO sparse weight matrices."""
    n = 4
    exc_val = 0.05
    inh_val = 0.20

    W_exc_sparse, W_inh_sparse = _make_sparse_weights(n, exc_val, inh_val)

    # Verify they are BCOO
    assert isinstance(W_exc_sparse, BCOO)
    assert isinstance(W_inh_sparse, BCOO)

    # Apply carbamazepine
    new_exc, new_inh = apply_drug(W_exc_sparse, W_inh_sparse, CARBAMAZEPINE)

    # Result should still be BCOO
    assert isinstance(new_exc, BCOO)
    assert isinstance(new_inh, BCOO)

    # Densify and check values
    expected_exc = np.full((n, n), exc_val * 0.7)
    npt.assert_allclose(np.asarray(new_exc.todense()), expected_exc, rtol=1e-6)
    npt.assert_allclose(np.asarray(new_inh.todense()), np.full((n, n), inh_val))

    # Apply TTX to sparse
    new_exc_ttx, new_inh_ttx = apply_drug(W_exc_sparse, W_inh_sparse, TTX)
    assert isinstance(new_exc_ttx, BCOO)
    assert isinstance(new_inh_ttx, BCOO)
    npt.assert_allclose(np.asarray(new_exc_ttx.todense()), 0.0)
    npt.assert_allclose(np.asarray(new_inh_ttx.todense()), 0.0)
