"""Tests for Tsodyks-Markram short-term plasticity (bl1.plasticity.stp)."""

import jax.numpy as jnp
import pytest

from bl1.plasticity.stp import (
    STPParams,
    STPState,
    create_stp_params,
    init_stp_state,
    stp_step,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _exc_mask(n: int) -> jnp.ndarray:
    """All excitatory."""
    return jnp.ones(n, dtype=jnp.bool_)


def _inh_mask(n: int) -> jnp.ndarray:
    """All inhibitory."""
    return jnp.zeros(n, dtype=jnp.bool_)


def _spike(n: int, idx: int) -> jnp.ndarray:
    """Single neuron *idx* fires."""
    return jnp.zeros(n, dtype=jnp.bool_).at[idx].set(True)


def _no_spike(n: int) -> jnp.ndarray:
    return jnp.zeros(n, dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# Tests: parameter creation
# ---------------------------------------------------------------------------

class TestCreateSTPParams:
    def test_excitatory_params(self):
        """Excitatory neurons get U=0.5, tau_rec=800, tau_fac~0."""
        n = 4
        is_exc = jnp.array([True, True, False, False])
        params = create_stp_params(n, is_exc)

        assert jnp.isclose(params.U[0], 0.5)
        assert jnp.isclose(params.U[1], 0.5)
        assert jnp.isclose(params.tau_rec[0], 800.0)
        assert jnp.isclose(params.tau_rec[1], 800.0)
        assert params.tau_fac[0] < 1.0  # effectively zero

    def test_inhibitory_params(self):
        """Inhibitory neurons get U=0.04, tau_rec=100, tau_fac=1000."""
        n = 4
        is_exc = jnp.array([True, True, False, False])
        params = create_stp_params(n, is_exc)

        assert jnp.isclose(params.U[2], 0.04)
        assert jnp.isclose(params.U[3], 0.04)
        assert jnp.isclose(params.tau_rec[2], 100.0)
        assert jnp.isclose(params.tau_rec[3], 100.0)
        assert jnp.isclose(params.tau_fac[2], 1000.0)
        assert jnp.isclose(params.tau_fac[3], 1000.0)


# ---------------------------------------------------------------------------
# Tests: state initialisation
# ---------------------------------------------------------------------------

class TestInitSTPState:
    def test_init_x_is_one(self):
        """x should start at 1 for all neurons."""
        n = 10
        params = create_stp_params(n, _exc_mask(n))
        state = init_stp_state(n, params)
        assert state.x.shape == (n,)
        assert jnp.allclose(state.x, 1.0)

    def test_init_u_equals_U(self):
        """u should start at U for all neurons."""
        n = 6
        is_exc = jnp.array([True, True, True, False, False, False])
        params = create_stp_params(n, is_exc)
        state = init_stp_state(n, params)
        assert state.u.shape == (n,)
        assert jnp.allclose(state.u, params.U)


# ---------------------------------------------------------------------------
# Tests: excitatory depression
# ---------------------------------------------------------------------------

class TestExcitatoryDepression:
    def test_monotonic_depression(self):
        """Excitatory neuron firing at 50Hz: scale should monotonically decrease."""
        n = 1
        params = create_stp_params(n, _exc_mask(n))
        state = init_stp_state(n, params)
        dt = 0.5  # ms

        # 50Hz = spike every 20ms = 40 steps at dt=0.5
        steps_between_spikes = 40
        spike = jnp.array([True])
        no_spike = jnp.array([False])

        scales = []
        for _ in range(10):
            # Inter-spike interval: no spikes for 39 steps
            for _ in range(steps_between_spikes - 1):
                state, _ = stp_step(state, params, no_spike, dt)
            # Spike
            state, scale = stp_step(state, params, spike, dt)
            scales.append(float(scale[0]))

        # Each successive scale should be smaller (depression)
        for i in range(1, len(scales)):
            assert scales[i] < scales[i - 1], (
                f"Scale did not decrease at spike {i+1}: "
                f"{scales[i]:.6f} >= {scales[i-1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests: excitatory recovery
# ---------------------------------------------------------------------------

class TestExcitatoryRecovery:
    def test_recovery_after_depression(self):
        """After depression, x recovers close to 1 after 2000ms of silence."""
        n = 1
        params = create_stp_params(n, _exc_mask(n))
        state = init_stp_state(n, params)
        dt = 0.5
        spike = jnp.array([True])
        no_spike = jnp.array([False])

        # Fire a few spikes to depress
        for _ in range(5):
            state, _ = stp_step(state, params, spike, dt)

        # x should be substantially depressed
        assert state.x[0] < 0.5, f"Expected depression, x={state.x[0]}"

        # Wait 2000ms = 4000 steps at dt=0.5
        for _ in range(4000):
            state, _ = stp_step(state, params, no_spike, dt)

        # x should recover close to 1.0 (tau_rec=800ms, waited 2.5 * tau_rec)
        assert state.x[0] > 0.9, (
            f"Expected recovery to ~1.0, got x={state.x[0]}"
        )


# ---------------------------------------------------------------------------
# Tests: inhibitory facilitation
# ---------------------------------------------------------------------------

class TestInhibitoryFacilitation:
    def test_facilitation_increases_scale(self):
        """Inhibitory neuron firing at 50Hz: scale should increase during early spikes.

        For facilitating synapses (low U, long tau_fac), the release probability u
        builds up over successive spikes.  Eventually x depletion can counteract
        facilitation, but the first several spikes should show clear facilitation
        (increasing scale).  We also verify the overall scale is higher than the
        first spike, confirming facilitation is the dominant effect.
        """
        n = 1
        params = create_stp_params(n, _inh_mask(n))
        state = init_stp_state(n, params)
        dt = 0.5

        # 50Hz = spike every 20ms = 40 steps
        steps_between_spikes = 40
        spike = jnp.array([True])
        no_spike = jnp.array([False])

        scales = []
        for _ in range(10):
            for _ in range(steps_between_spikes - 1):
                state, _ = stp_step(state, params, no_spike, dt)
            state, scale = stp_step(state, params, spike, dt)
            scales.append(float(scale[0]))

        # First 5 spikes should show facilitation (monotonically increasing)
        for i in range(1, 5):
            assert scales[i] > scales[i - 1], (
                f"Scale did not increase at spike {i+1}: "
                f"{scales[i]:.6f} <= {scales[i-1]:.6f}"
            )

        # The last spike scale should still exceed the first (net facilitation)
        assert scales[-1] > scales[0], (
            f"Overall facilitation failed: last={scales[-1]:.6f} <= first={scales[0]:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: no spike = no transmission
# ---------------------------------------------------------------------------

class TestNoSpikeNoTransmission:
    def test_scale_zero_without_spike(self):
        """When no neuron spikes, all scale values should be 0."""
        n = 5
        params = create_stp_params(n, _exc_mask(n))
        state = init_stp_state(n, params)

        state, scale = stp_step(state, params, _no_spike(n), dt=0.5)
        assert jnp.allclose(scale, 0.0), f"Expected all zeros, got {scale}"


# ---------------------------------------------------------------------------
# Tests: first-spike values
# ---------------------------------------------------------------------------

class TestFirstSpikeExcitatory:
    def test_first_spike_scale(self):
        """First spike from excitatory neuron with x=1, u=U=0.5.

        After continuous decay for one dt step (negligible at dt=0.5 / tau_rec=800),
        u_spike = u + U*(1-u).  With u ~ U = 0.5:
          u_spike ~ 0.5 + 0.5*(1-0.5) = 0.75
          scale = u_spike * x ~ 0.75 * 1.0 = 0.75

        But more precisely, u first decays: u = 0.5 + 0.5*(0.5-0.5)/0.001 (huge
        decay toward U, but u already equals U so no change).  Actually
        u = u + dt*(U-u)/tau_fac = 0.5 + 0.5*(0.5-0.5)/0.001 = 0.5.
        x = 1 + 0.5*(1-1)/800 = 1.  So u_spike = 0.5 + 0.5*0.5 = 0.75.
        scale = 0.75 * 1.0 = 0.75.
        """
        n = 1
        params = create_stp_params(n, _exc_mask(n))
        state = init_stp_state(n, params)
        spike = jnp.array([True])

        _, scale = stp_step(state, params, spike, dt=0.5)

        # u_spike = 0.5 + 0.5*(1-0.5) = 0.75, scale = 0.75 * 1.0
        assert jnp.isclose(scale[0], 0.75, atol=1e-4), (
            f"Expected scale ~0.75, got {scale[0]}"
        )


class TestFirstSpikeInhibitory:
    def test_first_spike_scale(self):
        """First spike from inhibitory neuron with x=1, u=U=0.04.

        u_spike = u + U*(1-u) = 0.04 + 0.04*(1-0.04) = 0.04 + 0.0384 = 0.0784.
        scale = u_spike * x = 0.0784 * 1.0 = 0.0784.
        """
        n = 1
        params = create_stp_params(n, _inh_mask(n))
        state = init_stp_state(n, params)
        spike = jnp.array([True])

        _, scale = stp_step(state, params, spike, dt=0.5)

        expected = 0.04 + 0.04 * (1.0 - 0.04)  # 0.0784
        assert jnp.isclose(scale[0], expected, atol=1e-4), (
            f"Expected scale ~{expected:.4f}, got {scale[0]}"
        )
