"""Tests for STDP plasticity (bl1.plasticity.stdp)."""

import math

import jax.numpy as jnp
import pytest

from bl1.plasticity.stdp import STDPParams, STDPState, init_stdp_state, stdp_update


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_params() -> STDPParams:
    return STDPParams()  # uses class defaults


def _no_spikes(n: int) -> jnp.ndarray:
    return jnp.zeros(n, dtype=jnp.bool_)


def _all_excitatory(n: int) -> jnp.ndarray:
    return jnp.ones(n, dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitSTDPState:
    def test_init_stdp_state(self):
        """init_stdp_state(100) returns zero traces of shape (100,)."""
        state = init_stdp_state(100)
        assert state.pre_trace.shape == (100,)
        assert state.post_trace.shape == (100,)
        assert jnp.allclose(state.pre_trace, 0.0)
        assert jnp.allclose(state.post_trace, 0.0)


class TestTraceDecay:
    def test_trace_decay_one_step(self):
        """Pre-trace of 1.0 with no spikes decays to exp(-dt/tau_plus) after one step."""
        params = _default_params()
        n = 50
        state = STDPState(
            pre_trace=jnp.ones(n, dtype=jnp.float32),
            post_trace=jnp.zeros(n, dtype=jnp.float32),
        )
        W = jnp.zeros((n, n), dtype=jnp.float32)
        is_exc = _all_excitatory(n)
        spikes = _no_spikes(n)

        new_state, _ = stdp_update(state, params, spikes, W, is_exc)

        expected = math.exp(-0.5 / 20.0)  # dt=0.5, tau_plus=20
        assert jnp.allclose(new_state.pre_trace, expected, atol=1e-6)

    def test_trace_decay_many_steps(self):
        """After 500 steps (250ms) with no spikes the trace should approach zero."""
        params = _default_params()
        n = 50
        state = STDPState(
            pre_trace=jnp.ones(n, dtype=jnp.float32),
            post_trace=jnp.zeros(n, dtype=jnp.float32),
        )
        W = jnp.zeros((n, n), dtype=jnp.float32)
        is_exc = _all_excitatory(n)
        spikes = _no_spikes(n)

        for _ in range(500):
            state, W = stdp_update(state, params, spikes, W, is_exc)

        # After 250ms with tau_plus=20ms: exp(-250/20) ≈ 3.7e-6
        assert jnp.all(state.pre_trace < 1e-3)


class TestTraceIncrementOnSpike:
    def test_trace_increment_on_spike(self):
        """A single spike should bump pre_trace by A_plus and post_trace by A_minus."""
        params = _default_params()
        n = 10
        state = init_stdp_state(n)
        W = jnp.zeros((n, n), dtype=jnp.float32)
        is_exc = _all_excitatory(n)

        # Neuron 3 fires
        spikes = jnp.zeros(n, dtype=jnp.bool_).at[3].set(True)
        new_state, _ = stdp_update(state, params, spikes, W, is_exc)

        # Traces start at 0, decay is identity for 0, then increment by A_plus / A_minus
        assert jnp.isclose(new_state.pre_trace[3], params.A_plus, atol=1e-7)
        assert jnp.isclose(new_state.post_trace[3], params.A_minus, atol=1e-7)

        # Other neurons should still be zero
        others = jnp.delete(jnp.arange(n), 3)
        assert jnp.allclose(new_state.pre_trace[others], 0.0)
        assert jnp.allclose(new_state.post_trace[others], 0.0)


class TestLTP:
    def test_ltp_pre_before_post(self):
        """Pre fires first, then post fires => weight should increase (LTP)."""
        params = _default_params()
        n = 2
        state = init_stdp_state(n)

        # W[1, 0] = 0.05 means connection from neuron 0 (pre) to neuron 1 (post)
        W = jnp.zeros((n, n), dtype=jnp.float32).at[1, 0].set(0.05)
        is_exc = _all_excitatory(n)

        # Step 1: neuron 0 fires (pre) -- builds up pre_trace[0]
        spikes_pre = jnp.array([True, False])
        state, W = stdp_update(state, params, spikes_pre, W, is_exc)

        # Step 2: neuron 1 fires (post) -- should trigger LTP on W[1, 0]
        spikes_post = jnp.array([False, True])
        state, W = stdp_update(state, params, spikes_post, W, is_exc)

        assert W[1, 0] > 0.05, f"Expected LTP: weight should increase from 0.05, got {W[1, 0]}"


class TestLTD:
    def test_ltd_post_before_pre(self):
        """Post fires first, then pre fires => weight should decrease (LTD)."""
        params = _default_params()
        n = 2
        state = init_stdp_state(n)

        # W[1, 0] = 0.05 means connection from neuron 0 (pre) to neuron 1 (post)
        W = jnp.zeros((n, n), dtype=jnp.float32).at[1, 0].set(0.05)
        is_exc = _all_excitatory(n)

        # Step 1: neuron 1 fires (post) -- builds up post_trace[1]
        spikes_post = jnp.array([False, True])
        state, W = stdp_update(state, params, spikes_post, W, is_exc)

        # Step 2: neuron 0 fires (pre) -- should trigger LTD on W[1, 0]
        spikes_pre = jnp.array([True, False])
        state, W = stdp_update(state, params, spikes_pre, W, is_exc)

        assert W[1, 0] < 0.05, f"Expected LTD: weight should decrease from 0.05, got {W[1, 0]}"


class TestWeightBounds:
    def test_weight_does_not_exceed_w_max(self):
        """Repeated LTP should not push weight above w_max."""
        params = _default_params()
        n = 2
        state = init_stdp_state(n)

        # Start near w_max
        W = jnp.zeros((n, n), dtype=jnp.float32).at[1, 0].set(0.095)
        is_exc = _all_excitatory(n)

        # Repeatedly fire pre then post to drive LTP
        for _ in range(200):
            spikes_pre = jnp.array([True, False])
            state, W = stdp_update(state, params, spikes_pre, W, is_exc)
            spikes_post = jnp.array([False, True])
            state, W = stdp_update(state, params, spikes_post, W, is_exc)

        assert W[1, 0] <= params.w_max + 1e-7, (
            f"Weight {W[1, 0]} exceeded w_max {params.w_max}"
        )

    def test_weight_does_not_go_below_w_min(self):
        """Repeated LTD should not push weight below w_min."""
        params = _default_params()
        n = 2
        state = init_stdp_state(n)

        # Start near w_min
        W = jnp.zeros((n, n), dtype=jnp.float32).at[1, 0].set(0.005)
        is_exc = _all_excitatory(n)

        # Repeatedly fire post then pre to drive LTD
        for _ in range(200):
            spikes_post = jnp.array([False, True])
            state, W = stdp_update(state, params, spikes_post, W, is_exc)
            spikes_pre = jnp.array([True, False])
            state, W = stdp_update(state, params, spikes_pre, W, is_exc)

        assert W[1, 0] >= params.w_min - 1e-7, (
            f"Weight {W[1, 0]} went below w_min {params.w_min}"
        )


class TestInhibitoryWeightsUnchanged:
    def test_inhibitory_weights_unchanged(self):
        """STDP should not modify weights from inhibitory pre-synaptic neurons."""
        params = _default_params()
        n = 3

        # Neuron 0 is inhibitory, neurons 1 and 2 are excitatory
        is_exc = jnp.array([False, True, True])

        # Connection from neuron 0 (inhibitory) to neuron 1: W[1, 0] = 0.05
        W = jnp.zeros((n, n), dtype=jnp.float32).at[1, 0].set(0.05)
        state = init_stdp_state(n)

        initial_weight = W[1, 0].item()

        # Fire both neurons in patterns that would normally trigger LTP/LTD
        for _ in range(50):
            # Pre fires
            spikes_pre = jnp.array([True, False, False])
            state, W = stdp_update(state, params, spikes_pre, W, is_exc)
            # Post fires
            spikes_post = jnp.array([False, True, False])
            state, W = stdp_update(state, params, spikes_post, W, is_exc)

        assert jnp.isclose(W[1, 0], initial_weight, atol=1e-7), (
            f"Inhibitory weight changed from {initial_weight} to {W[1, 0]}"
        )
