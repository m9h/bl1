"""Tests for axonal conduction delay buffer (bl1.core.delays).

Uses small deterministic networks (2-10 neurons) with known delay patterns
to verify that spikes are delivered to postsynaptic targets at the correct
time, and that instantaneous transmission is prevented.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from bl1.core.delays import (
    DelayBufferState,
    compute_max_delay,
    delay_buffer_step,
    delays_to_dense,
    init_delay_buffer,
    read_delayed_spikes,
)


# ---------------------------------------------------------------------------
# 1. init_delay_buffer
# ---------------------------------------------------------------------------


def test_init_delay_buffer():
    """Buffer should be zeros with correct shape."""
    n_neurons = 10
    max_delay = 5
    state = init_delay_buffer(n_neurons, max_delay)

    assert state.buffer.shape == (max_delay, n_neurons)
    np.testing.assert_array_equal(state.buffer, 0.0)
    assert int(state.head) == 0


# ---------------------------------------------------------------------------
# 2. delay_buffer_step
# ---------------------------------------------------------------------------


def test_delay_buffer_step():
    """Writing spikes should appear in buffer at head position."""
    n_neurons = 5
    max_delay = 4
    state = init_delay_buffer(n_neurons, max_delay)

    # Write spikes from neuron 0 and 3
    spikes = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])
    state = delay_buffer_step(state, spikes)

    # Head was 0, so spikes should be at row 0, head now 1
    assert int(state.head) == 1
    np.testing.assert_array_equal(state.buffer[0], spikes)
    # Other rows still zero
    np.testing.assert_array_equal(state.buffer[1], 0.0)


def test_delay_buffer_step_sequential():
    """Sequential writes should fill successive rows."""
    n = 3
    max_d = 4
    state = init_delay_buffer(n, max_d)

    for t in range(3):
        spikes = jnp.zeros(n).at[t].set(1.0)
        state = delay_buffer_step(state, spikes)

    # Row 0: neuron 0 spiked, row 1: neuron 1 spiked, row 2: neuron 2 spiked
    assert float(state.buffer[0, 0]) == 1.0
    assert float(state.buffer[1, 1]) == 1.0
    assert float(state.buffer[2, 2]) == 1.0
    assert int(state.head) == 3


# ---------------------------------------------------------------------------
# 3. read_delayed_spikes with unit delay (all delays = 1)
# ---------------------------------------------------------------------------


def test_read_delayed_spikes_unit_delay():
    """With all delays=1, read_delayed_spikes should equal weights @ previous_spikes."""
    N = 5
    max_delay = 3
    state = init_delay_buffer(N, max_delay)

    # All-to-all unit weights, all delays = 1
    weights = jnp.ones((N, N)) - jnp.eye(N)  # no self-connections
    delay_matrix = jnp.ones((N, N), dtype=jnp.int32)

    # Step 0: neurons 0 and 1 spike
    spikes_t0 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    state = delay_buffer_step(state, spikes_t0)

    # Now read at step 1 (after the buffer step above, head=1)
    # Delay=1 means we look 1 step back = head-1 = row 0 = spikes_t0
    I_post = read_delayed_spikes(state, delay_matrix, weights)

    expected = weights @ spikes_t0
    np.testing.assert_allclose(I_post, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. read_delayed_spikes with varied delays
# ---------------------------------------------------------------------------


def test_read_delayed_spikes_varied():
    """Two pre-neurons with different delays: verify spikes arrive at correct times.

    Setup:
    - 3 neurons, one post-synaptic neuron (neuron 2)
    - Neuron 0 -> 2 with delay=1, weight=1.0
    - Neuron 1 -> 2 with delay=3, weight=2.0
    - Both pre-neurons spike at t=0

    Expected:
    - At t=1 (head=1): neuron 2 receives 1.0 from neuron 0
    - At t=2 (head=2): neuron 2 receives 0.0 (no new spikes arrive)
    - At t=3 (head=3): neuron 2 receives 2.0 from neuron 1
    """
    N = 3
    max_delay = 4

    # Only connections 0->2 and 1->2
    weights = jnp.zeros((N, N))
    weights = weights.at[2, 0].set(1.0)  # 0->2, weight=1
    weights = weights.at[2, 1].set(2.0)  # 1->2, weight=2

    delay_matrix = jnp.zeros((N, N), dtype=jnp.int32)
    delay_matrix = delay_matrix.at[2, 0].set(1)  # 0->2, delay=1
    delay_matrix = delay_matrix.at[2, 1].set(3)  # 1->2, delay=3

    state = init_delay_buffer(N, max_delay)

    # t=0: both pre-neurons spike
    spikes = jnp.array([1.0, 1.0, 0.0])
    state = delay_buffer_step(state, spikes)  # head -> 1

    # t=1: no new spikes, read output
    state = delay_buffer_step(state, jnp.zeros(N))  # head -> 2
    # At head=2, delay=1 reads head-1=1 (the zeros), delay=3 reads head-3=-1 (buffer wraparound, zeros)
    # Wait -- we need to be more careful. After writing spikes at head=0,
    # the buffer step advanced head to 1. Then we wrote zeros at head=1,
    # advancing to 2.
    # For delay=1: look at head-1=1, which has zeros (written at second step)
    # For delay=3: look at head-3=-1 % 4 = 3, which has zeros (init)
    # Actually, we should read BEFORE the next write to see the delay=1 result.

    # Let me redo with clearer timing.
    state2 = init_delay_buffer(N, max_delay)

    # t=0: both neurons spike, write into buffer
    state2 = delay_buffer_step(state2, spikes)  # head=1; buffer[0] = spikes

    # Read at head=1: delay=1 reads buffer[(1-1)%4] = buffer[0] = spikes from t=0
    I_t1 = read_delayed_spikes(state2, delay_matrix, weights)
    # Only the 0->2 synapse (delay=1) delivers: weight * spike = 1.0 * 1.0 = 1.0
    assert abs(float(I_t1[2]) - 1.0) < 1e-5, f"Expected 1.0 at t=1, got {float(I_t1[2])}"

    # t=1: no new spikes
    state2 = delay_buffer_step(state2, jnp.zeros(N))  # head=2
    I_t2 = read_delayed_spikes(state2, delay_matrix, weights)
    # delay=1 reads buffer[(2-1)%4]=buffer[1]=zeros; delay=3 reads buffer[(2-3)%4]=buffer[3]=zeros
    assert abs(float(I_t2[2])) < 1e-5, f"Expected 0.0 at t=2, got {float(I_t2[2])}"

    # t=2: no new spikes
    state2 = delay_buffer_step(state2, jnp.zeros(N))  # head=3
    I_t3 = read_delayed_spikes(state2, delay_matrix, weights)
    # delay=1 reads buffer[(3-1)%4]=buffer[2]=zeros
    # delay=3 reads buffer[(3-3)%4]=buffer[0]=spikes from t=0
    # 1->2 synapse: weight=2.0 * spike=1.0 = 2.0
    assert abs(float(I_t3[2]) - 2.0) < 1e-5, f"Expected 2.0 at t=3, got {float(I_t3[2])}"


# ---------------------------------------------------------------------------
# 5. Ring buffer wrapping
# ---------------------------------------------------------------------------


def test_delay_buffer_ring():
    """After writing more than max_delay spikes, old entries should be overwritten."""
    N = 4
    max_delay = 3
    state = init_delay_buffer(N, max_delay)

    # Write max_delay + 2 rounds of spikes
    for t in range(max_delay + 2):
        spikes = jnp.full(N, float(t + 1))
        state = delay_buffer_step(state, spikes)

    # After 5 writes to a buffer of size 3, the buffer should contain
    # the last 3 writes: values 3.0, 4.0, 5.0.
    # Write sequence:
    #   t=0: head=0, buffer[0%3=0]=1.0, head->1
    #   t=1: head=1, buffer[1%3=1]=2.0, head->2
    #   t=2: head=2, buffer[2%3=2]=3.0, head->3
    #   t=3: head=3, buffer[3%3=0]=4.0, head->4
    #   t=4: head=4, buffer[4%3=1]=5.0, head->5
    # So: buffer[0]=4.0, buffer[1]=5.0, buffer[2]=3.0
    assert float(state.buffer[0, 0]) == 4.0
    assert float(state.buffer[1, 0]) == 5.0  # most recent
    assert float(state.buffer[2, 0]) == 3.0  # oldest surviving


# ---------------------------------------------------------------------------
# 6. No instantaneous transmission
# ---------------------------------------------------------------------------


def test_no_instantaneous():
    """With all delays >= 1, spikes from the current timestep should NOT appear in output.

    This is the critical safety check: the delay buffer must enforce causal
    ordering.  A spike written at the current head position should not be
    readable at delay=0 (which we never use).
    """
    N = 4
    max_delay = 3
    state = init_delay_buffer(N, max_delay)

    # All connections with delay=1
    weights = jnp.ones((N, N))
    delay_matrix = jnp.ones((N, N), dtype=jnp.int32)

    # Spike from neuron 0 at this timestep
    spikes = jnp.array([1.0, 0.0, 0.0, 0.0])

    # Write the spike (head advances from 0 to 1)
    state = delay_buffer_step(state, spikes)

    # Now read delayed spikes -- delay=1 looks at buffer[(1-1)%3]=buffer[0]
    # which IS the spike we just wrote. This is correct: the spike entered
    # the buffer one step ago from the perspective of "current head".
    #
    # But what about delay=0 (instantaneous)? Our delay_matrix never has
    # delay=0, so it should never contribute.  Let's verify by checking
    # that zero-delay synapses don't transmit.
    delay_zero = jnp.zeros((N, N), dtype=jnp.int32)
    I_zero = read_delayed_spikes(state, delay_zero, weights)

    # Since scan range is arange(1, max_delay+1), delay=0 is never matched
    np.testing.assert_allclose(I_zero, 0.0, atol=1e-7)

    # Also verify that with delay=1, the spike IS delivered (sanity check)
    I_one = read_delayed_spikes(state, delay_matrix, weights)
    assert float(I_one[0]) > 0.0  # neuron 0 receives input from itself via delay=1


# ---------------------------------------------------------------------------
# 7. compute_max_delay
# ---------------------------------------------------------------------------


def test_compute_max_delay_dense():
    """compute_max_delay on a dense array returns correct maximum."""
    delays = jnp.array([[0, 3, 1], [2, 0, 5], [1, 4, 0]])
    assert compute_max_delay(delays) == 5


def test_compute_max_delay_bcoo():
    """compute_max_delay on a BCOO sparse array returns correct maximum."""
    dense = jnp.array([[0, 3, 0], [2, 0, 0], [0, 4, 0]], dtype=jnp.float32)
    sp = BCOO.fromdense(dense)
    assert compute_max_delay(sp) == 4


# ---------------------------------------------------------------------------
# 8. delays_to_dense
# ---------------------------------------------------------------------------


def test_delays_to_dense_identity():
    """A dense array passed through delays_to_dense should be unchanged."""
    delays = jnp.array([[0, 3, 1], [2, 0, 5], [1, 4, 0]])
    result = delays_to_dense(delays)
    np.testing.assert_array_equal(result, delays)


def test_delays_to_dense_bcoo():
    """BCOO -> dense conversion preserves values."""
    dense = jnp.array([[0, 3, 0], [2, 0, 0], [0, 4, 0]], dtype=jnp.float32)
    sp = BCOO.fromdense(dense)
    result = delays_to_dense(sp)
    np.testing.assert_array_equal(result, dense)


# ---------------------------------------------------------------------------
# 9. Integration: multi-step simulation with delay buffer
# ---------------------------------------------------------------------------


def test_integration_delay_propagation():
    """End-to-end test: run several timesteps and verify correct spike delivery.

    Network: 5 neurons, sparse connectivity.
    - 0 -> 1 (delay=2, weight=1.0)
    - 0 -> 2 (delay=4, weight=0.5)
    - 3 -> 4 (delay=1, weight=3.0)

    Neuron 0 spikes at t=0, neuron 3 spikes at t=1.
    """
    N = 5
    max_delay = 5

    weights = jnp.zeros((N, N))
    weights = weights.at[1, 0].set(1.0)
    weights = weights.at[2, 0].set(0.5)
    weights = weights.at[4, 3].set(3.0)

    delay_matrix = jnp.zeros((N, N), dtype=jnp.int32)
    delay_matrix = delay_matrix.at[1, 0].set(2)
    delay_matrix = delay_matrix.at[2, 0].set(4)
    delay_matrix = delay_matrix.at[4, 3].set(1)

    state = init_delay_buffer(N, max_delay)

    # Collect postsynaptic input at each step
    results = []

    # t=0: neuron 0 spikes
    state = delay_buffer_step(state, jnp.array([1, 0, 0, 0, 0], dtype=jnp.float32))
    results.append(np.array(read_delayed_spikes(state, delay_matrix, weights)))

    # t=1: neuron 3 spikes
    state = delay_buffer_step(state, jnp.array([0, 0, 0, 1, 0], dtype=jnp.float32))
    results.append(np.array(read_delayed_spikes(state, delay_matrix, weights)))

    # t=2: no spikes
    state = delay_buffer_step(state, jnp.zeros(N))
    results.append(np.array(read_delayed_spikes(state, delay_matrix, weights)))

    # t=3: no spikes
    state = delay_buffer_step(state, jnp.zeros(N))
    results.append(np.array(read_delayed_spikes(state, delay_matrix, weights)))

    # t=4: no spikes
    state = delay_buffer_step(state, jnp.zeros(N))
    results.append(np.array(read_delayed_spikes(state, delay_matrix, weights)))

    # Verify:
    # t=0 (head=1): nothing arrives yet (delay>=1, spike just entered)
    #   delay=2 for 0->1: reads head-2= -1 %5 = 4, buffer[4]=zeros. No.
    #   Wait: head=1 after first write. delay=2 reads (1-2)%5=4 which is zeros.
    #   delay=1 for 3->4 reads (1-1)%5=0 which is [1,0,0,0,0] but 3->4 needs neuron 3.
    #   neuron 3 didn't spike, so 0.
    np.testing.assert_allclose(results[0][1], 0.0, atol=1e-5)  # neuron 1: no arrival yet
    np.testing.assert_allclose(results[0][4], 0.0, atol=1e-5)  # neuron 4: no arrival yet

    # t=1 (head=2): delay=1 for 3->4 reads (2-1)%5=1, buffer[1]=[0,0,0,1,0]
    #   neuron 3 spike -> neuron 4: 3.0 * 1.0 = 3.0
    #   delay=2 for 0->1 reads (2-2)%5=0, buffer[0]=[1,0,0,0,0]
    #   neuron 0 spike -> neuron 1: 1.0 * 1.0 = 1.0
    np.testing.assert_allclose(results[1][1], 1.0, atol=1e-5)
    np.testing.assert_allclose(results[1][4], 3.0, atol=1e-5)

    # t=2 (head=3): delay=4 for 0->2 reads (3-4)%5=4, buffer[4]=zeros. No.
    np.testing.assert_allclose(results[2][2], 0.0, atol=1e-5)

    # t=3 (head=4): delay=4 for 0->2 reads (4-4)%5=0, buffer[0]=[1,0,0,0,0]
    #   neuron 0 spike -> neuron 2: 0.5 * 1.0 = 0.5
    np.testing.assert_allclose(results[3][2], 0.5, atol=1e-5)
