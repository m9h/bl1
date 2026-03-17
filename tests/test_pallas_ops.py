"""Tests for event-driven CSC sparse kernels (bl1.core.pallas_ops).

Verifies that:
1. BCOO -> CSC conversion preserves matrix structure
2. CSC event-driven matmul matches dense matmul
3. Sparse spikes produce correct results
4. max_active overflow is handled gracefully
5. Performance comparison (informational)
"""

import timeit

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.experimental.sparse import BCOO

from bl1.core.pallas_ops import (
    CSCWeights,
    bcoo_to_csc,
    csc_event_driven_input,
    event_driven_input,
    is_pallas_available,
)
from bl1.core.sparse_ops import bcoo_to_raw, fast_sparse_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_bcoo(key, n_rows, n_cols=None, density=0.05):
    """Create a random BCOO sparse matrix with given density."""
    if n_cols is None:
        n_cols = n_rows
    k1, k2 = jax.random.split(key)
    dense = jax.random.uniform(k1, (n_rows, n_cols))
    mask = jax.random.uniform(k2, (n_rows, n_cols)) < density
    dense = jnp.where(mask, dense, 0.0)
    return BCOO.fromdense(dense), dense


def _dense_matmul(dense_W, spikes):
    """Reference dense matmul."""
    return dense_W @ spikes


# ---------------------------------------------------------------------------
# Test 1: BCOO -> CSC conversion
# ---------------------------------------------------------------------------

def test_bcoo_to_csc_preserves_structure():
    """CSC conversion should preserve the matrix structure (dense round-trip)."""
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)

    csc = bcoo_to_csc(W_bcoo)

    assert isinstance(csc, CSCWeights)
    assert csc.n_post == N
    assert csc.n_pre == N
    assert csc.col_ptr.shape == (N + 1,)

    # Reconstruct dense matrix from CSC and verify it matches
    W_reconstructed = np.zeros((N, N), dtype=np.float32)
    col_ptr = np.asarray(csc.col_ptr)
    row_indices = np.asarray(csc.row_indices)
    data = np.asarray(csc.data)

    for col in range(N):
        start = col_ptr[col]
        end = col_ptr[col + 1]
        for idx in range(start, end):
            row = row_indices[idx]
            W_reconstructed[row, col] = data[idx]

    npt.assert_allclose(W_reconstructed, np.asarray(W_dense), atol=1e-6)


def test_bcoo_to_csc_col_ptr_monotonic():
    """col_ptr should be monotonically non-decreasing."""
    key = jax.random.PRNGKey(7)
    N = 200
    W_bcoo, _ = _make_random_bcoo(key, N, density=0.05)
    csc = bcoo_to_csc(W_bcoo)

    col_ptr = np.asarray(csc.col_ptr)
    assert np.all(col_ptr[1:] >= col_ptr[:-1])
    assert col_ptr[0] == 0
    assert col_ptr[-1] == csc.data.shape[0]


def test_bcoo_to_csc_max_synapses():
    """max_synapses_per_neuron should match the actual maximum."""
    key = jax.random.PRNGKey(13)
    N = 50
    W_bcoo, _ = _make_random_bcoo(key, N, density=0.2)
    csc = bcoo_to_csc(W_bcoo)

    col_ptr = np.asarray(csc.col_ptr)
    counts = col_ptr[1:] - col_ptr[:-1]
    assert csc.max_synapses_per_neuron == int(np.max(counts))


def test_bcoo_to_csc_rectangular():
    """CSC conversion works for rectangular matrices."""
    key = jax.random.PRNGKey(21)
    W_bcoo, W_dense = _make_random_bcoo(key, n_rows=80, n_cols=120, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    assert csc.n_post == 80
    assert csc.n_pre == 120
    assert csc.col_ptr.shape == (121,)

    # Verify round-trip
    W_reconstructed = np.zeros((80, 120), dtype=np.float32)
    col_ptr = np.asarray(csc.col_ptr)
    row_indices = np.asarray(csc.row_indices)
    data = np.asarray(csc.data)

    for col in range(120):
        start = col_ptr[col]
        end = col_ptr[col + 1]
        for idx in range(start, end):
            W_reconstructed[row_indices[idx], col] = data[idx]

    npt.assert_allclose(W_reconstructed, np.asarray(W_dense), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: CSC event-driven matches dense matmul
# ---------------------------------------------------------------------------

def test_csc_event_driven_matches_dense():
    """CSC event-driven input should match dense W @ spikes."""
    key = jax.random.PRNGKey(42)
    N = 200
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    k2 = jax.random.PRNGKey(99)
    # ~5% of neurons spiking
    spikes = (jax.random.uniform(k2, (N,)) < 0.05).astype(jnp.float32)

    expected = W_dense @ spikes
    actual = csc_event_driven_input(csc, spikes, max_active=100)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_csc_event_driven_matches_dense_medium():
    """CSC event-driven input matches for a 1K network."""
    key = jax.random.PRNGKey(7)
    N = 1000
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.03)
    csc = bcoo_to_csc(W_bcoo)

    k2 = jax.random.PRNGKey(13)
    spikes = (jax.random.uniform(k2, (N,)) < 0.01).astype(jnp.float32)

    expected = W_dense @ spikes
    actual = csc_event_driven_input(csc, spikes, max_active=100)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-4)


# ---------------------------------------------------------------------------
# Test 3: Sparse spikes
# ---------------------------------------------------------------------------

def test_csc_event_driven_sparse_spikes():
    """With only ~1% neurons spiking, result should match dense matmul."""
    key = jax.random.PRNGKey(55)
    N = 500
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.05)
    csc = bcoo_to_csc(W_bcoo)

    k2 = jax.random.PRNGKey(66)
    # ~1% spiking
    spikes = (jax.random.uniform(k2, (N,)) < 0.01).astype(jnp.float32)

    expected = W_dense @ spikes
    actual = csc_event_driven_input(csc, spikes, max_active=50)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_csc_event_driven_single_spike():
    """A single spiking neuron should produce correct postsynaptic input."""
    key = jax.random.PRNGKey(77)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    # Only neuron 42 spikes
    spikes = jnp.zeros(N, dtype=jnp.float32).at[42].set(1.0)

    expected = W_dense @ spikes  # = column 42 of W
    actual = csc_event_driven_input(csc, spikes, max_active=10)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_csc_event_driven_no_spikes():
    """No spikes should produce zero output."""
    key = jax.random.PRNGKey(88)
    N = 200
    W_bcoo, _ = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    spikes = jnp.zeros(N, dtype=jnp.float32)
    result = csc_event_driven_input(csc, spikes, max_active=50)

    npt.assert_allclose(np.asarray(result), 0.0, atol=1e-7)


def test_csc_event_driven_scaled_spikes():
    """Non-binary spike amplitudes (STP-like) should scale correctly."""
    key = jax.random.PRNGKey(21)
    N = 300
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.05)
    csc = bcoo_to_csc(W_bcoo)

    k2 = jax.random.PRNGKey(77)
    # STP-like: some neurons have varying amplitudes
    mask = (jax.random.uniform(jax.random.PRNGKey(88), (N,)) < 0.03).astype(jnp.float32)
    amplitudes = jax.random.uniform(k2, (N,)) * mask
    spikes = amplitudes

    expected = W_dense @ spikes
    actual = csc_event_driven_input(csc, spikes, max_active=50)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: max_active overflow
# ---------------------------------------------------------------------------

def test_csc_max_active_overflow():
    """When more neurons spike than max_active, result is still reasonable.

    Some spikes may be dropped, but the result should be a valid
    partial computation (not NaN or crash).
    """
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    # Make 50% of neurons spike but only allow max_active=5
    k2 = jax.random.PRNGKey(99)
    spikes = (jax.random.uniform(k2, (N,)) < 0.5).astype(jnp.float32)

    result = csc_event_driven_input(csc, spikes, max_active=5)

    # Should not contain NaN
    assert not jnp.any(jnp.isnan(result)), "Result contains NaN values"

    # Should be a valid array of the right shape
    assert result.shape == (N,)

    # The result is a partial sum (only first max_active spiking neurons
    # are processed), so it should be <= the full result in magnitude
    # for each element (approximately, due to positive weights).
    # We just check it doesn't crash and returns finite values.
    assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"


def test_csc_max_active_exact():
    """When max_active >= actual spikes, result matches exactly."""
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    # Only 3 neurons spike
    spikes = jnp.zeros(N, dtype=jnp.float32).at[10].set(1.0).at[20].set(1.0).at[30].set(1.0)

    # max_active=10 is plenty
    result = csc_event_driven_input(csc, spikes, max_active=10)
    expected = W_dense @ spikes

    npt.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: event_driven_input dispatcher
# ---------------------------------------------------------------------------

def test_event_driven_input_dispatcher():
    """event_driven_input dispatches to csc_event_driven_input."""
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    spikes = (jax.random.uniform(jax.random.PRNGKey(99), (N,)) < 0.05).astype(jnp.float32)

    result_dispatch = event_driven_input(csc, spikes, max_active=50)
    result_direct = csc_event_driven_input(csc, spikes, max_active=50)

    npt.assert_allclose(np.asarray(result_dispatch), np.asarray(result_direct), atol=1e-7)


# ---------------------------------------------------------------------------
# Test 6: Pallas availability check
# ---------------------------------------------------------------------------

def test_is_pallas_available():
    """is_pallas_available should return a boolean without crashing."""
    result = is_pallas_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 7: JIT compatibility
# ---------------------------------------------------------------------------

def test_csc_event_driven_jit():
    """CSC event-driven input should work inside jax.jit."""
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    @jax.jit
    def compute(spikes):
        return csc_event_driven_input(csc, spikes, max_active=50)

    spikes = (jax.random.uniform(jax.random.PRNGKey(99), (N,)) < 0.05).astype(jnp.float32)

    result = compute(spikes)
    expected = W_dense @ spikes

    npt.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5)


def test_csc_event_driven_jit_multiple_calls():
    """JIT-compiled CSC kernel produces correct results across multiple calls."""
    key = jax.random.PRNGKey(42)
    N = 100
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.1)
    csc = bcoo_to_csc(W_bcoo)

    @jax.jit
    def compute(spikes):
        return csc_event_driven_input(csc, spikes, max_active=50)

    for i in range(5):
        k = jax.random.PRNGKey(i)
        spikes = (jax.random.uniform(k, (N,)) < 0.05).astype(jnp.float32)
        result = compute(spikes)
        expected = W_dense @ spikes
        npt.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5,
                            err_msg=f"Mismatch on call {i}")


# ---------------------------------------------------------------------------
# Test 8: Performance comparison (informational, no assertion)
# ---------------------------------------------------------------------------

def test_event_driven_performance():
    """Compare event-driven vs fast_sparse_input timing (informational).

    This test prints timing results but does not assert speed.
    The benefit is expected at 50K+ scale, not at test sizes.
    """
    key = jax.random.PRNGKey(42)
    N = 1000
    W_bcoo, W_dense = _make_random_bcoo(key, N, density=0.03)
    csc = bcoo_to_csc(W_bcoo)
    raw = bcoo_to_raw(W_bcoo)

    k2 = jax.random.PRNGKey(99)
    # ~0.5% spike rate (realistic for 5 Hz at dt=0.5ms)
    spikes = (jax.random.uniform(k2, (N,)) < 0.005).astype(jnp.float32)
    n_active = int(jnp.sum(spikes > 0))

    # Warmup both paths
    _ = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post).block_until_ready()
    _ = csc_event_driven_input(csc, spikes, max_active=100).block_until_ready()

    n_repeats = 50

    def fast_fn():
        return fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post).block_until_ready()

    def event_fn():
        return csc_event_driven_input(csc, spikes, max_active=100).block_until_ready()

    t_fast = timeit.timeit(fast_fn, number=n_repeats) / n_repeats
    t_event = timeit.timeit(event_fn, number=n_repeats) / n_repeats

    print(f"\n  N={N}, nnz={raw.data.shape[0]:,}, active={n_active}")
    print(f"  fast_sparse:  {t_fast * 1000:.3f} ms/call")
    print(f"  event_driven: {t_event * 1000:.3f} ms/call")
    print(f"  ratio: {t_fast / t_event:.2f}x" if t_event > 0 else "  ratio: N/A")

    # No assertion — this is informational. The event-driven path
    # may actually be slower at small scale due to the gather overhead.
    # The benefit appears at 50K-100K+ neurons where nnz >> active_synapses.
