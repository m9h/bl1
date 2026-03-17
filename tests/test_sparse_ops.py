"""Tests for fast sparse synaptic computation (bl1.core.sparse_ops).

Verifies that :func:`fast_sparse_input` produces results identical to
BCOO matmul, and that the conversion helper preserves data.
"""

import timeit

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.experimental.sparse import BCOO

from bl1.core.sparse_ops import (
    RawSparseWeights,
    bcoo_to_raw,
    fast_sparse_input,
    fast_sparse_input_raw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_bcoo(key, N, density=0.05):
    """Create a random BCOO sparse matrix with given density."""
    k1, k2 = jax.random.split(key)
    dense = jax.random.uniform(k1, (N, N))
    mask = jax.random.uniform(k2, (N, N)) < density
    dense = jnp.where(mask, dense, 0.0)
    return BCOO.fromdense(dense)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fast_sparse_matches_bcoo_small():
    """fast_sparse_input produces the same result as BCOO @ vector (small)."""
    key = jax.random.PRNGKey(42)
    N = 100
    W = _make_random_bcoo(key, N, density=0.1)
    raw = bcoo_to_raw(W)

    k2 = jax.random.PRNGKey(99)
    spikes = (jax.random.uniform(k2, (N,)) < 0.05).astype(jnp.float32)

    expected = W @ spikes
    actual = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_fast_sparse_matches_bcoo_medium():
    """fast_sparse_input matches BCOO @ vector for a 1K network."""
    key = jax.random.PRNGKey(7)
    N = 1000
    W = _make_random_bcoo(key, N, density=0.03)
    raw = bcoo_to_raw(W)

    k2 = jax.random.PRNGKey(13)
    spikes = (jax.random.uniform(k2, (N,)) < 0.01).astype(jnp.float32)

    expected = W @ spikes
    actual = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_fast_sparse_matches_bcoo_scaled():
    """fast_sparse_input works with non-binary spike amplitudes (STP)."""
    key = jax.random.PRNGKey(21)
    N = 500
    W = _make_random_bcoo(key, N, density=0.05)
    raw = bcoo_to_raw(W)

    k2 = jax.random.PRNGKey(77)
    # STP-like scaled spikes: some neurons fire with varying amplitudes
    spikes = jax.random.uniform(k2, (N,)) * (jax.random.uniform(jax.random.PRNGKey(88), (N,)) < 0.03).astype(jnp.float32)

    expected = W @ spikes
    actual = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)

    npt.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_fast_sparse_all_zero_spikes():
    """No spikes should produce zero input."""
    key = jax.random.PRNGKey(10)
    N = 200
    W = _make_random_bcoo(key, N, density=0.1)
    raw = bcoo_to_raw(W)

    spikes = jnp.zeros(N, dtype=jnp.float32)
    result = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)

    npt.assert_allclose(np.asarray(result), 0.0, atol=1e-7)


def test_fast_sparse_input_raw_wrapper():
    """fast_sparse_input_raw convenience wrapper matches direct call."""
    key = jax.random.PRNGKey(55)
    N = 100
    W = _make_random_bcoo(key, N, density=0.1)
    raw = bcoo_to_raw(W)

    spikes = (jax.random.uniform(jax.random.PRNGKey(66), (N,)) < 0.1).astype(jnp.float32)

    direct = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post)
    wrapped = fast_sparse_input_raw(raw, spikes)

    npt.assert_allclose(np.asarray(wrapped), np.asarray(direct), atol=1e-7)


def test_bcoo_to_raw_preserves_data():
    """bcoo_to_raw should preserve data, rows, cols from the BCOO matrix."""
    key = jax.random.PRNGKey(123)
    N = 50
    W = _make_random_bcoo(key, N, density=0.2)
    raw = bcoo_to_raw(W)

    assert raw.n_post == N

    # Data should match
    npt.assert_array_equal(np.asarray(raw.data), np.asarray(W.data))

    # Indices should match the two columns of W.indices
    npt.assert_array_equal(np.asarray(raw.rows), np.asarray(W.indices[:, 0]))
    npt.assert_array_equal(np.asarray(raw.cols), np.asarray(W.indices[:, 1]))


def test_bcoo_to_raw_type():
    """bcoo_to_raw should return a RawSparseWeights NamedTuple."""
    key = jax.random.PRNGKey(0)
    W = _make_random_bcoo(key, 10, density=0.5)
    raw = bcoo_to_raw(W)
    assert isinstance(raw, RawSparseWeights)
    assert hasattr(raw, "data")
    assert hasattr(raw, "rows")
    assert hasattr(raw, "cols")
    assert hasattr(raw, "n_post")


def test_fast_sparse_speed():
    """fast_sparse_input should be at least as fast as BCOO matmul for 5K neurons.

    This is a soft timing test — it verifies the fast path is not
    dramatically slower.  On GPU the speedup is more pronounced.
    """
    key = jax.random.PRNGKey(42)
    N = 5000
    W = _make_random_bcoo(key, N, density=0.01)
    raw = bcoo_to_raw(W)

    k2 = jax.random.PRNGKey(99)
    spikes = (jax.random.uniform(k2, (N,)) < 0.005).astype(jnp.float32)

    # Warmup both paths (JIT compilation)
    _ = (W @ spikes).block_until_ready()
    _ = fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post).block_until_ready()

    n_repeats = 20

    def bcoo_fn():
        return (W @ spikes).block_until_ready()

    def fast_fn():
        return fast_sparse_input(raw.data, raw.rows, raw.cols, spikes, raw.n_post).block_until_ready()

    t_bcoo = timeit.timeit(bcoo_fn, number=n_repeats) / n_repeats
    t_fast = timeit.timeit(fast_fn, number=n_repeats) / n_repeats

    # The fast path should not be more than 3x slower than BCOO
    # (in practice it should be faster, especially on GPU)
    assert t_fast < t_bcoo * 3.0, (
        f"fast_sparse_input ({t_fast*1000:.2f}ms) is more than 3x slower "
        f"than BCOO matmul ({t_bcoo*1000:.2f}ms)"
    )
