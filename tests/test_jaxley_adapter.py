"""Tests for the Jaxley multi-compartment neuron adapter (bl1.core.jaxley_adapter).

Tests are split into two groups:

1. Import/availability tests -- always run, even when Jaxley is not installed.
2. Functional tests -- skipped when Jaxley is not available.
"""

import jax.numpy as jnp
import pytest

from bl1.core.jaxley_adapter import (
    JaxleyConfig,
    JaxleyNetwork,
    JaxleyState,
    is_jaxley_available,
)

# Marker for tests that require Jaxley to be installed.
requires_jaxley = pytest.mark.skipif(
    not is_jaxley_available(),
    reason="Jaxley is not installed (pip install 'bl1[jaxley]')",
)


# ---------------------------------------------------------------------------
# Group 1: import / availability tests (always run)
# ---------------------------------------------------------------------------


class TestImportWithoutJaxley:
    """These tests validate that the adapter module loads and provides useful
    metadata even when Jaxley is not installed."""

    def test_is_jaxley_available_returns_bool(self):
        """is_jaxley_available() should return a plain bool."""
        result = is_jaxley_available()
        assert isinstance(result, bool)

    def test_module_imports_without_jaxley(self):
        """The adapter module must be importable regardless of Jaxley."""
        # If we got this far the module imported.  Double-check the public
        # names are accessible.
        from bl1.core import jaxley_adapter  # noqa: F811

        assert hasattr(jaxley_adapter, "JaxleyNetwork")
        assert hasattr(jaxley_adapter, "JaxleyState")
        assert hasattr(jaxley_adapter, "JaxleyConfig")
        assert hasattr(jaxley_adapter, "is_jaxley_available")
        assert hasattr(jaxley_adapter, "make_differentiable_step")

    def test_jaxley_state_is_namedtuple(self):
        """JaxleyState should be a NamedTuple with expected fields."""
        assert issubclass(JaxleyState, tuple)
        assert "voltages" in JaxleyState._fields
        assert "spikes" in JaxleyState._fields
        assert "internal_state" in JaxleyState._fields

    def test_jaxley_config_is_namedtuple(self):
        """JaxleyConfig should be a NamedTuple with expected fields and defaults."""
        assert issubclass(JaxleyConfig, tuple)
        assert "n_neurons" in JaxleyConfig._fields
        assert "n_compartments_per_neuron" in JaxleyConfig._fields
        assert "spike_threshold_mv" in JaxleyConfig._fields
        assert "dt_ms" in JaxleyConfig._fields

        # Check defaults
        cfg = JaxleyConfig(n_neurons=5, n_compartments_per_neuron=8)
        assert cfg.spike_threshold_mv == -20.0
        assert cfg.dt_ms == 0.025

    def test_network_constructor_requires_jaxley(self):
        """JaxleyNetwork() should raise ImportError when Jaxley is absent."""
        if is_jaxley_available():
            pytest.skip("Jaxley IS installed; cannot test missing-import path")
        config = JaxleyConfig(
            n_neurons=2,
            n_compartments_per_neuron=8,
        )
        with pytest.raises(ImportError, match="[Jj]axley"):
            JaxleyNetwork(config)

    def test_from_swc_requires_jaxley(self):
        """from_swc should raise ImportError when Jaxley is absent."""
        if is_jaxley_available():
            pytest.skip("Jaxley IS installed; cannot test missing-import path")
        with pytest.raises(ImportError, match="[Jj]axley"):
            JaxleyNetwork.from_swc("fake.swc", n_neurons=1)

    def test_ball_and_stick_requires_jaxley(self):
        """ball_and_stick should raise ImportError when Jaxley is absent."""
        if is_jaxley_available():
            pytest.skip("Jaxley IS installed; cannot test missing-import path")
        with pytest.raises(ImportError, match="[Jj]axley"):
            JaxleyNetwork.ball_and_stick(n_neurons=1)

    def test_exports_from_core_init(self):
        """The public names should be re-exported from bl1.core."""
        from bl1.core import (
            JaxleyConfig,   # noqa: F811
            JaxleyNetwork,  # noqa: F811
            JaxleyState,    # noqa: F811
            is_jaxley_available,  # noqa: F811
        )
        assert callable(is_jaxley_available)


# ---------------------------------------------------------------------------
# Group 2: functional tests (require Jaxley)
# ---------------------------------------------------------------------------


@requires_jaxley
class TestBallAndStickCreation:
    """Test creating a ball-and-stick JaxleyNetwork."""

    def test_creates_network(self):
        net = JaxleyNetwork.ball_and_stick(n_neurons=4)
        assert net.config.n_neurons == 4
        assert net._network is not None

    def test_custom_compartments(self):
        net = JaxleyNetwork.ball_and_stick(n_neurons=2, n_compartments=16)
        assert net.config.n_compartments_per_neuron == 16

    def test_custom_threshold(self):
        net = JaxleyNetwork.ball_and_stick(
            n_neurons=2, spike_threshold_mv=-10.0,
        )
        assert net.config.spike_threshold_mv == -10.0


@requires_jaxley
class TestInitState:
    """Test state initialisation."""

    def test_init_state_shapes(self):
        n = 5
        n_comp = 8
        net = JaxleyNetwork.ball_and_stick(n_neurons=n, n_compartments=n_comp)
        state = net.init_state()

        assert state.voltages.shape == (n, n_comp)
        assert state.spikes.shape == (n,)
        assert state.spikes.dtype == jnp.bool_

    def test_init_state_resting_potential(self):
        net = JaxleyNetwork.ball_and_stick(n_neurons=3, n_compartments=4)
        state = net.init_state()
        # All compartments should be near -65 mV at rest
        assert jnp.allclose(state.voltages, -65.0)

    def test_init_state_no_spikes(self):
        net = JaxleyNetwork.ball_and_stick(n_neurons=3)
        state = net.init_state()
        assert not jnp.any(state.spikes)


@requires_jaxley
class TestStep:
    """Test single-step integration."""

    def test_step_returns_correct_types(self):
        net = JaxleyNetwork.ball_and_stick(n_neurons=4, n_compartments=8)
        state = net.init_state()
        I_ext = jnp.zeros(4)

        new_state, spikes = net.step(state, I_ext)

        assert isinstance(new_state, JaxleyState)
        assert spikes.shape == (4,)
        assert spikes.dtype == jnp.bool_

    def test_step_preserves_shape(self):
        n, n_comp = 3, 8
        net = JaxleyNetwork.ball_and_stick(n_neurons=n, n_compartments=n_comp)
        state = net.init_state()
        I_ext = jnp.zeros(n)

        new_state, _ = net.step(state, I_ext)
        assert new_state.voltages.shape == (n, n_comp)
        assert new_state.spikes.shape == (n,)

    def test_get_soma_voltages(self):
        n = 5
        net = JaxleyNetwork.ball_and_stick(n_neurons=n, n_compartments=8)
        state = net.init_state()
        soma_v = net.get_soma_voltages(state)
        assert soma_v.shape == (n,)


@requires_jaxley
class TestStepMultiple:
    """Test multi-step integration (time-scale bridging)."""

    def test_step_multiple_returns_correct_types(self):
        n = 4
        net = JaxleyNetwork.ball_and_stick(n_neurons=n)
        state = net.init_state()
        I_ext = jnp.zeros(n)

        new_state, any_spikes = net.step_multiple(state, I_ext, n_steps=20)

        assert isinstance(new_state, JaxleyState)
        assert any_spikes.shape == (n,)
        assert any_spikes.dtype == jnp.bool_

    def test_step_multiple_20_steps(self):
        """20 Jaxley steps at 0.025 ms = 0.5 ms = one BL-1 Izhikevich step."""
        n = 2
        net = JaxleyNetwork.ball_and_stick(n_neurons=n)
        state = net.init_state()
        I_ext = jnp.zeros(n)

        # Should run without error
        state, _ = net.step_multiple(state, I_ext, n_steps=20)
        assert state.voltages.shape[0] == n
