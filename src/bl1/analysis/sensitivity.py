"""Gradient-based parameter sensitivity analysis.

Leverages JAX's automatic differentiation to compute how simulation
outputs change with respect to model parameters. This is a capability
that biological cultures lack --- enabling rapid parameter exploration,
model fitting, and mechanistic insight.

Key functions:
- parameter_sensitivity: Compute gradients of a scalar metric w.r.t. parameters
- sweep_parameter: Run parameter sweeps efficiently with vmap
- fit_parameters: Gradient-descent optimization of parameters to match target data
"""

from __future__ import annotations

from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step, izhikevich_step_surrogate
from bl1.core.surrogate import superspike_threshold
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    compute_synaptic_current,
)


def parameter_sensitivity(
    metric_fn: Callable,
    params: Any,
    state: NeuronState,
    I_external: Array,
    dt: float = 0.5,
    n_steps: int = 1000,
    surrogate_fn=None,
    beta: float = 10.0,
) -> Any:
    """Compute gradients of a scalar metric with respect to neuron parameters.

    Uses jax.grad to differentiate a simulation metric (e.g., mean firing rate,
    synchrony index) with respect to model parameters.

    By default, uses the SuperSpike surrogate gradient so that spike-count-based
    metrics (like mean firing rate) produce non-zero gradients.  Previously,
    the hard threshold ``v >= V_PEAK`` had zero gradient almost everywhere,
    making ``jax.grad`` of spike-count metrics useless.

    Args:
        metric_fn: A function (spike_history) -> scalar that computes the metric
            to differentiate. Must be JAX-compatible (no Python control flow on
            traced values).
        params: IzhikevichParams or any pytree of parameters to differentiate.
        state: Initial NeuronState.
        I_external: (n_steps, N) external current array.
        dt: Timestep in ms.
        n_steps: Number of simulation steps.
        surrogate_fn: Surrogate gradient function from ``bl1.core.surrogate``.
            Defaults to ``superspike_threshold``.  Pass ``None`` explicitly and
            set ``beta`` to 0 to disable (not recommended).
        beta: Sharpness parameter for the surrogate gradient.

    Returns:
        Gradient pytree with same structure as params, where each leaf
        contains d(metric)/d(param).

    Example:
        # How does mean firing rate change with parameter 'a'?
        def mean_rate(spikes):
            return jnp.mean(spikes.astype(jnp.float32)) * 1000 / dt

        grads = parameter_sensitivity(mean_rate, params, state, I_ext)
        # grads.a is shape (N,) --- sensitivity of rate to each neuron's 'a'
    """
    if surrogate_fn is None:
        surrogate_fn = superspike_threshold

    # Ensure initial state has float32 spikes to match surrogate step output
    init_state = NeuronState(
        v=state.v,
        u=state.u,
        spikes=state.spikes.astype(jnp.float32),
    )

    def simulate_and_measure(params):
        """Inner function that runs simulation and computes metric."""

        def scan_fn(carry, I_t):
            s = carry
            s = izhikevich_step_surrogate(
                s, params, I_t, dt,
                surrogate_fn=surrogate_fn, beta=beta,
            )
            return s, s.spikes

        _, spike_history = jax.lax.scan(scan_fn, init_state, I_external)
        return metric_fn(spike_history)

    return jax.grad(simulate_and_measure)(params)


def sweep_parameter(
    simulate_fn: Callable,
    param_values: Array,
    base_params: Any,
    param_name: str,
    param_index: int | None = None,
) -> Array:
    """Efficiently sweep a single parameter using jax.vmap.

    Runs the same simulation with different values of one parameter,
    vectorized across the parameter axis.

    Args:
        simulate_fn: Function (params) -> scalar_metric
        param_values: 1D array of parameter values to sweep
        base_params: Base parameter set (pytree)
        param_name: Name of the parameter field to sweep (e.g., "a", "b")
        param_index: If not None, index into the parameter array (for per-neuron
            params, this sets all neurons to the swept value)

    Returns:
        Array of metric values, one per parameter value.

    Example:
        # Sweep STDP A_plus from 0.001 to 0.01
        values = jnp.linspace(0.001, 0.01, 20)
        metrics = sweep_parameter(sim_fn, values, base_params, "A_plus")
    """

    def run_with_param(val):
        # Create modified params with the swept value
        field = getattr(base_params, param_name)
        if param_index is not None:
            new_field = field.at[param_index].set(val)
        elif hasattr(field, "shape") and field.ndim > 0:
            new_field = jnp.full_like(field, val)
        else:
            new_field = val
        modified = base_params._replace(**{param_name: new_field})
        return simulate_fn(modified)

    return jax.vmap(run_with_param)(param_values)


def fit_parameters(
    target_metric: float,
    metric_fn: Callable,
    params: Any,
    state: NeuronState,
    I_external: Array,
    param_names: list[str],
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    dt: float = 0.5,
    surrogate_fn=None,
    beta: float = 10.0,
) -> tuple[Any, Array]:
    """Fit parameters to match a target metric using gradient descent.

    Minimizes (metric_fn(simulate(params)) - target_metric)^2 with respect
    to the specified parameters.

    Uses surrogate gradients by default so that spike-count-based metrics
    produce useful gradients.

    Args:
        target_metric: Target value for the metric
        metric_fn: Function (spike_history) -> scalar
        params: Initial parameter values
        state: Initial neuron state
        I_external: (T, N) external current
        param_names: List of parameter names to optimize
        learning_rate: Gradient descent step size
        n_iterations: Number of optimization steps
        dt: Simulation timestep
        surrogate_fn: Surrogate gradient function (default: superspike_threshold)
        beta: Sharpness parameter for surrogate gradient

    Returns:
        (optimized_params, loss_history) where loss_history is (n_iterations,)
    """
    if surrogate_fn is None:
        surrogate_fn = superspike_threshold

    # Ensure initial state has float32 spikes to match surrogate step output
    init_state = NeuronState(
        v=state.v,
        u=state.u,
        spikes=state.spikes.astype(jnp.float32),
    )

    def loss_fn(params):
        def scan_fn(carry, I_t):
            s = carry
            s = izhikevich_step_surrogate(
                s, params, I_t, dt,
                surrogate_fn=surrogate_fn, beta=beta,
            )
            return s, s.spikes

        _, spike_history = jax.lax.scan(scan_fn, init_state, I_external)
        metric = metric_fn(spike_history)
        return (metric - target_metric) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_fn_jit = jax.jit(loss_fn)

    loss_history = []
    current_params = params

    for _i in range(n_iterations):
        loss = loss_fn_jit(current_params)
        loss_history.append(float(loss))

        grads = grad_fn(current_params)

        # Update only specified parameters
        updates = {}
        for name in param_names:
            grad_val = getattr(grads, name)
            current_val = getattr(current_params, name)
            updates[name] = current_val - learning_rate * grad_val

        current_params = current_params._replace(**updates)

    return current_params, jnp.array(loss_history)


# ---------------------------------------------------------------------------
# Common metric functions (JAX-compatible)
# ---------------------------------------------------------------------------


def mean_firing_rate(spike_history: Array, dt_ms: float = 0.5) -> Array:
    """Mean population firing rate in Hz. Differentiable."""
    return jnp.mean(spike_history) * (1000.0 / dt_ms)


def synchrony_index(spike_history: Array) -> Array:
    """Population synchrony index (variance of population rate / mean variance).

    Higher values indicate more synchronous firing. Differentiable.
    """
    pop_rate = spike_history.sum(axis=1)  # (T,)
    var_pop = jnp.var(pop_rate)
    # Mean of individual neuron variances
    var_individual = jnp.mean(jnp.var(spike_history, axis=0))
    N = spike_history.shape[1]
    return var_pop / (N * jnp.maximum(var_individual, 1e-8))


def temporal_sparseness(spike_history: Array) -> Array:
    """Temporal sparseness of activity. Differentiable.

    1.0 = maximally sparse (single spike), 0.0 = uniform firing.
    """
    rates = spike_history.mean(axis=0)  # (N,) mean rate per neuron
    N = rates.shape[0]
    mean_rate = jnp.mean(rates)
    mean_sq_rate = jnp.mean(rates**2)
    # Treves-Rolls sparseness
    return (1.0 - (mean_rate**2) / jnp.maximum(mean_sq_rate, 1e-10)) / (
        1.0 - 1.0 / N
    )
