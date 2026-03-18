Quickstart
==========

This guide walks through the core BL-1 workflow: creating a network,
running a simulation, visualizing the results, recording from a virtual
MEA, and running a closed-loop Pong experiment.  Every import path and
function call below matches the actual source code.

Creating a Network
------------------

A BL-1 simulation starts by placing neurons on a substrate, assigning
Izhikevich parameters, and building distance-dependent connectivity.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from bl1.core.izhikevich import create_population
   from bl1.network.topology import place_neurons, build_connectivity

   key = jax.random.PRNGKey(42)
   k1, k2, k3 = jax.random.split(key, 3)

   N_NEURONS = 200

   # Place neurons uniformly on a 3000 x 3000 um substrate
   positions = place_neurons(k1, N_NEURONS, substrate_um=(3000.0, 3000.0))

   # Create the neuron population: Izhikevich params, initial state, E/I mask
   izh_params, neuron_state, is_excitatory = create_population(k2, N_NEURONS)

   # Build distance-dependent connectivity (returns BCOO sparse matrices)
   W_exc, W_inh, delays = build_connectivity(
       k3, positions, is_excitatory,
       lambda_um=200.0,   # length constant for connection probability
       p_max=0.21,        # max connection probability at zero distance
       g_exc=0.05,        # excitatory synaptic weight
       g_inh=0.20,        # inhibitory synaptic weight
   )

``create_population`` returns three values:

- ``izh_params`` -- an ``IzhikevichParams`` NamedTuple with per-neuron
  ``(a, b, c, d)`` arrays for five cell types (RS, IB, CH, FS, LTS).
- ``neuron_state`` -- a ``NeuronState`` NamedTuple with membrane potential
  ``v``, recovery variable ``u``, and spike indicators.
- ``is_excitatory`` -- a boolean array marking excitatory neurons (80% by
  default).

``build_connectivity`` returns sparse BCOO matrices for excitatory weights,
inhibitory weights, and axonal conduction delays.

Running a Simulation
--------------------

The ``simulate`` function uses ``jax.lax.scan`` to compile the entire
time loop into a single XLA program.  It jointly integrates neurons and
four synapse types (AMPA, NMDA, GABA_A, GABA_B) with optional plasticity.

.. code-block:: python

   from bl1.core.synapses import create_synapse_state
   from bl1.core.integrator import simulate

   DT = 0.5  # ms per timestep
   SIM_DURATION_MS = 1000.0
   T_STEPS = int(SIM_DURATION_MS / DT)  # 2000 steps

   # Initialize synapse conductances at zero
   syn_state = create_synapse_state(N_NEURONS)

   # External drive: tonic current to push neurons above rheobase
   I_external = jnp.full((T_STEPS, N_NEURONS), 7.0)

   # Run the simulation
   result = simulate(
       params=izh_params,
       init_state=neuron_state,
       syn_state=syn_state,
       stdp_state=None,
       W_exc=W_exc,
       W_inh=W_inh,
       I_external=I_external,
       dt=DT,
       plasticity_fn=None,
   )

   spike_history = result.spike_history  # (T, N) boolean array
   total_spikes = int(jnp.sum(spike_history))
   mean_rate_hz = total_spikes / N_NEURONS / (SIM_DURATION_MS / 1000.0)
   print(f"Total spikes: {total_spikes}, Mean rate: {mean_rate_hz:.1f} Hz")

``simulate`` returns a ``SimulationResult`` NamedTuple with fields:

- ``spike_history`` -- ``(T, N)`` boolean spike raster
- ``final_neuron_state`` -- ``NeuronState`` at the end of the simulation
- ``final_syn_state`` -- ``SynapseState`` at the end
- ``final_stdp_state`` -- plasticity state (or ``None``)
- ``final_W_exc`` -- excitatory weight matrix (potentially updated by
  plasticity)

Visualization
-------------

BL-1 includes raster plot functions that follow the conventions of
cortical-culture MEA literature.

.. code-block:: python

   import numpy as np
   from bl1.visualization.raster import plot_raster, plot_raster_with_rate

   n_exc = int(is_excitatory.sum())

   # Spike raster with excitatory (blue) / inhibitory (red) colouring
   fig = plot_raster(
       np.asarray(spike_history),
       dt_ms=DT,
       ei_boundary=n_exc,
       title="Spike Raster (1 s simulation)",
   )

   # Raster with population firing-rate trace below
   fig = plot_raster_with_rate(
       np.asarray(spike_history),
       dt_ms=DT,
       rate_bin_ms=10.0,
       title="Neural Activity with Population Rate",
   )

MEA Recording
-------------

BL-1 includes a virtual multi-electrode array.  The default ``cl1_64ch``
configuration is an 8x8 grid with 200 um spacing, matching the
CorticalLabs CL1 hardware.

.. code-block:: python

   from bl1.mea.electrode import MEA, build_neuron_electrode_map
   from bl1.mea.recording import compute_electrode_rates

   # Create the MEA
   mea = MEA("cl1_64ch")
   print(f"{mea.n_electrodes} electrodes, detection radius: "
         f"{mea.detection_radius_um} um")

   # Build the neuron-electrode spatial mapping: (E, N) boolean mask
   ne_map = build_neuron_electrode_map(
       positions, mea.positions, mea.detection_radius_um
   )

   # Compute per-electrode firing rates over the full simulation
   electrode_rates = compute_electrode_rates(
       spike_history, ne_map, window_ms=SIM_DURATION_MS, dt=DT
   )

   rates_np = np.asarray(electrode_rates)
   print(f"Electrode rate range: {rates_np.min():.1f} - {rates_np.max():.1f} Hz")

The rates can be reshaped to ``mea.config.grid_shape`` (8, 8) and
displayed as a heatmap for spatial activity visualization.

Closed-Loop Pong
-----------------

BL-1 can connect a cortical culture to a Pong game in a closed-loop
experiment, replicating the DishBrain protocol.  The ``ClosedLoop``
controller handles sensory encoding, motor decoding, feedback stimulation,
and STDP-based learning.

.. code-block:: python

   from bl1.games.pong import Pong
   from bl1.loop.controller import ClosedLoop
   from bl1.mea.electrode import MEA
   from bl1.network.types import NetworkParams
   from bl1.plasticity.stdp import STDPParams

   key = jax.random.PRNGKey(0)
   k1, k2, k3, k_run = jax.random.split(key, 4)
   n_neurons = 100

   # Build the culture
   positions = place_neurons(k1, n_neurons)
   izh_params, _, is_excitatory = create_population(k2, n_neurons)
   W_exc, W_inh, delays = build_connectivity(k3, positions, is_excitatory)

   net_params = NetworkParams(
       positions=positions,
       is_excitatory=is_excitatory,
       W_exc=W_exc,
       W_inh=W_inh,
       delays=delays,
   )

   mea = MEA("cl1_64ch")
   game = Pong()

   # Define sensory and motor electrode regions
   sensory_channels = list(range(8))
   motor_regions = {"up": list(range(8, 12)), "down": list(range(12, 16))}

   cl = ClosedLoop(
       network_params=net_params,
       neuron_params=izh_params,
       mea=mea,
       sensory_channels=sensory_channels,
       motor_regions=motor_regions,
       game=game,
       stdp_params=STDPParams(),
   )

   results = cl.run(
       key=k_run,
       duration_s=10.0,       # experiment duration in seconds
       dt_ms=0.5,             # neural simulation timestep
       feedback="fep",        # free-energy-principle feedback
       game_dt_ms=20.0,       # game update interval
   )

   print(f"Game events: {len(results['game_events'])}")
   print(f"Rally lengths: {results['rally_lengths']}")

The ``run`` method returns a dictionary with:

- ``spike_history`` -- full spike raster
- ``game_events`` -- list of (time, event_type) tuples
- ``rally_lengths`` -- consecutive-hit counts per rally
- ``final_neuron_state``, ``final_syn_state``, ``final_game_state``
- ``population_rates`` -- per-game-step firing rates

Three feedback modes are available: ``"fep"`` (predictable stimulation on
hits, unpredictable on misses), ``"open_loop"`` (random stimulation), and
``"silent"`` (no stimulation).

Next Steps
----------

- **Notebooks** -- See ``notebooks/00_quickstart.ipynb`` for a runnable
  version of this guide with plots and MEA heatmaps.
- **API Reference** -- Full documentation of every module is in the
  :doc:`api/index` section.
- **Plasticity** -- Add STDP to evolve synaptic weights; see
  ``bl1.plasticity.stdp``.
- **Analysis** -- Measure criticality with ``bl1.analysis.criticality``
  and detect bursts with ``bl1.analysis.bursts``.
- **Scale up** -- Use ``simulate(..., use_fast_sparse=True)`` or
  ``simulate(..., use_event_driven=True)`` for networks with 50K+ neurons.
- **3D cultures** -- Use ``place_neurons_spheroid`` or
  ``place_neurons_layered`` for organoid and cortical-layer models.
- **Configuration** -- YAML config files in ``configs/`` provide
  pre-tuned parameter sets for common experiment types.
