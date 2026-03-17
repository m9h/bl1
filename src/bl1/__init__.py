"""BL-1: In-silico cortical culture simulator.

A JAX-based framework for simulating dissociated cortical cultures on
multi-electrode arrays (MEAs), inspired by the DishBrain system (Kagan
et al. 2022).

Key components
--------------
- **core** -- Spiking neuron models (Izhikevich, AdEx), conductance-based
  synapses (AMPA, NMDA, GABA_A, GABA_B), and a ``jax.lax.scan``-based
  time-stepper.
- **network** -- Spatial neuron placement, distance-dependent connectivity,
  and developmental growth models.
- **plasticity** -- STDP, homeostatic scaling, short-term plasticity
  (Tsodyks-Markram), and structural plasticity.
- **mea** -- Virtual multi-electrode array with spike detection, LFP
  approximation, and electrical stimulation.
- **loop** -- Closed-loop controller connecting the culture to a game
  environment via sensory encoding and motor decoding.
- **games** -- Game environments (Pong) for closed-loop experiments.
- **analysis** -- Criticality metrics, burst detection, performance
  analysis, and pharmacological modelling.

Quick start
-----------
::

    import jax
    from bl1 import Culture, MEA, ClosedLoop, Pong

    key = jax.random.PRNGKey(42)
    net_params, state, izh_params = Culture.create(key, n_neurons=1000)
    mea = MEA("cl1_64ch")
    game = Pong()
"""

from bl1.games.pong import Pong
from bl1.loop.controller import ClosedLoop
from bl1.mea.electrode import MEA
from bl1.network.types import Culture, CultureState, NetworkParams

__all__ = ["Culture", "CultureState", "NetworkParams", "MEA", "ClosedLoop", "Pong"]
