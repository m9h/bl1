BL-1: In-Silico Cortical Culture Simulator
==========================================

**BL-1** is a JAX-based framework for simulating dissociated cortical
cultures on multi-electrode arrays (MEAs), inspired by the DishBrain system
(Kagan et al. 2022).  It provides everything you need to build, run, and
analyse virtual cortical cultures -- from single-neuron dynamics to
closed-loop game-playing experiments.

The simulator models populations of Izhikevich spiking neurons with five
cortical cell types (RS, IB, CH, FS, LTS) and an 80/20
excitatory-inhibitory split.  Neurons are placed on a virtual substrate and
connected through distance-dependent synapses with four biophysically
distinct receptor types: AMPA and GABA_A (fast, single-exponential) plus
NMDA and GABA_B (slow, dual-exponential with voltage-dependent Mg2+ block
for NMDA).  Connectivity uses BCOO sparse matrices and scales from small
demonstration networks to 100K+ neuron simulations using event-driven or
segment-sum sparse acceleration paths.

A suite of synaptic plasticity mechanisms lets networks learn and adapt
over time.  STDP updates excitatory weights based on spike timing;
homeostatic scaling regulates overall activity levels; short-term
plasticity (Tsodyks-Markram) produces synaptic depression and facilitation;
and structural plasticity allows rewiring.  These mechanisms work together
to produce the spontaneous bursting, criticality, and developmental
dynamics observed in real cortical cultures.

BL-1 includes a virtual 64-channel MEA for spike detection and LFP
approximation, a Pong game environment for closed-loop experiments, and a
controller that orchestrates sensory encoding, motor decoding, and
feedback stimulation following the DishBrain free-energy-principle
protocol.  Analysis tools cover burst detection, criticality metrics
(branching ratio, avalanche distributions), rally-length statistics, and
pharmacological modelling.  The entire simulation loop compiles into a
single XLA program via ``jax.lax.scan``, with no Python-level per-timestep
overhead.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index
