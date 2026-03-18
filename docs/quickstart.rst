Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install -e ".[dev]"

Basic Usage
-----------

Create a network and run a simulation:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from bl1.core.izhikevich import IzhikevichParams, izhikevich_step
   from bl1.core.synapses import fast_synapse_step
   from bl1.core.integrator import simulate
   from bl1.network.topology import create_connectivity

   # Create a small network
   key = jax.random.PRNGKey(0)
   n_neurons = 100
   conn = create_connectivity(key, n_neurons, p_connect=0.1)

   # Run simulation using the integrator
   # See bl1.core.integrator for full details

For more detailed examples, see the notebooks in the ``notebooks/`` directory.
