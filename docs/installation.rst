Installation
============

Prerequisites
-------------

- **Python >= 3.10** (3.11 or 3.12 recommended)
- **pip** (or `uv <https://github.com/astral-sh/uv>`_ for faster installs)

BL-1 depends on JAX, Equinox, Matplotlib, NumPy, h5py, and PyYAML.
These are installed automatically.

Basic Install
-------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/<org>/bl1.git
   cd bl1
   pip install -e .

Development Install
-------------------

Includes pytest, hypothesis, and ipykernel for running the test suite and
notebooks:

.. code-block:: bash

   pip install -e ".[dev,quality]"

Documentation Install
---------------------

Includes Sphinx, the ReadTheDocs theme, and related extensions:

.. code-block:: bash

   pip install -e ".[docs]"

Optional Dependencies
---------------------

GPU Support (CUDA)
^^^^^^^^^^^^^^^^^^

BL-1 uses JAX for all numerical computation. To run on an NVIDIA GPU,
install the CUDA-enabled JAX build:

.. code-block:: bash

   pip install jax[cuda12]

See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
for detailed instructions, including support for specific CUDA versions.

Jaxley Morphological Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Jaxley <https://github.com/jaxleyverse/jaxley>`_ provides multi-compartment
morphological neuron models that can be integrated with BL-1:

.. code-block:: bash

   pip install -e ".[jaxley]"

VizDoom Game Environment
^^^^^^^^^^^^^^^^^^^^^^^^

For 3D game environments beyond the built-in Pong:

.. code-block:: bash

   pip install -e ".[vizdoom]"

Verifying the Installation
--------------------------

.. code-block:: bash

   python -c "import bl1; print('OK')"

You can also check which JAX backend is active:

.. code-block:: bash

   python -c "import jax; print(jax.default_backend())"

This prints ``cpu``, ``gpu``, or ``tpu`` depending on your hardware.
