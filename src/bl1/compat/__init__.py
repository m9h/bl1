"""CL-SDK compatibility layer — drop-in replacement for biological CL1 hardware.

Provides two integration paths:

- **cl_sdk**: Python API mimicking the Cortical Labs CL-SDK (``cl.open()``,
  ``neurons.loop()``, ``neurons.stim()``).
- **udp_bridge**: UDP server that speaks doom-neuron's binary protocol so
  the training server can connect to BL-1 without code changes.
"""

from bl1.compat.cl_sdk import BurstDesign, ChannelSet, Neurons, SpikeEvent, StimDesign, Tick, open
from bl1.compat.udp_bridge import VirtualCL1Server

__all__ = [
    "open", "Neurons", "ChannelSet", "StimDesign", "BurstDesign",
    "Tick", "SpikeEvent", "VirtualCL1Server",
]
