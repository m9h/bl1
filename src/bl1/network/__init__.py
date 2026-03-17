"""Network topology, connectivity, and developmental growth.

Provides spatial neuron placement on a 2-D substrate, distance-dependent
connectivity (with automatic dispatch between dense and spatial-hashing
algorithms), and a NETMORPH-inspired growth model that simulates network
maturation across days in vitro (DIV).
"""

from bl1.network.types import CultureState, Culture, NetworkParams
from bl1.network.topology import build_connectivity, place_neurons
from bl1.network.growth import (
    GrowthParams,
    GrowthState,
    init_growth,
    grow_to_div,
    mature_network,
)

__all__ = [
    "Culture",
    "CultureState",
    "GrowthParams",
    "GrowthState",
    "NetworkParams",
    "build_connectivity",
    "grow_to_div",
    "init_growth",
    "mature_network",
    "place_neurons",
]
