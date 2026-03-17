"""Network topology, connectivity, and developmental growth.

Provides spatial neuron placement on a 2-D substrate or in 3-D volumes
(organoids, spheroids, layered cortical structures), distance-dependent
connectivity (with automatic dispatch between dense and spatial-hashing
algorithms), and a NETMORPH-inspired growth model that simulates network
maturation across days in vitro (DIV).
"""

from bl1.network.growth import (
    GrowthParams,
    GrowthState,
    grow_to_div,
    init_growth,
    mature_network,
)
from bl1.network.topology import (
    build_connectivity,
    place_neurons,
    place_neurons_layered,
    place_neurons_spheroid,
)
from bl1.network.types import Culture, CultureState, NetworkParams

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
    "place_neurons_layered",
    "place_neurons_spheroid",
]
