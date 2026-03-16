"""Analysis tools: criticality, bursts, learning metrics."""

from bl1.analysis.criticality import branching_ratio, avalanche_size_distribution
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.analysis.metrics import rally_length, performance_comparison

__all__ = [
    "branching_ratio",
    "avalanche_size_distribution",
    "detect_bursts",
    "burst_statistics",
    "rally_length",
    "performance_comparison",
]
