from dataclasses import dataclass
from typing import *

import numpy as np


@dataclass
class SensorNode:
    """Represents a sensor node with its properties."""
    id: int
    x: float
    y: float
    energy_level: float  # Remaining energy (0-1)
    cluster_id: int = -1
    is_cluster_head: bool = False

@dataclass
class ClusterMetrics:
    """Container for cluster evaluation metrics."""
    cluster_id: int
    center: Tuple[float, float]
    nodes: List[SensorNode]
    density: int
    selected_head: SensorNode
    fuzzy_score: float

@dataclass
class SensorMetrics:
    """Container for sensor placement metrics."""
    n_sensors: int
    sensor_positions: np.ndarray
    count_rate: float
    coverage_rate: float
    over_coverage_rate: float
    placement_rate: float
    connectivity_rate: float

class Individual:
    """Represents an individual in the population for optimization."""
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome
        self.objectives = None
        self.strength = 0
        self.raw_fitness = 0
        self.density = 0
        self.fitness = 0