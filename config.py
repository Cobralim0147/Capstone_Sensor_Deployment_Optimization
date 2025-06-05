"""
Configuration settings for sensor placement optimization.
"""

from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration parameters for sensor optimization."""
    environment_field_length: float = 100.0
    environment_field_width: float = 100.0
    environment_bed_width: float = 0.8
    environment_bed_length: float = 5.0
    environment_furrow_width: float = 1.5
    environment_grid_size: float = 1.0
    environment_dot_spacing: float = 0.4
    deployment_sensor_range: float = 5.0
    deployment_max_sensors: int = 500
    deployment_comm_range: float = 15.0
    deployment_grid_space: float = 2.0
    deployment_pop_size: int = 50
    deployment_archive_size: int = 50
    deployment_generations: int = 150
    deployment_random_seed: int = 42
    clustering_max_cluster_size: int = 10
    clustering_max_iterations: int = 50
    clustering_tolerance: float = 1e-4

@dataclass
class SolutionMetrics:
    """Container for solution evaluation metrics."""
    n_sensors: int
    coverage_rate: float
    over_coverage_rate: float
    connectivity_rate: float
