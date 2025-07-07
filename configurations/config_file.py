"""
Configuration settings for sensor placement optimization.
"""

from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration parameters for sensor optimization."""
    #test parameters
    test_num_runs: int = 30
    #environment parameters
    environment_base_station: tuple[float, float] = (150.0, 50.0)
    environment_field_length: float = 100.0
    environment_field_width: float = 100.0
    environment_bed_width: float = 0.8
    environment_bed_length: float = 5.0
    environment_furrow_width: float = 1.5
    environment_grid_size: float = 1.0
    environment_dot_spacing: float = 0.4
    #deployment parameters
    deployment_sensor_range: float = 5.0
    deployment_max_sensors: int = 500
    deployment_comm_range: float = 15.0
    deployment_grid_space: float = 2.0
    deployment_pop_size: int = 50
    deployment_archive_size: int = 50
    deployment_generations: int = 10
    deployment_random_seed: int = 42
    #clustering parameters
    clustering_max_cluster_size: int = 15
    clustering_max_iterations: int = 50
    clustering_tolerance: float = 1e-4
    clustering_max_num_cluster: int = 30
    clustering_min_num_cluster: int = 2
    clustering_use_mobile_rover_calc: bool = False
    clustering_max_restart_attempts: int = 10
    clustering_max_iteration_assignments: int = 10
    #data collection parameters
    collection_alpha: float = 1.0
    collection_beta: float = 2.0
    collection_Q: float = 100.0
    collection_rho: float = 0.1
    collection_gamma: float = 0.1
    collection_n_ants: int = 10
    collection_n_iterations: int = 100
    collection_sample_distance: float = 0.5 #checks the accuracy of the path
    collection_buffer_distance: float = 1.0

@dataclass
class SolutionMetrics:
    """Container for solution evaluation metrics."""
    n_sensors: int
    coverage_rate: float
    over_coverage_rate: float
    connectivity_rate: float
