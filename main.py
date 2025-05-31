"""
Sensor Placement Optimization using SPEA-II Multi-Objective Algorithm

This module provides functionality for optimizing sensor placement in agricultural fields
using multi-objective optimization to balance coverage, connectivity, and cost.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Third-party imports
try:
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError as e:
    raise ImportError(f"Required package 'pymoo' not found: {e}")

# Custom module imports - wrapped in try-except for better error handling
try:
    from generate_field import environment_generator, plot_field
    from deploment import VegSensorProblem
    from initial_clustering import SensorNetworkClustering
except ImportError as e:
    logging.warning(f"Custom module import failed: {e}")
    print(f"Warning: {e}")
    print("Please ensure all custom modules are available in the Python path")


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
    deployment_sensor_range: float = 3.0
    deployment_max_sensors: int = 500
    deployment_comm_range: float = 15.0
    deployment_grid_space: float = 2.0
    deployment_pop_size: int = 50
    deployment_archive_size: int = 30
    deployment_generations: int = 50
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


class SensorOptimizer:
    """Main class for sensor placement optimization."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.field_data = None
        self.problem = None
        self.clustering_system = None
        
    
    def generate_field_environment(self) -> Dict[str, Any]:
        """Generate field environment with error handling."""
        try:
            print("Generating field environment...")
            
            results = environment_generator(
                field_length=self.config.environment_field_length,
                field_width=self.config.environment_field_width,
                bed_width=self.config.environment_bed_width,
                bed_length=self.config.environment_bed_length,
                furrow_width=self.config.environment_furrow_width,
                grid_size=self.config.environment_grid_size,
                dot_spacing=self.config.environment_dot_spacing
            )
            
            # Unpack results with validation
            if len(results) != 7:
                raise ValueError(f"Expected 7 values from environment_generator, got {len(results)}")
            
            field_map, environment_grid_size, environment_field_length, environment_field_width, monitor_location, vegetable_pos, bed_coords = results
            
            self.field_data = {
                'field_map': field_map,
                'environment_grid_size': environment_grid_size,
                'environment_field_length': environment_field_length,
                'environment_field_width': environment_field_width,
                'monitor_location': monitor_location,
                'vegetable_pos': vegetable_pos,
                'bed_coords': bed_coords
            }
            
            print(f"Field generated: {environment_field_length}m x {environment_field_width}m, "
                           f"{len(bed_coords)} beds, {len(vegetable_pos)} vegetables")
            
            return self.field_data
            
        except Exception as e:
            print(f"Failed to generate field environment: {e}")
            raise
    
    def setup_optimization_problem(self) -> None:
        """Set up the optimization problem."""
        if self.field_data is None:
            raise ValueError("Field data not generated. Call generate_field_environment() first.")
        
        try:
            print("Setting up optimization problem...")
            
            vegetable_pos = self.field_data['vegetable_pos']
            bed_coords = self.field_data['bed_coords']
            
            # Extract vegetable coordinates safely
            if vegetable_pos:
                veg_x = np.array([pos[0] for pos in vegetable_pos])
                veg_y = np.array([pos[1] for pos in vegetable_pos])
            else:
                print("No vegetable positions found, using empty arrays")
                veg_x = np.array([])
                veg_y = np.array([])
            
            self.problem = VegSensorProblem(
                veg_x=veg_x,
                veg_y=veg_y,
                bed_coords=bed_coords,
                sensor_range=self.config.deployment_sensor_range,
                max_sensors=self.config.deployment_max_sensors,
                comm_range=self.config.deployment_comm_range,
                grid_spacing=self.config.deployment_grid_space
            )
            
            print(f"Problem setup complete: "
                           f"{len(self.problem.possible_positions)} possible positions, "
                           f"{self.problem.n_var} variables, {self.problem.n_obj} objectives")
            
        except Exception as e:
            print(f"Failed to setup optimization problem: {e}")
            raise
    
    def run_optimization(self) -> Any:
        """Run the SPEA-II optimization algorithm."""
        if self.problem is None:
            raise ValueError("Problem not set up. Call setup_optimization_problem() first.")
        
        try:
            print("Starting SPEA-II optimization...")
            
            algorithm = SPEA2(
                pop_size=self.config.deployment_pop_size,
                archive_size=self.config.deployment_archive_size
            )
            
            termination = get_termination("n_gen", self.config.deployment_generations)
            
            result = minimize(
                self.problem,
                algorithm,
                termination,
                save_history=True,
                verbose=True,
                seed=self.config.deployment_random_seed
            )
            
            print("Optimization completed successfully")
            return result
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise
    
    def analyze_pareto_front(self, result: Any, top_n: int = 5) -> List[Tuple[str, int, np.ndarray]]:
        """Analyze the Pareto front and return top solutions."""
        if result.X is None or len(result.X) == 0:
            print("No solutions found in optimization result")
            return []
        
        try:
            F = result.F
            X = result.X
            
            print(f"Analyzing Pareto front with {len(X)} solutions")
            
            # Log objective ranges
            self._log_objective_ranges(F)
            
            solutions = []
            
            # Best coverage solution
            best_coverage_idx = np.argmin(F[:, 1])
            solutions.append(("Best Coverage", best_coverage_idx, X[best_coverage_idx]))
            
            # Most efficient solution (good coverage with fewer sensors)
            efficiency_score = -F[:, 1] / (F[:, 0] + 1e-6)
            best_efficiency_idx = np.argmax(efficiency_score)
            solutions.append(("Most Efficient", best_efficiency_idx, X[best_efficiency_idx]))
            
            # Best connectivity solution
            best_connectivity_idx = np.argmin(F[:, 4])
            solutions.append(("Best Connectivity", best_connectivity_idx, X[best_connectivity_idx]))
            
            # Log solution details
            self._log_solution_details(solutions)
            
            return solutions
            
        except Exception as e:
            print(f"Failed to analyze Pareto front: {e}")
            raise
    
    def _log_objective_ranges(self, F: np.ndarray) -> None:
        """Log the ranges of objective values."""
        print("Objective ranges:")
        print(f"  Count rate: [{F[:, 0].min():.1f}, {F[:, 0].max():.1f}]")
        print(f"  Coverage rate: [{-F[:, 1].max():.1f}, {-F[:, 1].min():.1f}]")
        print(f"  Over-coverage: [{F[:, 2].min():.1f}, {F[:, 2].max():.1f}]")
        print(f"  Connectivity: [{-F[:, 4].max():.1f}, {-F[:, 4].min():.1f}]")
    
    def _log_solution_details(self, solutions: List[Tuple[str, int, np.ndarray]]) -> None:
        """Log details of the selected solutions."""
        for name, idx, chromosome in solutions:
            try:
                metrics = self.problem.evaluate_solution(chromosome)
                print(f"{name} Solution:")
                print(f"  Sensors: {metrics['n_sensors']}")
                print(f"  Coverage: {metrics['coverage_rate']:.1f}%")
                print(f"  Over-coverage: {metrics['over_coverage_rate']:.1f}%")
                print(f"  Connectivity: {metrics['connectivity_rate']:.1f}%")
            except Exception as e:
                print(f"Failed to evaluate {name} solution: {e}")
    
    def visualize_solution(self, chromosome: np.ndarray, title: str = "Sensor Placement Solution") -> None:
        """Visualize a sensor placement solution."""
        if self.field_data is None or self.problem is None:
            print("Field data or problem not initialized")
            return
        
        try:
            sensors = self.problem.get_sensor_positions(chromosome)
            metrics = self.problem.evaluate_solution(chromosome)
            
            vegetable_pos = self.field_data['vegetable_pos']
            bed_coords = self.field_data['bed_coords']
            environment_field_length = self.field_data['environment_field_length']
            environment_field_width = self.field_data['environment_field_width']
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Plot bed boundaries
            self._plot_bed_boundaries(bed_coords)
            
            # Plot vegetables
            self._plot_vegetables(vegetable_pos)
            
            # Plot sensors and their coverage
            self._plot_sensors(sensors, metrics)
            
            # Set plot properties
            plt.xlim(0, environment_field_length)
            plt.ylim(0, environment_field_width)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'{title}\n'
                     f'Coverage: {metrics["coverage_rate"]:.1f}%, '
                     f'Sensors: {metrics["n_sensors"]}, '
                     f'Connectivity: {metrics["connectivity_rate"]:.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Failed to visualize solution: {e}")
    
    def _plot_bed_boundaries(self, bed_coords: List[Tuple]) -> None:
        """Plot bed boundaries."""
        for bed in bed_coords:
            x_min, y_min, x_max, y_max = bed
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 
                    'b-', linewidth=2, alpha=0.7, label='Beds' if bed == bed_coords[0] else "")
    
    def _plot_vegetables(self, vegetable_pos: List[Tuple]) -> None:
        """Plot vegetable positions."""
        if vegetable_pos:
            veg_x = [pos[0] for pos in vegetable_pos]
            veg_y = [pos[1] for pos in vegetable_pos]
            plt.plot(veg_x, veg_y, 'g.', markersize=6, alpha=0.7, label='Vegetables')
    
    def _plot_sensors(self, sensors: np.ndarray, metrics: Dict[str, Any]) -> None:
        """Plot sensors, their coverage, and communication links."""
        if sensors.shape[0] > 0:
            plt.plot(sensors[:, 0], sensors[:, 1], 'ro', markersize=10, 
                    label=f'Sensors ({sensors.shape[0]})')
            
            # Plot sensor coverage circles
            for sensor in sensors:
                circle = plt.Circle(sensor, self.problem.deployment_sensor_range, 
                                  fill=False, color='red', alpha=0.3, linestyle='--')
                plt.gca().add_patch(circle)
            
            # Plot communication links
            self._plot_communication_links(sensors)
    
    def _plot_communication_links(self, sensors: np.ndarray) -> None:
        """Plot communication links between sensors."""
        if sensors.shape[0] > 1:
            distances = np.linalg.norm(
                sensors[:, None, :] - sensors[None, :, :], axis=2
            )
            for i in range(sensors.shape[0]):
                for j in range(i + 1, sensors.shape[0]):
                    if distances[i, j] <= self.problem.comm_range:
                        plt.plot([sensors[i, 0], sensors[j, 0]], 
                                [sensors[i, 1], sensors[j, 1]], 
                                'b--', alpha=0.5, linewidth=1,

                                label='Communication' if i == 0 and j == 1 else "")
                        
    def setup_clustering_system(self) -> None:
        """Initialize the enhanced clustering system."""
        self.clustering_system = SensorNetworkClustering(
            comm_range=self.config.deployment_comm_range,
            max_cluster_size=self.config.clustering_max_cluster_size,
            max_iterations=self.config.clustering_max_iterations,
            tolerance=self.config.clustering_tolerance
        )

        print(f"Clustering system initialized with max cluster size: {self.config.clustering_max_cluster_size}")
    
    def perform_clustering(self, chromosome: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Perform enhanced clustering on deployed sensors."""
        if self.problem is None:
            raise ValueError("Problem not initialized")
        
        if self.clustering_system is None:
            self.setup_clustering_system()
        
        try:
            deployed_sensors = self.problem.get_sensor_positions(chromosome)
            
            if len(deployed_sensors) == 0:
                print("No sensors deployed for clustering")
                return np.array([]), np.array([]), {}
            
            # Calculate optimal clusters if not specified
            if n_clusters is None:
                network_area = self.config.environment_field_length * self.config.environment_field_width
                d_bs_avg = np.mean(np.linalg.norm(deployed_sensors - [self.config.environment_field_length/2, self.config.environment_field_width/2], axis=1))
                d_th = self.config.deployment_sensor_range
                
                n_clusters = self.clustering_system.calculate_optimal_clusters(
                    len(deployed_sensors), network_area, d_bs_avg, d_th
                )
                print(f"Calculated optimal clusters: {n_clusters}")
            
            # Perform enhanced clustering
            assignments, cluster_heads, info = self.clustering_system.perform_clustering(
                deployed_sensors, 
                min(n_clusters, len(deployed_sensors)),
                random_state=self.config.deployment_random_seed
            )
            
            print(f"Enhanced clustering completed: {len(np.unique(assignments))} clusters for {len(deployed_sensors)} sensors")
            print(f"Cluster sizes: {info['cluster_sizes']}")
            print(f"Total SSE: {info['total_sse']:.2f}")
            
            return assignments, cluster_heads, info
            
        except Exception as e:
            print(f"Enhanced clustering failed: {e}")
            raise
    
    def visualize_cluster(self, sensors: np.ndarray, assignments: np.ndarray, 
                              cluster_heads: np.ndarray, info: dict) -> None:
        """Visualize enhanced sensor clusters with cluster heads."""
        if self.clustering_system is None:
            self.setup_clustering_system()
            
        try:
            # Use the clustering system's visualization
            title = f"Enhanced WSN Clustering (SSE: {info.get('total_sse', 0):.1f})"
            self.clustering_system.plot_clustering_results(
                sensors, assignments, cluster_heads, title
            )
        except Exception as e:
            print(f"Failed to visualize enhanced clusters: {e}")


def main() -> Tuple[Any, Any]:
    """Main function to run the sensor placement optimization."""
    config = OptimizationConfig()
    optimizer = SensorOptimizer(config)
    
    try:
        # Generate field environment
        optimizer.generate_field_environment()
        
        # Setup optimization problem
        optimizer.setup_optimization_problem()
        
        # Run optimization
        result = optimizer.run_optimization()
        
        # Analyze results
        if result.X is not None and len(result.X) > 0:
            solutions = optimizer.analyze_pareto_front(result)
            
            # Visualize best solutions (uncomment to enable)
            # for name, idx, chromosome in solutions[:2]:  # Show top 2 solutions
            #     optimizer.visualize_solution(chromosome, title=f"{name} Solution")
            
            # Perform clustering on the most efficient solution
            if solutions:
                name, idx, chromosome = solutions[1]  # Most efficient solution
                
                # Use enhanced clustering
                assignments, cluster_heads, info = optimizer.perform_clustering(chromosome)  # n_clusters auto-calculated
                
                if len(assignments) > 0:
                    deployed_sensors = optimizer.problem.get_sensor_positions(chromosome)
                    
                    # Use enhanced visualization
                    optimizer.visualize_enhanced_clusters(deployed_sensors, assignments, cluster_heads, info)
                    
                    # Optional: Generate elbow plot
                    k_values, sse_values = optimizer.clustering_system.generate_elbow_plot(
                        deployed_sensors, k_min=2, k_max=min(8, len(deployed_sensors)), 
                        random_state=config.deployment_random_seed
                    )
                    
                    # Plot elbow curve
                    plt.figure(figsize=(8, 6))
                    plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=8)
                    plt.xlabel('Number of Clusters (K)')
                    plt.ylabel('Total SSE')
                    plt.title('Elbow Method for Sensor Network')
                    plt.grid(True, alpha=0.3)
                    plt.show()
        
        return result, optimizer.problem
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    try:
        result, problem = main()
        print("\nSensor placement optimization completed successfully!")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()