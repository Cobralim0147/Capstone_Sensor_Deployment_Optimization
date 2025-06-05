"""
Sensor Placement Optimization using SPEA2 Algorithm

This module provides functionality for optimizing sensor placement in agricultural fields
using multi-objective optimization to balance coverage, connectivity, and cost.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import logging

# Custom module imports
try:
    from generate_field import environment_generator, plot_field  # Updated import
    from visualization import PlotUtils    # Updated import
    from deploment import VegSensorProblem, SPEA2, SensorMetrics
    from initial_clustering import SensorNetworkClustering
    from config import OptimizationConfig, SolutionMetrics
except ImportError as e:
    logging.warning(f"Custom module import failed: {e}")
    print(f"Warning: {e}")
    print("Please ensure all custom modules are available in the Python path")


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
            
            # # Optionally plot the field
            # plot_field(
            #     field_map=field_map,
            #     vegetable_pos=vegetable_pos,
            #     grid_size=environment_grid_size,
            #     field_length=environment_field_length,
            #     field_width=environment_field_width,
            #     monitor_location=monitor_location
            # )

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
        """Run our custom SPEA2 optimization algorithm."""
        if self.problem is None:
            raise ValueError("Problem not set up. Call setup_optimization_problem() first.")
        
        try:
            print("Starting SPEA2 optimization...")

            algorithm = SPEA2(
                problem=self.problem,
                pop_size=self.config.deployment_pop_size,
                archive_size=self.config.deployment_archive_size,
                max_generations=self.config.deployment_generations
            )
            
            final_archive, history = algorithm.run()
            
            class Result:
                def __init__(self, archive):
                    self.X = np.array([ind.chromosome for ind in archive])
                    self.F = np.array([ind.objectives for ind in archive])
                    self.G = np.zeros((len(archive), 1))
                    self.history = history
                    
            result = Result(final_archive)
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
            
            self._log_objective_ranges(F)
            
            solutions = []
            
            best_coverage_idx = np.argmin(F[:, 1])
            solutions.append(("Best Coverage", best_coverage_idx, X[best_coverage_idx]))
            
            efficiency_score = -F[:, 1] / (F[:, 0] + 1e-6)
            best_efficiency_idx = np.argmax(efficiency_score)
            solutions.append(("Most Efficient", best_efficiency_idx, X[best_efficiency_idx]))
            
            best_connectivity_idx = np.argmin(F[:, 4])
            solutions.append(("Best Connectivity", best_connectivity_idx, X[best_connectivity_idx]))
            
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
                print(f"  Sensors: {metrics.n_sensors}")
                print(f"  Coverage: {metrics.coverage_rate:.1f}%")
                print(f"  Over-coverage: {metrics.over_coverage_rate:.1f}%")
                print(f"  Connectivity: {metrics.connectivity_rate:.1f}%")
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
            
            plt.figure(figsize=(12, 10))
            
            PlotUtils.plot_bed_boundaries(self.field_data['bed_coords'])
            PlotUtils.plot_vegetables(self.field_data['vegetable_pos'])
            PlotUtils.plot_sensors(sensors, metrics, self.problem.sensor_range)
            
            plt.xlim(0, self.field_data['environment_field_length'])
            plt.ylim(0, self.field_data['environment_field_width'])
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'{title}\n'
                     f'Coverage: {metrics.coverage_rate:.1f}%, '
                     f'Sensors: {metrics.n_sensors}, '
                     f'Connectivity: {metrics.connectivity_rate:.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Failed to visualize solution: {e}")
            raise


def main() -> Tuple[Any, Any]:
    """Main function to run the sensor placement optimization."""
    config = OptimizationConfig()
    optimizer = SensorOptimizer(config)
    
    try:
        # Generate field environment and setup
        field_data = optimizer.generate_field_environment()
        optimizer.setup_optimization_problem()
        optimizer.clustering_system = SensorNetworkClustering(
            comm_range=config.deployment_comm_range,
            max_cluster_size=config.clustering_max_cluster_size,
            max_iterations=config.clustering_max_iterations,
            tolerance=config.clustering_tolerance
        )
        
        # Run optimization
        result = optimizer.run_optimization()

        if result.X is not None and len(result.X) > 0:
            solutions = optimizer.analyze_pareto_front(result)
            
            if solutions:
                # Use best coverage solution for clustering
                name, idx, chromosome = solutions[0]
                deployed_sensors = optimizer.problem.get_sensor_positions(chromosome)
                metrics = optimizer.problem.evaluate_solution(chromosome)
                
                # Create visualization
                plt.figure(figsize=(15, 10))
                
                # Plot field layout
                PlotUtils.plot_bed_boundaries(optimizer.field_data['bed_coords'])
                PlotUtils.plot_vegetables(optimizer.field_data['vegetable_pos'])
                
                # Perform clustering
                optimal_k = 5
                assignments, cluster_heads, info = optimizer.clustering_system.perform_clustering(
                    deployed_sensors, optimal_k, random_state=config.deployment_random_seed
                )
                
                # Convert cluster_heads to sensor positions
                cluster_head_positions = deployed_sensors[cluster_heads]
                
                # Plot clustered sensors
                PlotUtils.plot_clustered_sensors(deployed_sensors, assignments, 
                                               cluster_head_positions, optimizer.problem.sensor_range)
                
                # Set plot properties
                plt.xlim(0, optimizer.field_data['environment_field_length'])
                plt.ylim(0, optimizer.field_data['environment_field_width'])
                plt.title(f'Field Layout with Clustered Sensors\n' 
                         f'Coverage: {metrics.coverage_rate:.1f}%, '
                         f'Sensors: {metrics.n_sensors}, '
                         f'Clusters: {len(np.unique(assignments))}')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.tight_layout()
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