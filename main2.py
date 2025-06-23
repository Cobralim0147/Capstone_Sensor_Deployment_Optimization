"""
Sensor Placement Optimization using SPEA2 Algorithm

This module provides functionality for optimizing sensor placement in agricultural fields
using multi-objective optimization to balance coverage, connectivity, and cost.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import logging

# Custom module imports
try:
    from visualization.environment_generator import environment_generator, plot_field
    from visualization.visualization import PlotUtils
    from algorithms.spea2 import VegSensorProblem, SPEA2
    from algorithms.k_mean import KMeansClustering
    from algorithms.ant_colony import AntColonyOptimizer
    from configurations.config_file import OptimizationConfig
    from algorithms.integrated_clustering import IntegratedClusteringSystem
    from analysis.clustering_comparison import FuzzyClusteringOptimizer

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
        self.integrated_clustering = None  # Add fuzzy clustering integration
        self.fuzzy_validator = None  # Add fuzzy validator
        
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
    
    def setup_generated_field(self) -> None:
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
            raise ValueError("Problem not set up. Call setup_generated_field() first.")
        
        try:
            print("Starting SPEA2 optimization...")

            algorithm = SPEA2(problem=self.problem)            
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

    def setup_kmeans_clustering_system(self) -> None:
        """Initialize the K-means clustering system only."""
        if self.field_data is None:
            raise ValueError("Field data not generated. Call generate_field_environment() first.")
        
        try:
            # Initialize K-means clustering system
            self.clustering_system = KMeansClustering()
            
            print("K-means clustering system initialized")
            
        except Exception as e:
            print(f"Failed to setup K-means clustering system: {e}")
            raise

    def perform_kmeans_clustering_analysis(self, sensor_positions: np.ndarray,
                                     use_hybrid: bool = True,
                                     n_clusters: int = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform K-means clustering analysis with hybrid cluster calculation.
        
        Parameters:
        -----------
        sensor_positions : np.ndarray
            Array of sensor positions
        use_hybrid : bool
            If True, use both formula and elbow methods
            If False, use only formula method
        n_clusters : int
            Manual number of clusters (overrides automatic calculation)
        """
        if self.clustering_system is None:
            raise ValueError("K-means clustering system not initialized. Call setup_kmeans_clustering_system() first.")
        
        try:
            print(f"Performing K-means clustering with {len(sensor_positions)} sensors...")
            
            # Perform K-means clustering with hybrid method
            assignments, cluster_heads, clustering_info = self.clustering_system.perform_clustering(
                sensor_positions, 
                n_clusters=n_clusters,
                use_hybrid=use_hybrid,
                random_state=self.config.deployment_random_seed
            )
            
            # Print cluster calculation details
            calc_info = clustering_info.get('calculation_method', {})
            if calc_info:
                print("\n=== CLUSTER CALCULATION DETAILS ===")
                print(f"Formula method suggested: {calc_info.get('k_formula', 'N/A')} clusters")
                print(f"Elbow method suggested: {calc_info.get('k_elbow', 'N/A')} clusters")
                print(f"Final decision: {calc_info.get('k_final', 'N/A')} clusters")
                print(f"Method used: {calc_info.get('method_used', 'N/A')}")
            
            # Analyze connectivity
            connectivity_results = self.clustering_system.analyze_cluster_connectivity(
                sensor_positions, assignments, cluster_heads
            )
            
            print(f"\nK-means clustering completed:")
            print(f"  Number of clusters: {len(np.unique(assignments))}")
            print(f"  Iterations: {clustering_info['iterations']}")
            print(f"  Total SSE: {clustering_info['total_sse']:.2f}")
            print(f"  Converged: {clustering_info['converged']}")
            
            # Print connectivity analysis
            print("\n=== CONNECTIVITY ANALYSIS ===")
            for cluster_id, conn_info in connectivity_results.items():
                print(f"Cluster {cluster_id + 1}:")
                print(f"  Size: {conn_info['cluster_size']} sensors")
                print(f"  Connected to head: {conn_info['connected_to_head']}")
                print(f"  Connectivity rate: {conn_info['connectivity_rate']:.1f}%")
                if conn_info['isolated_sensors']:
                    print(f"  Isolated sensors: {len(conn_info['isolated_sensors'])}")
            
            return assignments, cluster_heads, clustering_info
            
        except Exception as e:
            print(f"Failed to perform K-means clustering analysis: {e}")
            raise

    def compare_clustering_methods(self, sensor_positions: np.ndarray) -> None:
        """Compare different clustering methods."""
        if self.clustering_system is None:
            raise ValueError("K-means clustering system not initialized.")
        
        try:
            print("\n=== CLUSTERING METHOD COMPARISON ===")
            
            # Method 1: Formula only
            print("\n--- Method 1: Formula Only ---")
            assignments1, heads1, info1 = self.perform_kmeans_clustering_analysis(
                sensor_positions, use_hybrid=False
            )
            
            # Method 2: Hybrid (formula + elbow)
            print("\n--- Method 2: Hybrid (Formula + Elbow) ---")
            assignments2, heads2, info2 = self.perform_kmeans_clustering_analysis(
                sensor_positions, use_hybrid=True
            )
            
            # Comparison summary
            print("\n=== COMPARISON SUMMARY ===")
            print(f"Formula only:     {len(heads1)} clusters, SSE: {info1['total_sse']:.2f}")
            print(f"Hybrid method:    {len(heads2)} clusters, SSE: {info2['total_sse']:.2f}")
            
            # Determine better method
            if info2['total_sse'] < info1['total_sse']:
                print("Winner: Hybrid method (lower SSE)")
                return assignments2, heads2, info2
            else:
                print("Winner: Formula only (lower SSE)")
                return assignments1, heads1, info1
                
        except Exception as e:
            print(f"Failed to compare clustering methods: {e}")
            raise

    def visualize_kmeans_clustering(self, sensor_positions: np.ndarray, 
                                assignments: np.ndarray, 
                                cluster_heads: np.ndarray,
                                metrics: Any) -> None:
        """Visualize K-means clustering results with field environment."""
        if self.clustering_system is None:
            print("K-means clustering system not initialized")
            return
        
        try:
            print("Generating K-means clustering visualization...")
            
            # Use the enhanced plotting function with field data
            self.clustering_system.plot_clustering_results(
                sensor_positions, assignments, cluster_heads,
                title=f"Traditional K-means Clustering Results",
                field_data=self.field_data  # Pass field environment data
            )
            
        except Exception as e:
            print(f"Failed to visualize K-means clustering: {e}")
            raise

    def generate_clustering_report(self, results: Dict, comparison: Dict) -> str:
        """Generate comprehensive clustering analysis report."""
        if self.integrated_clustering is None:
            raise ValueError("Integrated clustering system not initialized")
        
        try:
            report = self.integrated_clustering.generate_report(results, comparison)
            return report
            
        except Exception as e:
            print(f"Failed to generate clustering report: {e}")
            raise


def main() -> Tuple[Any, Any]:
    """Main function to run the sensor placement optimization."""
    config = OptimizationConfig()
    optimizer = SensorOptimizer(config)
    
    try:
        # Generate field environment and setup
        print("=== FIELD ENVIRONMENT GENERATION ===")
        field_data = optimizer.generate_field_environment()
        optimizer.setup_generated_field()
        
        # Setup K-means clustering system only
        print("\n=== K-MEANS CLUSTERING SYSTEM INITIALIZATION ===")
        optimizer.setup_kmeans_clustering_system()  # Use the new method
        
        # Run optimization
        print("\n=== SENSOR PLACEMENT OPTIMIZATION ===")
        result = optimizer.run_optimization()

        if result.X is not None and len(result.X) > 0:
            print("\n=== SOLUTION ANALYSIS ===")
            solutions = optimizer.analyze_pareto_front(result)
            
        if solutions:
            # Use best coverage solution for clustering analysis
            name, idx, chromosome = solutions[0]
            deployed_sensors = optimizer.problem.get_sensor_positions(chromosome)
            metrics = optimizer.problem.evaluate_solution(chromosome)
            
            print(f"\nAnalyzing {name} solution with {len(deployed_sensors)} sensors")
            
            # Perform K-means clustering with hybrid method
            print("\n=== K-MEANS CLUSTERING ANALYSIS (HYBRID METHOD) ===")
            
            # Option 1: Use hybrid method (recommended)
            assignments, cluster_heads, clustering_info = optimizer.perform_kmeans_clustering_analysis(
                deployed_sensors, use_hybrid=True
            )
                
            # Create comprehensive visualization
            print("\n=== VISUALIZATION GENERATION ===")
            
            # Plot K-means clustering results
            optimizer.visualize_kmeans_clustering(deployed_sensors, assignments, cluster_heads, metrics)
            
            # Print clustering summary
            print("\n=== K-MEANS CLUSTERING SUMMARY ===")
            print(f"Number of clusters: {len(np.unique(assignments))}")
            print(f"Total SSE: {clustering_info['total_sse']:.2f}")
            print(f"Iterations: {clustering_info['iterations']}")
            print(f"Converged: {clustering_info['converged']}")
            print(f"Cluster sizes: {clustering_info['cluster_sizes']}")
            print(f"Connectivity rate: {metrics.connectivity_rate:.1f}%")
            
                       
        return result, optimizer.problem
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    try:
        print("Starting Sensor Placement Optimization with Integrated Fuzzy Clustering...")
        print("=" * 70)
        
        result, problem = main()
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()