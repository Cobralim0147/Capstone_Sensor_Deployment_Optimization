"""
Sensor Placement and Path Optimization using SPEA2, K-Means, and ACO

This module provides a complete workflow for optimizing sensor networks in
agricultural fields. It handles:
1. Generating a virtual field environment.
2. Optimizing sensor placement using the SPEA2 multi-objective algorithm.
3. Clustering the deployed sensors into manageable groups using K-Means.
4. Planning an optimal data collection path for a mobile sink using Ant Colony
   Optimization (ACO), including obstacle avoidance.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import logging
import os 
import sys

# Custom module imports with error handling
try:
    from environment.sym_env import environment_generator
    from environment.sym_env import SymmetricFieldGenerator
    from environment.l_shaped_env import LShapedFieldGenerator
    from environment.circ_env import CircularFieldGenerator
    from environment.rand_env import RandomFieldGenerator
    from visualization.visualization import PlotUtils
    from utilities.veg_sensor_problem import VegSensorProblem 
    from algorithms.spea2 import SPEA2
    from algorithms.k_mean import KMeansClustering
    from algorithms.ant_colony import AntColonyOptimizer

    from configurations.config_file import OptimizationConfig
    # from algorithms.integrated_clustering import IntegratedClusteringSystem
    from analysis.clustering_comparison import FuzzyClusteringOptimizer


except ImportError as e:
    logging.warning(f"Custom module import failed: {e}")
    print(f"Warning: {e}. Please ensure all custom modules are in the Python path.")


class SensorOptimizer:
    """
    Main class orchestrating the entire optimization workflow, from field
    generation to sensor placement, clustering, and path planning.
    """

    # =========================================================================
    # 1. Initialization & Environment Setup
    # =========================================================================

    def __init__(self, config: OptimizationConfig):
        """Initializes the main optimizer class."""
        self.config = config
        self.field_data = None
        self.problem = None
        self.clustering_system = None
        self.integrated_clustering = None
        self.fuzzy_validator = None

    def generate_field_environment(self) -> Dict[str, Any]:
        """
        Generates the virtual agricultural field, including beds, furrows, and
        vegetable locations based on the configuration file.
        """
        try:

            # check for environment shape
            if self.config.environment_shape == 'symmetric':
                self.environment_generator = SymmetricFieldGenerator()
            elif self.config.environment_shape == 'L':
                self.environment_generator = LShapedFieldGenerator()
            elif self.config.environment_shape == 'circular':
                self.environment_generator = CircularFieldGenerator()   
            elif self.config.environment_shape == 'irregular':
                self.environment_generator = RandomFieldGenerator()

            print("Generating field environment...")
            results = self.environment_generator.generate_environment(
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
            
            field_map, grid_size, length, width, monitor_loc, veg_pos, bed_coords = results
            
            self.field_data = {
                'field_map': field_map,
                'environment_grid_size': grid_size,
                'environment_field_length': length,
                'environment_field_width': width,
                'monitor_location': monitor_loc,
                'vegetable_pos': veg_pos,
                'bed_coords': bed_coords
            }
            
            print(f"Field generated: {length}m x {width}m, "
                  f"{len(bed_coords)} beds, {len(veg_pos)} vegetables")
            return self.field_data
            
        except Exception as e:
            print(f"Failed to generate field environment: {e}")
            raise

    def setup_generated_field(self) -> None:
        """
        Initializes the optimization problem (VegSensorProblem) using the
        generated field data. This defines the objectives and constraints for SPEA2.
        """
        if self.field_data is None:
            raise ValueError("Field data not generated. Call generate_field_environment() first.")
        
        try:
            print("Setting up optimization problem for sensor placement...")
            veg_pos = self.field_data['vegetable_pos']
            veg_x = np.array([p[0] for p in veg_pos]) if veg_pos else np.array([])
            veg_y = np.array([p[1] for p in veg_pos]) if veg_pos else np.array([])
            
            self.problem = VegSensorProblem(
                veg_x=veg_x, 
                veg_y=veg_y,
                bed_coords=self.field_data['bed_coords'],
                sensor_range=self.config.deployment_sensor_range,
                max_sensors=self.config.deployment_max_sensors,
                comm_range=self.config.deployment_comm_range,
                grid_spacing=self.config.deployment_grid_space
            )
            
            print(f"Problem setup complete: {len(self.problem.possible_positions)} possible positions, "
                  f"{self.problem.n_var} variables, {self.problem.n_obj} objectives")
            
        except Exception as e:
            print(f"Failed to setup optimization problem: {e}")
            raise
    
    # =========================================================================
    # 2. Phase 1 - Sensor Placement (SPEA2)
    # =========================================================================

    def run_optimization(self) -> Any:
        """
        Executes the SPEA2 multi-objective optimization algorithm to find a set
        of non-dominated sensor placement solutions (the Pareto front).
        """
        if self.problem is None:
            raise ValueError("Problem not set up. Call setup_generated_field() first.")
        
        try:
            print("Starting SPEA2 optimization for sensor placement...")
            algorithm = SPEA2(problem=self.problem)            
            final_archive, history = algorithm.run()
            
            class Result: # Simple class to mimic the pymoo result object structure
                def __init__(self, archive, history):
                    self.X = np.array([ind.chromosome for ind in archive])
                    self.F = np.array([ind.objectives for ind in archive])
                    self.history = history
                    
            result = Result(final_archive, history)
            print("Optimization completed successfully.")
            return result

        except Exception as e:
            print(f"Optimization failed: {e}")
            raise

    def analyze_pareto_front(self, result: Any) -> List[Tuple[str, int, np.ndarray]]:
        """
        Analyzes the Pareto front from the SPEA2 result to identify key solutions,
        such as the one with the best coverage or best efficiency.
        """
        if result.X is None or len(result.X) == 0:
            print("No solutions found in optimization result.")
            return []
        
        try:
            F, X = result.F, result.X
            print(f"Analyzing Pareto front with {len(X)} solutions...")
            self._log_objective_ranges(F)
            
            solutions = []
            solutions.append(("Best Coverage", np.argmin(F[:, 1]), X[np.argmin(F[:, 1])]))
            efficiency = -F[:, 1] / (F[:, 0] + 1e-6)
            solutions.append(("Most Efficient", np.argmax(efficiency), X[np.argmax(efficiency)]))
            solutions.append(("Best Connectivity", np.argmin(F[:, 4]), X[np.argmin(F[:, 4])]))
            
            self._log_solution_details(solutions)
            return solutions
            
        except Exception as e:
            print(f"Failed to analyze Pareto front: {e}")
            raise
            
    def visualize_solution(self, chromosome: np.ndarray, title: str) -> None:
        """
        Visualizes a single sensor placement solution on the field map, showing
        sensor locations, coverage radii, and performance metrics.
        """
        if self.field_data is None or self.problem is None:
            print("Field data or problem not initialized for visualization.")
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
            plt.title(f'{title}\nCoverage: {metrics.coverage_rate:.1f}%, '
                      f'Sensors: {metrics.n_sensors}, Connectivity: {metrics.connectivity_rate:.1f}%')
            plt.xlabel('X (m)'); plt.ylabel('Y (m)')
            plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal'); plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Failed to visualize solution: {e}")
            raise

    # =========================================================================
    # 3. Phase 2 - Clustering (K-Means)
    # =========================================================================

    def setup_kmeans_clustering_system(self) -> None:
        """Initializes the K-means clustering system."""
        if self.field_data is None:
            raise ValueError("Field data not generated. Call generate_field_environment() first.")
        try:
            self.clustering_system = KMeansClustering()
            print("K-means clustering system initialized.")
        except Exception as e:
            print(f"Failed to setup K-means clustering system: {e}")
            raise

    def perform_kmeans_clustering_analysis(self, sensor_positions: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Applies optimized K-means clustering that ensures 100% coverage before optimization.
        """
        if self.clustering_system is None:
            raise ValueError("K-means clustering system not initialized.")
        
        try:
            print(f"Performing optimized K-means clustering on {len(sensor_positions)} sensors...")
            print("Strategy: Ensure 100% coverage → Optimize cluster count → Converge")
            
            assignments, cluster_heads, info = self.clustering_system.perform_clustering(
                sensor_positions, n_clusters=n_clusters, random_state=self.config.deployment_random_seed
            )
            
            # Enhanced logging for the new approach
            print(f"\n📊 CLUSTERING RESULTS:")
            print(f"   Coverage Status: {'✅ 100% Achieved' if info.get('coverage_achieved', False) else '❌ Not Achieved'}")
            print(f"   Initial clusters (for coverage): {info.get('initial_clusters_for_coverage', 'N/A')}")
            print(f"   Final optimized clusters: {info.get('final_optimized_clusters', len(np.unique(assignments)))}")
            print(f"   Optimization attempts: {info.get('optimization_attempts', 'N/A')}")
            print(f"   Clusters reduced: {info.get('clusters_reduced', 'N/A')}")
            
            # Analyze connectivity
            connectivity = self.clustering_system.analyze_cluster_connectivity(sensor_positions, assignments, cluster_heads)
            print(f"\n📡 CONNECTIVITY ANALYSIS:")
            for cid, c_info in connectivity.items():
                print(f"   Cluster {cid + 1}: {c_info['cluster_size']} sensors, "
                    f"Connectivity: {c_info['connectivity_rate']:.1f}%")

            return assignments, cluster_heads, info
            
        except Exception as e:
            print(f"Failed to perform optimized K-means clustering: {e}")
            raise

    def visualize_kmeans_clustering(self, sensor_positions: np.ndarray, assignments: np.ndarray, cluster_heads: np.ndarray) -> None:
        """Visualizes the K-means clustering results on the field map."""
        if self.clustering_system is None:
            print("K-means clustering system not initialized.")
            return
        
        try:
            print("Generating K-means clustering visualization...")
            self.clustering_system.plot_clustering_results(
                sensor_positions, assignments, cluster_heads,
                title="K-means Clustering Results"
            )
        except Exception as e:
            print(f"Failed to visualize K-means clustering: {e}")
            raise

    # =========================================================================
    # 4. Phase 3 - Mobile Sink Path Optimization (ACO)
    # =========================================================================

    def setup_mobile_sink_optimization(self) -> None:
        """Placeholder to signal initialization of the path optimization phase."""
        print("Mobile sink path optimization system initialized.")

    def optimize_rover_path(self, cluster_heads_positions: np.ndarray) -> Tuple[List[int], float, np.ndarray, AntColonyOptimizer]:
        """
        Finds the optimal data collection path for the mobile sink using the
        Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesperson
        Problem for the cluster heads with obstacle avoidance.
        """
        try:
            print(f"\nOptimizing rover path for {len(cluster_heads_positions)} cluster heads...")
            aco_optimizer = AntColonyOptimizer(
                cluster_heads=cluster_heads_positions,
                field_data=self.field_data
            )
            
            optimal_path, total_distance, _ = aco_optimizer.optimize()
            
            # Get detailed path coordinates with obstacle avoidance waypoints
            path_coordinates = aco_optimizer.get_full_path_for_plotting(optimal_path)
            
            print(f"Optimal path found: Total distance = {total_distance:.2f} m")
            print(f"Path includes {len(path_coordinates)} waypoints (with detours)")
            
            return optimal_path, total_distance, path_coordinates, aco_optimizer
            
        except Exception as e:
            print(f"Failed to optimize rover path: {e}")
            raise

    def analyze_path_efficiency(self, total_distance: float, path_coordinates: np.ndarray) -> dict:
        """
        Analyzes the efficiency of the generated path using pre-calculated distance.
        """
        try:
            if len(path_coordinates) <= 1:
                return {'total_distance': 0, 'num_waypoints': len(path_coordinates), 'avg_segment_length': 0}

            analysis = {
                'total_distance': total_distance,
                'num_waypoints': len(path_coordinates),
                'avg_segment_length': total_distance / (len(path_coordinates) - 1)
            }
            
            print(f"Path efficiency analysis:")
            print(f"  - Total distance: {analysis['total_distance']:.2f} m")
            print(f"  - Number of waypoints: {analysis['num_waypoints']}")
            print(f"  - Average segment length: {analysis['avg_segment_length']:.2f} m")
            
            return analysis
            
        except Exception as e:
            print(f"Failed to analyze path efficiency: {e}")
            return {}
        
    def visualize_rover_path(self, path_coordinates: np.ndarray, cluster_heads_positions: np.ndarray,
                            sensor_positions: np.ndarray, assignments: np.ndarray,
                            aco_optimizer: AntColonyOptimizer, optimal_path: List[int]) -> None:
        """
        Visualizes the final optimized rover path with obstacle avoidance, showing the route between
        the base station and cluster heads, overlaid on the sensor clusters and field map.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            self._plot_field_environment(ax, self.field_data)
            
            # Plot sensors and clusters
            colors = plt.cm.tab10
            for i in range(len(cluster_heads_positions)):
                cluster_sensors = sensor_positions[assignments == i]
                ax.scatter(cluster_sensors[:, 0], cluster_sensors[:, 1], 
                        color=colors(i), s=40, alpha=0.6, label=f'Cluster {i+1}')
            
            # Plot cluster heads
            ax.scatter(cluster_heads_positions[:, 0], cluster_heads_positions[:, 1], 
                    c='red', s=150, marker='^', label='Cluster Heads', zorder=5)
            
            # Plot base station
            ax.scatter(path_coordinates[0, 0], path_coordinates[0, 1], 
                    c='black', s=200, marker='s', label='Base Station', zorder=6)
            
            # Plot the detailed path with obstacle avoidance
            ax.plot(path_coordinates[:, 0], path_coordinates[:, 1], 
                'r-', lw=3, alpha=0.8, label='Rover Path (with detours)')
            
            # Plot waypoints to show detours
            ax.scatter(path_coordinates[1:-1, 0], path_coordinates[1:-1, 1], 
                    c='orange', s=50, marker='o', alpha=0.7, label='Waypoints', zorder=4)
            
            # Add direction arrows
            for i in range(0, len(path_coordinates) - 1, max(1, len(path_coordinates) // 10)):
                dx = path_coordinates[i+1, 0] - path_coordinates[i, 0]
                dy = path_coordinates[i+1, 1] - path_coordinates[i, 1]
                ax.arrow(path_coordinates[i, 0], path_coordinates[i, 1], 
                        dx * 0.3, dy * 0.3,
                        head_width=1, head_length=1, fc='red', ec='red', alpha=0.7)
            
            # Add path segment analysis - FIXED CODE
            direct_segments = 0
            detour_segments = 0
            
            for i in range(len(optimal_path) - 1):
                start_node = optimal_path[i]
                end_node = optimal_path[i + 1]
                
                start_coords = aco_optimizer.grid_nodes[start_node]
                end_coords = aco_optimizer.grid_nodes[end_node]
                
                # Get the A* path between these nodes
                path_indices = aco_optimizer._find_grid_path(start_node, end_node)
                
                # Check if this segment uses detours (more than 2 waypoints means detour)
                if len(path_indices) > 2:
                    detour_segments += 1
                else:
                    direct_segments += 1
            
            # Calculate total distance from path coordinates
            total_distance = sum(np.linalg.norm(path_coordinates[i+1] - path_coordinates[i]) 
                            for i in range(len(path_coordinates) - 1))
            
            # Add title with path information
            ax.set_title(f'Mobile Sink Path Optimization\n'
                        f'Total Distance: {total_distance:.2f}m | '
                        f'Segments: {direct_segments} direct, {detour_segments} detours', 
                        fontsize=14, fontweight='bold')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()
            
            # Print detailed path analysis
            print(f"\nPath Analysis:")
            print(f"  - Direct segments: {direct_segments}")
            print(f"  - Detour segments: {detour_segments}")
            print(f"  - Total waypoints: {len(path_coordinates)}")
            
        except Exception as e:
            print(f"Failed to visualize rover path: {e}")
            import traceback
            traceback.print_exc()   


    def visualize_field_occupancy(self, field_data: dict) -> None:
        """
        Simple visualization to check bed coordinates by showing occupied (black) 
        and free (white) spaces in the field.
        """
        if not field_data or 'bed_coords' not in field_data:
            print("No field data or bed coordinates available")
            return
        
        try:
            # Get field dimensions
            field_length = field_data.get('environment_field_length', 100)
            field_width = field_data.get('environment_field_width', 60)
            bed_coords = field_data['bed_coords']
            
            print(f"Field size: {field_length}m x {field_width}m")
            print(f"Number of beds: {len(bed_coords)}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set white background (free space)
            ax.set_facecolor('white')
            
            # Draw field border (thick black)
            border = patches.Rectangle((0, 0), field_length, field_width, 
                                    fill=False, edgecolor='black', linewidth=4)
            ax.add_patch(border)
            
            # Draw beds as black rectangles
            for i, bed in enumerate(bed_coords):
                if len(bed) >= 4:  # [x1, y1, x2, y2]
                    bed_width = abs(bed[2] - bed[0])
                    bed_height = abs(bed[3] - bed[1])
                    
                    # Draw bed as black rectangle
                    bed_rect = patches.Rectangle((bed[0], bed[1]), bed_width, bed_height,
                                            facecolor='black', edgecolor='black')
                    ax.add_patch(bed_rect)
                    
                    # Debug: print bed coordinates
                    print(f"Bed {i+1}: [{bed[0]:.1f}, {bed[1]:.1f}, {bed[2]:.1f}, {bed[3]:.1f}] "
                        f"Size: {bed_width:.1f}x{bed_height:.1f}")
            
            # Calculate and display occupancy
            total_field_area = field_length * field_width
            total_bed_area = sum(abs(bed[2] - bed[0]) * abs(bed[3] - bed[1]) 
                            for bed in bed_coords if len(bed) >= 4)
            free_area = total_field_area - total_bed_area
            occupancy_percent = (total_bed_area / total_field_area) * 100
            
            print(f"Total field area: {total_field_area:.1f} m²")
            print(f"Total bed area: {total_bed_area:.1f} m²")
            print(f"Free area: {free_area:.1f} m² ({100-occupancy_percent:.1f}%)")
            print(f"Bed occupancy: {occupancy_percent:.1f}%")
            
            # Set axis properties
            ax.set_xlim(-5, field_length + 5)
            ax.set_ylim(-5, field_width + 5)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title('Field Occupancy Check\nBlack = Beds (Occupied), White = Free Space', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add text annotation
            ax.text(0.02, 0.98, f'Bed Occupancy: {occupancy_percent:.1f}%\nFree Space: {100-occupancy_percent:.1f}%', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Failed to visualize field occupancy: {e}")
            import traceback
            traceback.print_exc()

    def debug_field_occupancy(self) -> None:
        """Debug method to visualize field occupancy and check bed coordinates."""
        if self.field_data is None:
            print("Field data not generated. Call generate_field_environment() first.")
            return
        
        self.visualize_field_occupancy(self.field_data)

    # =========================================================================
    # 5. Reporting & Utility Methods
    # =========================================================================

    def _log_objective_ranges(self, F: np.ndarray) -> None:
        """Helper to log the min/max values for each objective."""
        print("Objective ranges on Pareto front:")
        print(f"  Sensor Count: [{F[:, 0].min():.1f}, {F[:, 0].max():.1f}]")
        print(f"  Coverage Rate: [{-F[:, 1].max():.1f}, {-F[:, 1].min():.1f}]")

    def _log_solution_details(self, solutions: List[Tuple[str, int, np.ndarray]]) -> None:
        """Helper to log the metrics of specific, named solutions."""
        for name, idx, chromosome in solutions:
            metrics = self.problem.evaluate_solution(chromosome)
            print(f"{name} Solution: Sensors={metrics.n_sensors}, Coverage={metrics.coverage_rate:.1f}%, "
                  f"Connectivity={metrics.connectivity_rate:.1f}%")

    def _plot_field_environment(self, ax, field_data):
        """Helper method to plot the field beds and vegetables on a given axis."""
        try:
            if 'bed_coords' in field_data:
                for bed in field_data['bed_coords']:
                    rect = patches.Rectangle((bed[0], bed[1]), bed[2]-bed[0], bed[3]-bed[1],
                                             facecolor='lightblue', alpha=0.3)
                    ax.add_patch(rect)
            if 'vegetable_pos' in field_data and field_data['vegetable_pos']:
                veg_pos = np.array(field_data['vegetable_pos'])
                ax.scatter(veg_pos[:, 0], veg_pos[:, 1], c='green', s=20, alpha=0.6, marker='s', label='Vegetables')
        except Exception as e:
            print(f"Warning: Could not plot field environment: {e}")

# =========================================================================
# Main Execution Block
# =========================================================================

def main() -> None:
    """Main function to run the full optimization workflow."""
    config = OptimizationConfig()
    optimizer = SensorOptimizer(config)
    
    try:
        # === Step 1: Environment and Problem Setup ===
        print("--- STEP 1: ENVIRONMENT SETUP ---")
        optimizer.generate_field_environment()
        optimizer.setup_generated_field()
        optimizer.setup_kmeans_clustering_system()
        
        # === Step 2: Sensor Placement Optimization (SPEA2) ===
        print("\n--- STEP 2: SENSOR PLACEMENT OPTIMIZATION ---")
        spea_result = optimizer.run_optimization()

        if not (spea_result and spea_result.X is not None and len(spea_result.X) > 0):
            print("Optimization did not yield any solutions. Exiting.")
            return

        solutions = optimizer.analyze_pareto_front(spea_result)
        if not solutions:
            print("Pareto front analysis failed. Exiting.")
            return

        # Select the "Best Coverage" solution for the next steps
        name, _, chromosome = solutions[0]
        deployed_sensors = optimizer.problem.get_sensor_positions(chromosome)
        metrics = optimizer.problem.evaluate_solution(chromosome)
        print(f"\nProceeding with '{name}' solution ({len(deployed_sensors)} sensors).")

        optimizer.visualize_solution(chromosome, title=f"Best Coverage Solution ({len(deployed_sensors)} sensors)")
        
        # === Step 3: Clustering (K-Means) ===
        print("\n--- STEP 3: SENSOR CLUSTERING ---")
        assignments, cluster_heads_indices, _ = optimizer.perform_kmeans_clustering_analysis(deployed_sensors)
        cluster_heads_indices = np.asarray(cluster_heads_indices, dtype=int)
        cluster_heads_positions = deployed_sensors[cluster_heads_indices]

        optimizer.visualize_kmeans_clustering(deployed_sensors, assignments, cluster_heads_indices)

        # === Step 4: Path Optimization (ACO) ===
        print("\n--- STEP 4: ROVER PATH OPTIMIZATION ---")
        optimal_path, total_distance, path_coords, aco_optimizer = optimizer.optimize_rover_path(cluster_heads_positions)
        path_analysis = optimizer.analyze_path_efficiency(total_distance, path_coords)

        # === Step 5: Visualization & Summary ===
        print("\n--- STEP 5: VISUALIZATION & SUMMARY ---")
        optimizer.visualize_rover_path(
            path_coordinates=path_coords,
            cluster_heads_positions=cluster_heads_positions, 
            sensor_positions=deployed_sensors,
            assignments=assignments,
            aco_optimizer=aco_optimizer,
            optimal_path=optimal_path
        )
        
        print("\n=== FINAL SUMMARY ===")
        print(f"Sensor Deployment:      {len(deployed_sensors)} sensors")
        # print(f"Clustering:             {len(cluster_heads_positions)} cluster heads")
        print(f"Field Coverage:         {metrics.coverage_rate:.1f}%")
        print(f"Network Connectivity:   {metrics.connectivity_rate:.1f}%")
        # print(f"Rover Path Distance:    {total_distance:.2f} m")
        
    except Exception as e:
        print(f"An error occurred during the main workflow: {e}")
        raise

if __name__ == "__main__":
    """Script entry point."""
    try:
        print("=" * 70)
        print("Starting Full Sensor Network and Path Optimization Workflow")
        print("=" * 70)
        main()
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nFATAL ERROR: The optimization workflow failed. Reason: {e}")
        import traceback
        traceback.print_exc()