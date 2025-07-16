"""
Ant Colony Optimization Real-Time Visualizer

This module provides real-time visualization of the ACO algorithm by importing
and using the existing code from main2.py and algorithms.ant_colony.py.
The purpose is to debug and understand why the ACO is not finding optimal paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import time

# Import your existing code
from main import SensorOptimizer
from algorithms.ant_colony import AntColonyOptimizer
from configurations.config_file import OptimizationConfig

class ACOVisualizer:
    """
    Real-time visualizer for ACO algorithm that uses your existing code structure.
    Shows individual ant path construction and pheromone evolution.
    """
    
    def __init__(self):
        """Initialize the visualizer using your existing code."""
        self.config = OptimizationConfig()
        self.optimizer = SensorOptimizer(self.config)
        
        # Setup the field and sensors using your existing workflow
        self._setup_environment()
        
        # Variables for visualization
        self.fig = None
        self.axes = None
        self.current_iteration = 0
        self.ant_paths_history = []
        self.pheromone_history = []
        
    def _setup_environment(self):
        """Setup the environment using your existing main2.py workflow."""
        print("Setting up environment using existing main2.py code...")
        
        # Step 1: Generate field environment
        self.optimizer.generate_field_environment()
        self.optimizer.setup_generated_field()
        self.optimizer.setup_kmeans_clustering_system()
        
        # Step 2: Run sensor placement optimization
        print("Running sensor placement optimization...")
        result = self.optimizer.run_optimization()
        
        if not (result and result.X is not None and len(result.X) > 0):
            raise ValueError("Sensor placement optimization failed")
        
        # Get best solution
        solutions = self.optimizer.analyze_pareto_front(result)
        name, _, chromosome = solutions[0]  # Use "Best Coverage" solution
        self.deployed_sensors = self.optimizer.problem.get_sensor_positions(chromosome)
        
        # Step 3: Perform clustering
        print("Performing clustering...")
        assignments, cluster_heads_indices, _ = self.optimizer.perform_kmeans_clustering_analysis(self.deployed_sensors)
        self.cluster_heads_positions = self.deployed_sensors[cluster_heads_indices]
        self.assignments = assignments
        
        print(f"Environment setup complete:")
        print(f"  - Field: {self.optimizer.field_data['environment_field_length']}m x {self.optimizer.field_data['environment_field_width']}m")
        print(f"  - Sensors: {len(self.deployed_sensors)}")
        print(f"  - Cluster heads: {len(self.cluster_heads_positions)}")
        print(f"  - Beds: {len(self.optimizer.field_data['bed_coords'])}")
        
    def visualize_aco_process(self):
        """
        Run ACO with real-time visualization using your existing ACO code.
        """
        print("\n" + "="*60)
        print("STARTING ACO VISUALIZATION")
        print("="*60)
        
        # Create ACO optimizer using your existing code
        aco_optimizer = AntColonyOptimizer(
            cluster_heads=self.cluster_heads_positions,
            field_data=self.optimizer.field_data
        )
        
        # Setup visualization
        self._setup_visualization_plots()
        
        # Run modified ACO algorithm with visualization
        self._run_aco_with_visualization(aco_optimizer)
        
        # Show final results
        self._show_final_results(aco_optimizer)
        
    def _setup_visualization_plots(self):
        """Setup the visualization plots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('ACO Real-Time Visualization', fontsize=16, fontweight='bold')
        
        # Plot titles
        self.axes[0, 0].set_title('Current Ant Path Construction')
        self.axes[0, 1].set_title('Best Path So Far')
        self.axes[1, 0].set_title('Pheromone Trails')
        self.axes[1, 1].set_title('Convergence History')
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show()
        
    def _run_aco_with_visualization(self, aco_optimizer):
        """
        Run the ACO algorithm with step-by-step visualization.
        This modifies the existing ACO optimize() method to add visualization.
        """
        print(f"Starting ACO with visualization...")
        print(f"Nodes: {aco_optimizer.n_nodes} (1 base + {len(aco_optimizer.cluster_heads)} cluster heads)")
        print(f"Buffer distance: {aco_optimizer.config.collection_buffer_distance}m")
        
        # Initialize ACO variables (same as your existing code)
        aco_optimizer.best_path = None
        aco_optimizer.best_distance = float('inf')
        aco_optimizer.convergence_history = []
        
        for iteration in range(aco_optimizer.config.collection_n_iterations):
            print(f"\n--- ITERATION {iteration + 1} ---")
            self.current_iteration = iteration
            
            iteration_ant_paths = []
            iteration_ant_distances = []
            
            # Run ants for this iteration (same as your existing code)
            for ant_id in range(aco_optimizer.config.collection_n_ants):
                print(f"  Ant {ant_id + 1}: ", end="")
                
                # Construct ant solution using your existing method
                path, distance = aco_optimizer._construct_ant_solution()
                
                # Store ant data
                iteration_ant_paths.append(path)
                iteration_ant_distances.append(distance)
                
                # Update local pheromones using your existing method
                aco_optimizer._update_local_pheromones(path, distance)
                
                # Check for new best (same as your existing code)
                if distance < aco_optimizer.best_distance:
                    aco_optimizer.best_distance = distance
                    aco_optimizer.best_path = path.copy()
                    print(f"NEW BEST! Distance = {distance:.2f}m")
                else:
                    print(f"Distance = {distance:.2f}m")
                
                # Visualize this ant's path
                self._visualize_current_ant(aco_optimizer, ant_id, path, distance)
                
                # Small delay to see the progression
                plt.pause(0.1)
            
            # Update global pheromones using your existing method
            aco_optimizer._update_global_pheromones()
            
            # Record convergence (same as your existing code)
            aco_optimizer.convergence_history.append(aco_optimizer.best_distance)
            
            # Store iteration data
            self.ant_paths_history.append(iteration_ant_paths)
            self.pheromone_history.append(aco_optimizer.pheromone_matrix.copy())
            
            # Update iteration summary visualization
            self._update_iteration_summary(aco_optimizer)
            
            # Pause between iterations
            plt.pause(0.5)
        
        print(f"\nACO completed. Best distance: {aco_optimizer.best_distance:.2f}m")
        
    def _visualize_current_ant(self, aco_optimizer, ant_id, path, distance):
        """Visualize the current ant's path construction."""
        ax = self.axes[0, 0]
        ax.clear()
        
        # Draw field environment
        self._draw_field_environment(ax)
        
        # Draw the ant's path with detailed waypoints
        self._draw_ant_path(ax, aco_optimizer, path, ant_id)
        
        ax.set_title(f'Ant {ant_id + 1} Path (Iteration {self.current_iteration + 1})\nDistance: {distance:.2f}m')
        ax.set_xlim(-5, self.optimizer.field_data['environment_field_length'] + 5)
        ax.set_ylim(-5, self.optimizer.field_data['environment_field_width'] + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
    def _draw_ant_path(self, ax, aco_optimizer, path, ant_id):
        """Draw the ant's path with detailed waypoints using your existing code."""
        # Draw path segments between nodes
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            start_coords = aco_optimizer.rendezvous_nodes[start_node]
            end_coords = aco_optimizer.rendezvous_nodes[end_node]
            
            # Use your existing pathfinding code to get detailed waypoints
            try:
                distance, waypoints = aco_optimizer._find_best_path_segment(start_coords, end_coords)
                
                # Draw the path segment
                if len(waypoints) > 2:
                    # This is a detour path
                    waypoints_array = np.array(waypoints)
                    ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'b-', linewidth=2, alpha=0.7)
                    ax.scatter(waypoints_array[1:-1, 0], waypoints_array[1:-1, 1], c='blue', s=30, alpha=0.8)
                    
                    # Check if this path goes through obstacles
                    if not aco_optimizer._is_path_clear(start_coords, end_coords):
                        ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 
                               'r--', linewidth=1, alpha=0.5, label='Direct path (blocked)')
                else:
                    # Direct path
                    ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 
                           'g-', linewidth=2, alpha=0.7)
                    
            except Exception as e:
                print(f"Error visualizing path segment: {e}")
                # Fallback to direct line
                ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 
                       'r-', linewidth=1, alpha=0.5)
        
        # Draw nodes
        self._draw_nodes(ax, aco_optimizer)
        
        # Highlight the path order
        for i, node_id in enumerate(path):
            node_coords = aco_optimizer.rendezvous_nodes[node_id]
            ax.annotate(f'{i}', (node_coords[0], node_coords[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
    def _update_iteration_summary(self, aco_optimizer):
        """Update the iteration summary plots."""
        # Best path visualization
        ax1 = self.axes[0, 1]
        ax1.clear()
        self._draw_field_environment(ax1)
        
        if aco_optimizer.best_path:
            self._draw_best_path(ax1, aco_optimizer)
        
        self._draw_nodes(ax1, aco_optimizer)
        ax1.set_title(f'Best Path (Iteration {self.current_iteration + 1})\nDistance: {aco_optimizer.best_distance:.2f}m')
        ax1.set_xlim(-5, self.optimizer.field_data['environment_field_length'] + 5)
        ax1.set_ylim(-5, self.optimizer.field_data['environment_field_width'] + 5)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Pheromone visualization
        ax2 = self.axes[1, 0]
        ax2.clear()
        self._draw_field_environment(ax2)
        self._draw_pheromone_trails(ax2, aco_optimizer)
        self._draw_nodes(ax2, aco_optimizer)
        ax2.set_title(f'Pheromone Trails (Iteration {self.current_iteration + 1})')
        ax2.set_xlim(-5, self.optimizer.field_data['environment_field_length'] + 5)
        ax2.set_ylim(-5, self.optimizer.field_data['environment_field_width'] + 5)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Convergence plot
        ax3 = self.axes[1, 1]
        ax3.clear()
        if len(aco_optimizer.convergence_history) > 0:
            ax3.plot(range(1, len(aco_optimizer.convergence_history) + 1), 
                    aco_optimizer.convergence_history, 'b-o', linewidth=2, markersize=4)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Best Distance (m)')
            ax3.set_title('Convergence History')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        
    def _draw_field_environment(self, ax):
        """Draw the field environment using your existing field_data."""
        # Field boundary
        field_length = self.optimizer.field_data['environment_field_length']
        field_width = self.optimizer.field_data['environment_field_width']
        
        field_rect = patches.Rectangle((0, 0), field_length, field_width, 
                                     fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(field_rect)
        
        # Draw beds using your existing field_data
        bed_coords = self.optimizer.field_data['bed_coords']
        for i, bed in enumerate(bed_coords):
            if len(bed) >= 4:
                # Draw bed
                bed_width = abs(bed[2] - bed[0])
                bed_height = abs(bed[3] - bed[1])
                bed_rect = patches.Rectangle((bed[0], bed[1]), bed_width, bed_height,
                                           facecolor='black', edgecolor='black', alpha=0.6)
                ax.add_patch(bed_rect)
                
                # Draw buffer zone
                buffer = self.config.collection_buffer_distance
                buffer_rect = patches.Rectangle((bed[0] - buffer, bed[1] - buffer), 
                                              bed_width + 2*buffer, bed_height + 2*buffer,
                                              fill=False, edgecolor='red', linewidth=1, 
                                              linestyle='--', alpha=0.5)
                ax.add_patch(buffer_rect)
        
        # Draw sensors and clusters
        colors = plt.cm.tab10
        for i in range(len(self.cluster_heads_positions)):
            cluster_sensors = self.deployed_sensors[self.assignments == i]
            if len(cluster_sensors) > 0:
                ax.scatter(cluster_sensors[:, 0], cluster_sensors[:, 1], 
                          color=colors(i), s=20, alpha=0.4)
        
    def _draw_nodes(self, ax, aco_optimizer):
        """Draw base station and cluster heads."""
        # Base station
        base_station = aco_optimizer.rendezvous_nodes[0]
        ax.scatter(base_station[0], base_station[1], c='black', s=200, marker='s', 
                  label='Base Station', zorder=5)
        
        # Cluster heads
        cluster_heads = aco_optimizer.rendezvous_nodes[1:]
        ax.scatter(cluster_heads[:, 0], cluster_heads[:, 1], c='red', s=100, marker='^', 
                  label='Cluster Heads', zorder=5)
        
    def _draw_best_path(self, ax, aco_optimizer):
        """Draw the best path using your existing get_full_path_for_plotting method."""
        try:
            # Use your existing method to get detailed path
            detailed_path_coords = aco_optimizer.get_full_path_for_plotting(aco_optimizer.best_path)
            
            if len(detailed_path_coords) > 0:
                ax.plot(detailed_path_coords[:, 0], detailed_path_coords[:, 1], 
                       'b-', linewidth=3, alpha=0.8, label='Best Path')
                ax.scatter(detailed_path_coords[:, 0], detailed_path_coords[:, 1], 
                          c='blue', s=20, alpha=0.6, zorder=4)
        except Exception as e:
            print(f"Error drawing best path: {e}")
            
    def _draw_pheromone_trails(self, ax, aco_optimizer):
        """Draw pheromone trails between nodes."""
        max_pheromone = np.max(aco_optimizer.pheromone_matrix)
        if max_pheromone > 0:
            for i in range(aco_optimizer.n_nodes):
                for j in range(i + 1, aco_optimizer.n_nodes):
                    pheromone_strength = aco_optimizer.pheromone_matrix[i, j] / max_pheromone
                    if pheromone_strength > 0.1:  # Only show significant trails
                        start = aco_optimizer.rendezvous_nodes[i]
                        end = aco_optimizer.rendezvous_nodes[j]
                        
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                               'purple', linewidth=pheromone_strength * 4, alpha=0.6)
                        
    def _show_final_results(self, aco_optimizer):
        """Show final results and analysis."""
        print("\n" + "="*60)
        print("FINAL ACO RESULTS")
        print("="*60)
        print(f"Best path: {aco_optimizer.best_path}")
        print(f"Best distance: {aco_optimizer.best_distance:.2f}m")
        
        # Analyze path segments using your existing code
        if aco_optimizer.best_path:
            print("\nPath segment analysis:")
            total_detours = 0
            total_direct = 0
            problematic_segments = []
            
            for i in range(len(aco_optimizer.best_path) - 1):
                start_node = aco_optimizer.best_path[i]
                end_node = aco_optimizer.best_path[i + 1]
                start_coords = aco_optimizer.rendezvous_nodes[start_node]
                end_coords = aco_optimizer.rendezvous_nodes[end_node]
                
                try:
                    # Use your existing pathfinding code
                    distance, waypoints = aco_optimizer._find_best_path_segment(start_coords, end_coords)
                    direct_distance = np.linalg.norm(end_coords - start_coords)
                    
                    if len(waypoints) > 2:
                        total_detours += 1
                        print(f"  Segment {start_node}→{end_node}: DETOUR ({len(waypoints)} waypoints, {distance:.2f}m)")
                    else:
                        total_direct += 1
                        print(f"  Segment {start_node}→{end_node}: DIRECT ({distance:.2f}m)")
                    
                    # Check if direct path is blocked
                    if not aco_optimizer._is_path_clear(start_coords, end_coords):
                        problematic_segments.append((start_node, end_node))
                        
                except Exception as e:
                    print(f"  Segment {start_node}→{end_node}: ERROR - {e}")
                    
            print(f"\nSummary: {total_direct} direct paths, {total_detours} detours")
            
            if problematic_segments:
                print(f"\nSegments requiring detours: {len(problematic_segments)}")
                for start_node, end_node in problematic_segments:
                    print(f"  - Segment {start_node}→{end_node}")
            else:
                print("\nAll segments use direct paths (no obstacles detected)")
        
        # Keep the final visualization open
        plt.ioff()  # Turn off interactive mode
        plt.show()

def run_aco_visualization():
    """
    Main function to run the ACO visualization.
    Uses your existing code from main2.py and algorithms.ant_colony.py
    """
    try:
        visualizer = ACOVisualizer()
        visualizer.visualize_aco_process()
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("ACO Real-Time Visualizer")
    print("Uses existing code from main2.py and algorithms.ant_colony.py")
    print("Press Ctrl+C to stop the visualization")
    
    try:
        run_aco_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error: {e}")