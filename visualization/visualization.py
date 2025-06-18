"""
Visualization utilities for sensor placement and clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from algorithms.spea2 import SensorMetrics
import os
import sys
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
class PlotUtils:
    """Utility class for plotting field elements."""
    
    @staticmethod
    def plot_bed_boundaries(bed_coords: List[Tuple]) -> None:
        """Plot bed boundaries."""
        for i, bed in enumerate(bed_coords):
            x_min, y_min, x_max, y_max = bed
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min], 
                    'b-', linewidth=2, alpha=0.7, 
                    label='Beds' if i == 0 else "")
    
    @staticmethod
    def plot_vegetables(vegetable_pos: List[Tuple]) -> None:
        """Plot vegetable positions."""
        if vegetable_pos:
            veg_x = [pos[0] for pos in vegetable_pos]
            veg_y = [pos[1] for pos in vegetable_pos]
            plt.plot(veg_x, veg_y, 'g.', markersize=6, alpha=0.7, label='Vegetables')
    
    @staticmethod
    def plot_sensors(sensors: np.ndarray, metrics: SensorMetrics, sensor_range: float) -> None:
        """Plot sensors and their coverage."""
        if sensors.shape[0] > 0:
            plt.scatter(sensors[:, 0], sensors[:, 1], 
                       c='red', s=100, marker='^', 
                       label=f'Sensors ({metrics.n_sensors})')
            
            # Plot sensor ranges
            for pos in sensors:
                circle = plt.Circle(pos, sensor_range, 
                                  fill=False, linestyle='--', color='red', alpha=0.3)
                plt.gca().add_patch(circle)
    
    @staticmethod
    def plot_clustered_sensors(deployed_sensors: np.ndarray, assignments: np.ndarray, 
                             cluster_head_positions: np.ndarray, sensor_range: float) -> None:
        """Plot clustered sensors with connections and coverage."""
        # Plot sensor coverage circles
        for pos in deployed_sensors:
            circle = plt.Circle(pos, sensor_range, 
                              fill=False, linestyle='--', color='red', alpha=0.2)
            plt.gca().add_patch(circle)
        
        # Plot clusters
        unique_assignments = np.unique(assignments)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_assignments)))
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_assignments):
            mask = assignments == cluster_id
            cluster_sensors = deployed_sensors[mask]
            head_pos = cluster_head_positions[i]
            
            # Plot sensors in this cluster
            plt.scatter(cluster_sensors[:, 0], cluster_sensors[:, 1], 
                      c=[colors[i]], s=100, alpha=0.8, marker='^',
                      label=f'Cluster {cluster_id+1} ({sum(mask)} sensors)')
            
            # Plot connections from cluster head to its sensors
            for sensor_pos in cluster_sensors:
                plt.plot([head_pos[0], sensor_pos[0]], 
                       [head_pos[1], sensor_pos[1]], 
                       c=colors[i], linestyle='--', alpha=0.5)
        
        # Plot cluster heads
        plt.scatter(cluster_head_positions[:, 0], cluster_head_positions[:, 1], 
                   c='black', s=200, marker='*', label='Cluster Heads')

class FieldVisualizer:
    """Class for visualizing field layouts and environments."""
    
    @staticmethod
    def plot_field_layout(field_map: np.ndarray, grid_size: float, 
                         field_length: float, field_width: float, 
                         monitor_location: np.ndarray, vegetable_pos: List) -> None:
        """
        Plot the field layout with beds, furrows, and vegetables.
        
        Parameters:
        -----------
        field_map : np.ndarray
            2D array representing the field (0=furrow, 1=bed)
        grid_size : float
            Size of grid cells
        field_length : float
            Length of the field
        field_width : float
            Width of the field
        monitor_location : np.ndarray
            Monitoring room coordinates
        vegetable_pos : List
            List of vegetable positions
        """
        vegetable_x = [pos[0] for pos in vegetable_pos]
        vegetable_y = [pos[1] for pos in vegetable_pos]

        # Plot the Field
        plt.figure(figsize=(10, 8))
        plt.imshow(field_map.T, extent=[0, field_length, 0, field_width], 
                  origin='lower', 
                  cmap=plt.cm.colors.ListedColormap([[0.7, 0.7, 0.7], [0.2, 0.5, 0.9]]))
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Field Layout: Raised Beds and Furrows (With Vegetables)')
        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Furrow', 'Bed'])
        
        # Plot vegetables
        if vegetable_x and vegetable_y:
            plt.plot(vegetable_x, vegetable_y, 'g.', markersize=12)
        
        # Plot Monitoring Room
        plt.plot(monitor_location[0], monitor_location[1], 'r*', 
                markersize=12, linewidth=2)
        plt.text(monitor_location[0] + 2, monitor_location[1], 
                'Monitoring Room', color='red', fontsize=10)
        
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# Convenience function to maintain backward compatibility
def plot_field(field_map: np.ndarray, grid_size: float, field_length: float, 
               field_width: float, monitor_location: np.ndarray, 
               vegetable_pos: List) -> None:
    """
    Convenience function for plotting field layouts.
    Maintains backward compatibility with existing code.
    """
    FieldVisualizer.plot_field_layout(
        field_map, grid_size, field_length, field_width, 
        monitor_location, vegetable_pos
    )
