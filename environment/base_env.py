"""
Base Field Environment Generation Module

This module provides the base functionality for generating agricultural field environments.
All specific field generators inherit from this base class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from abc import ABC, abstractmethod
from configurations.config_file import OptimizationConfig

class BaseFieldEnvironmentGenerator(ABC):
    """
    Abstract base class for field environment generators.
    Contains common functionality that all field types share.
    """
    
    def __init__(self):
        """Initialize the field generator."""
        self.config = OptimizationConfig()
        self.monitor_location = self.config.environment_base_station
        self.density_factor = self.config.environment_density_factor
    
    @abstractmethod
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """
        Abstract method that must be implemented by subclasses.
        Each field type has its own generation logic.
        """
        pass
    
    def _convert_to_grid_cells(self, padding: float, bed_width: float, 
                              bed_length: float, furrow_width: float, 
                              dot_spacing: float, grid_size: float) -> dict:
        """Convert physical measurements to grid cell units."""
        return {
            'pad_cells': int(round(padding / grid_size)),
            'bed_cells': int(round(bed_width / grid_size)),
            'furrow_cells': int(round(furrow_width / grid_size)),
            'bed_length_cells': int(round(bed_length / grid_size)),
            'dot_spacing_cells': int(np.ceil(dot_spacing / grid_size))
        }
    
    def _calculate_field_boundaries(self, x_cells: int, y_cells: int, 
                                   pad_cells: int) -> dict:
        """Calculate usable field boundaries excluding padding."""
        return {
            'x_start': pad_cells + 1,
            'x_end': x_cells - pad_cells,
            'y_start': pad_cells + 1,
            'y_end': y_cells - pad_cells
        }
    
    def _calculate_bed_coordinates(self, x_pointer: int, y_pointer: int, 
                                  bed_width_cells: int, bed_length_cells: int, 
                                  grid_size: float) -> List:
        """Calculate real-world coordinates for a bed."""
        bed_x_min = x_pointer * grid_size
        bed_y_min = y_pointer * grid_size
        bed_x_max = (x_pointer + bed_width_cells) * grid_size
        bed_y_max = (y_pointer + bed_length_cells) * grid_size
        return [bed_x_min, bed_y_min, bed_x_max, bed_y_max]
    
    def _create_grid(self, field_length: float, field_width: float, 
                    grid_size: float) -> Tuple[np.ndarray, int, int]:
        """Create the base grid for the field."""
        x_cells = int(round(field_length / grid_size))
        y_cells = int(round(field_width / grid_size))
        field_map = np.zeros((x_cells, y_cells))  # 0 = furrow, 1 = raised bed
        return field_map, x_cells, y_cells
    
    def _add_vegetables_to_bed(self, x_pointer: int, y_pointer: int, 
                              bed_width_cells: int, bed_length_cells: int,
                              grid_params: dict, grid_size: float, 
                              vegetable_pos: List, density_factor: float):
        """Add vegetables to a bed with specified density."""
        x_center = x_pointer + bed_width_cells / 2
        
        spacing = max(1, int(grid_params['dot_spacing_cells'] / density_factor))
        
        for y in range(y_pointer + 1, y_pointer + bed_length_cells, spacing):
            x_pos = x_center * grid_size
            y_pos = y * grid_size
            vegetable_pos.append([x_pos, y_pos])


def plot_field(field_map: np.ndarray, grid_size: float, field_length: float, 
               field_width: float, monitor_location: np.ndarray, 
               vegetable_pos: List, title: str = "Field Layout") -> None:
    """
    Plot the field layout with beds, furrows, and vegetables.
    This function works for all field types.
    """
    vegetable_x = [pos[0] for pos in vegetable_pos]
    vegetable_y = [pos[1] for pos in vegetable_pos]

    # Plot the Field
    plt.figure(figsize=(12, 10))
    plt.imshow(field_map.T, extent=[0, field_length, 0, field_width], 
              origin='lower', 
              cmap=plt.cm.colors.ListedColormap([[0.7, 0.7, 0.7], [0.2, 0.5, 0.9]]))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Furrow', 'Bed'])
    
    # Plot vegetables
    if vegetable_x and vegetable_y:
        plt.plot(vegetable_x, vegetable_y, 'g.', markersize=8, alpha=0.7)
    
    # Plot Monitoring Room
    plt.plot(monitor_location[0], monitor_location[1], 'r*', 
            markersize=15, linewidth=2)
    plt.text(monitor_location[0] + 5, monitor_location[1] + 5, 
            'Base Station', color='red', fontsize=10, weight='bold')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()