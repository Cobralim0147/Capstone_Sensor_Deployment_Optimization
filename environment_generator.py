"""
Field Environment Generation Module

This module provides functionality for generating agricultural field environments
with raised beds, furrows, and vegetable placement.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class FieldEnvironmentGenerator:
    """
    Class for generating agricultural field environments with beds and vegetation.
    """
    
    def __init__(self):
        """Initialize the field generator."""
        pass
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """
        Generate a field environment with raised beds and furrows, and place vegetables.
        
        Parameters:
        -----------
        field_length : float
            Length of the field in meters
        field_width : float
            Width of the field in meters
        bed_width : float
            Width of each raised bed in meters
        bed_length : float
            Length of each raised bed in meters
        furrow_width : float
            Width of furrows between beds in meters
        grid_size : float
            Size of each grid cell in meters
        dot_spacing : float
            Spacing between vegetable plants in meters
        
        Returns:
        --------
        tuple containing:
            field_map : np.ndarray
                2D array representing the field (0=furrow, 1=bed)
            grid_size : float
                Grid cell size
            field_length : float
                Field length
            field_width : float
                Field width
            monitor_location : np.ndarray
                Coordinates of monitoring room
            vegetable_pos : list
                List of [x, y] coordinates for vegetable plants
            bed_coordinates : np.ndarray
                Array of bed boundaries [x_min, y_min, x_max, y_max]
        """
        
        # Padding (1 furrow width around field)
        padding = furrow_width
        
        # Monitoring room location
        monitor_location = np.array([150, 50])
        
        # Create grid
        x_cells = int(round(field_length / grid_size))
        y_cells = int(round(field_width / grid_size))
        field_map = np.zeros((x_cells, y_cells))  # 0 = furrow, 1 = raised bed
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            padding, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Calculate field boundaries
        boundaries = self._calculate_field_boundaries(
            x_cells, y_cells, grid_params['pad_cells']
        )
        
        # Generate beds and get their coordinates
        bed_coordinates = self._generate_beds(
            field_map, boundaries, grid_params, grid_size
        )
        
        # Generate vegetable positions
        vegetable_pos = self._generate_vegetable_positions(
            boundaries, grid_params, grid_size
        )
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
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
    
    def _generate_beds(self, field_map: np.ndarray, boundaries: dict, 
                      grid_params: dict, grid_size: float) -> List:
        """Generate raised beds and return their coordinates."""
        bed_coordinates = []
        
        x_pointer = boundaries['x_start']
        while x_pointer + grid_params['bed_cells'] <= boundaries['x_end']:
            y_pointer = boundaries['y_start']
            while y_pointer + grid_params['bed_length_cells'] <= boundaries['y_end']:
                # Set this area as a raised bed (1)
                field_map[x_pointer:x_pointer + grid_params['bed_cells'], 
                         y_pointer:y_pointer + grid_params['bed_length_cells']] = 1
                
                # Store bed coordinates in real-world units
                bed_coords = self._calculate_bed_coordinates(
                    x_pointer, y_pointer, grid_params, grid_size
                )
                bed_coordinates.append(bed_coords)
                
                y_pointer += grid_params['bed_length_cells'] + grid_params['furrow_cells']
            x_pointer += grid_params['bed_cells'] + grid_params['furrow_cells']
        
        return bed_coordinates
    
    def _calculate_bed_coordinates(self, x_pointer: int, y_pointer: int, 
                                  grid_params: dict, grid_size: float) -> List:
        """Calculate real-world coordinates for a bed."""
        bed_x_min = x_pointer * grid_size
        bed_y_min = y_pointer * grid_size
        bed_x_max = (x_pointer + grid_params['bed_cells']) * grid_size
        bed_y_max = (y_pointer + grid_params['bed_length_cells']) * grid_size
        return [bed_x_min, bed_y_min, bed_x_max, bed_y_max]
    
    def _generate_vegetable_positions(self, boundaries: dict, 
                                     grid_params: dict, grid_size: float) -> List:
        """Generate vegetable coordinates inside the beds."""
        vegetable_pos = []
        
        x_pointer = boundaries['x_start']
        while x_pointer + grid_params['bed_cells'] <= boundaries['x_end']:
            # Center of the bed in grid coordinates
            x_center = x_pointer + grid_params['bed_cells'] / 2
            
            y_pointer = boundaries['y_start']
            while y_pointer + grid_params['bed_length_cells'] <= boundaries['y_end']:
                # Place vegetables along the length of the bed
                for y in range(y_pointer + 1, 
                             y_pointer + grid_params['bed_length_cells'], 
                             grid_params['dot_spacing_cells']):
                    x_pos = x_center * grid_size
                    y_pos = y * grid_size
                    vegetable_pos.append([x_pos, y_pos])
                
                y_pointer += grid_params['bed_length_cells'] + grid_params['furrow_cells']
            x_pointer += grid_params['bed_cells'] + grid_params['furrow_cells']
        
        return vegetable_pos


def plot_field(field_map: np.ndarray, grid_size: float, field_length: float, 
               field_width: float, monitor_location: np.ndarray, 
               vegetable_pos: List) -> None:
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
def environment_generator(field_length: float, field_width: float, 
                         bed_width: float, bed_length: float, 
                         furrow_width: float, grid_size: float, 
                         dot_spacing: float) -> Tuple:
    """
    Convenience function for generating field environments.
    Maintains backward compatibility with existing code.
    """
    generator = FieldEnvironmentGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )


# Re-export for backward compatibility
__all__ = ['environment_generator', 'plot_field', 'FieldEnvironmentGenerator']