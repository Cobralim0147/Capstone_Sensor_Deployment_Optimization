"""
L-Shaped Field Environment Generation Module

This module provides functionality for generating L-shaped agricultural field environments
with clustered beds and asymmetric layout.
"""

import numpy as np
import random
from typing import Tuple, List
from .base_env import BaseFieldEnvironmentGenerator

class LShapedFieldGenerator(BaseFieldEnvironmentGenerator):
    """
    Generator for L-shaped field environments with clustered beds.
    """
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """
        Generate L-shaped field with clustered beds.
        """
        # Monitoring room in corner
        monitor_location = self.monitor_location


        # Create grid
        field_map, x_cells, y_cells = self._create_grid(field_length, field_width, grid_size)
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            furrow_width, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Create L-shaped field by masking part of the field
        field_map[x_cells//2:, y_cells//2:] = -1  # -1 = blocked area
        
        bed_coordinates = []
        vegetable_pos = []
        
        # Generate beds in clusters
        clusters = [
            (grid_params['pad_cells'] + 10, grid_params['pad_cells'] + 10),
            (grid_params['pad_cells'] + 10, y_cells//2 - 30),
            (x_cells//2 - 40, grid_params['pad_cells'] + 10),
        ]
        
        for cluster_x, cluster_y in clusters:
            self._generate_cluster(
                field_map, cluster_x, cluster_y, grid_params, grid_size,
                x_cells, y_cells, bed_coordinates, vegetable_pos
            )
        
        # Set blocked areas back to 0 for visualization
        field_map[field_map == -1] = 0
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
    def _generate_cluster(self, field_map: np.ndarray, cluster_x: int, cluster_y: int,
                         grid_params: dict, grid_size: float, x_cells: int, y_cells: int,
                         bed_coordinates: List, vegetable_pos: List):
        
        density_factor = self.density_factor

        """Generate a 3x2 cluster of beds."""
        for i in range(10):
            for j in range(2):
                x_pos = cluster_x + i * (grid_params['bed_cells'] + grid_params['furrow_cells'])
                y_pos = cluster_y + j * (grid_params['bed_length_cells'] + grid_params['furrow_cells'])
                
                # Check if bed fits in field
                if (x_pos + grid_params['bed_cells'] < x_cells and 
                    y_pos + grid_params['bed_length_cells'] < y_cells and
                    field_map[x_pos, y_pos] != -1):
                    
                    # Create bed
                    field_map[x_pos:x_pos + grid_params['bed_cells'], 
                             y_pos:y_pos + grid_params['bed_length_cells']] = 1
                    
                    # Store coordinates
                    bed_coords = self._calculate_bed_coordinates(
                        x_pos, y_pos, grid_params['bed_cells'], 
                        grid_params['bed_length_cells'], grid_size
                    )
                    bed_coordinates.append(bed_coords)
                    
                    # Add vegetables with random density
                    self._add_vegetables_to_bed(
                        x_pos, y_pos, grid_params['bed_cells'], 
                        grid_params['bed_length_cells'], grid_params, 
                        grid_size, vegetable_pos, density_factor
                    )


# Convenience function
def environment_generator_l_shaped(field_length: float, field_width: float, 
                                  bed_width: float, bed_length: float, 
                                  furrow_width: float, grid_size: float, 
                                  dot_spacing: float) -> Tuple:
    """L-shaped field with clustered beds."""
    generator = LShapedFieldGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )