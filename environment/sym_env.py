"""
Symmetric Field Environment Generation Module

This module provides functionality for generating symmetric agricultural field environments
with regular raised beds and furrows.
"""

import numpy as np
from typing import Tuple, List
from .base_env import BaseFieldEnvironmentGenerator

class SymmetricFieldGenerator(BaseFieldEnvironmentGenerator):
    """
    Generator for symmetric field environments with regular bed patterns.
    """
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """
        Generate a symmetric field environment with regular bed pattern.
        """
        # Padding (1 furrow width around field)
        padding = furrow_width
        
        # Monitoring room location (standard position)
        monitor_location = self.monitor_location
        density_factor = self.density_factor
        
        # Create grid
        field_map, x_cells, y_cells = self._create_grid(field_length, field_width, grid_size)
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            padding, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Calculate field boundaries
        boundaries = self._calculate_field_boundaries(
            x_cells, y_cells, grid_params['pad_cells']
        )
        
        # Generate beds and get their coordinates
        bed_coordinates = self._generate_symmetric_beds(
            field_map, boundaries, grid_params, grid_size
        )
        
        # Generate vegetable positions
        vegetable_pos = self._generate_symmetric_vegetables(
            boundaries, grid_params, grid_size
        )
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
    def _generate_symmetric_beds(self, field_map: np.ndarray, boundaries: dict, 
                                grid_params: dict, grid_size: float) -> List:
        """Generate symmetric beds in regular pattern."""
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
                    x_pointer, y_pointer, grid_params['bed_cells'], 
                    grid_params['bed_length_cells'], grid_size
                )
                bed_coordinates.append(bed_coords)
                
                y_pointer += grid_params['bed_length_cells'] + grid_params['furrow_cells']
            x_pointer += grid_params['bed_cells'] + grid_params['furrow_cells']
        
        return bed_coordinates
    
    def _generate_symmetric_vegetables(self, boundaries: dict, 
                                     grid_params: dict, grid_size: float) -> List:
        """Generate vegetables in symmetric pattern."""
        vegetable_pos = [] 
        
        x_pointer = boundaries['x_start']
        while x_pointer + grid_params['bed_cells'] <= boundaries['x_end']:
            y_pointer = boundaries['y_start']
            while y_pointer + grid_params['bed_length_cells'] <= boundaries['y_end']:
                # Add vegetables to this bed
                self._add_vegetables_to_bed(
                    x_pointer, y_pointer, grid_params['bed_cells'], 
                    grid_params['bed_length_cells'], grid_params, 
                    grid_size, vegetable_pos, density_factor
                )
                
                y_pointer += grid_params['bed_length_cells'] + grid_params['furrow_cells']
            x_pointer += grid_params['bed_cells'] + grid_params['furrow_cells']
        
        return vegetable_pos


# Convenience function for backward compatibility
def environment_generator(field_length: float, field_width: float, 
                         bed_width: float, bed_length: float, 
                         furrow_width: float, grid_size: float, 
                         dot_spacing: float) -> Tuple:
    """
    Convenience function for generating symmetric field environments.
    """
    generator = SymmetricFieldGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )