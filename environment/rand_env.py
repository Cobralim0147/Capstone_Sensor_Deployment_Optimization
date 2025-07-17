"""
Random Field Environment Generation Module
"""

import numpy as np
import random
from typing import Tuple, List
from .base_env import BaseFieldEnvironmentGenerator

class RandomFieldGenerator(BaseFieldEnvironmentGenerator):
    """
    Generator for random field environments with non-patterned bed placements.
    """

    def generate_environment(self, field_length: float, field_width: float, 
                             bed_width: float, bed_length: float, 
                             furrow_width: float, grid_size: float, 
                             dot_spacing: float) -> Tuple:
        monitor_location = self.monitor_location
        density_factor = self.density_factor
        field_map, x_cells, y_cells = self._create_grid(field_length, field_width, grid_size)

        grid_params = self._convert_to_grid_cells(
            furrow_width, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )

        bed_coordinates = []
        vegetable_pos = []
        num_beds = 100
        attempts = 0
        max_attempts = 200

        while len(bed_coordinates) < num_beds and attempts < max_attempts:
            x_start = random.randint(0, x_cells - grid_params['bed_cells'] - 1)
            y_start = random.randint(0, y_cells - grid_params['bed_length_cells'] - 1)
            x_end = x_start + grid_params['bed_cells']
            y_end = y_start + grid_params['bed_length_cells']

            # Check for overlap
            if np.all(field_map[x_start:x_end, y_start:y_end] == 0):
                field_map[x_start:x_end, y_start:y_end] = 1
                bed_coords = self._calculate_bed_coordinates(
                    x_start, y_start, grid_params['bed_cells'], 
                    grid_params['bed_length_cells'], grid_size
                )
                bed_coordinates.append(bed_coords)

                self._add_vegetables_to_bed(
                    x_start, y_start, grid_params['bed_cells'], 
                    grid_params['bed_length_cells'], grid_params, 
                    grid_size, vegetable_pos, density_factor
                )

            attempts += 1

        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))


# Convenience function
def environment_generator_random(field_length: float, field_width: float, 
                                 bed_width: float, bed_length: float, 
                                 furrow_width: float, grid_size: float, 
                                 dot_spacing: float) -> Tuple:
    generator = RandomFieldGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )
