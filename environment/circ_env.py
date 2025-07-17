"""
Improved Circular Field Generator with Concentric Bed Rings
"""

import numpy as np
from typing import Tuple, List
from .base_env import BaseFieldEnvironmentGenerator

class CircularFieldGenerator(BaseFieldEnvironmentGenerator):
    """
    Generator for circular field environments with multiple concentric bed rings.
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

        center_x, center_y = x_cells // 2, y_cells // 2
        radius = min(x_cells, y_cells) // 2 - 10

        bed_coordinates = []
        vegetable_pos = []

        num_rings = 3  # Number of concentric rings
        ring_spacing = grid_params['bed_length_cells'] + grid_params['furrow_cells']

        for ring in range(1, num_rings + 1):
            ring_radius = ring * ring_spacing
            if ring_radius + grid_params['bed_length_cells'] // 2 >= radius:
                break  # Prevent going outside circle
            
            beds_in_ring = max(10, ring * 6)  # More beds in outer rings
            angle_step = 2 * np.pi / beds_in_ring

            for i in range(beds_in_ring):
                angle = i * angle_step
                bed_center_x = int(center_x + ring_radius * np.cos(angle))
                bed_center_y = int(center_y + ring_radius * np.sin(angle))

                x_start = bed_center_x - grid_params['bed_cells'] // 2
                y_start = bed_center_y - grid_params['bed_length_cells'] // 2
                x_end = x_start + grid_params['bed_cells']
                y_end = y_start + grid_params['bed_length_cells']

                if (x_start > 0 and y_start > 0 and x_end < x_cells and y_end < y_cells):
                    # Check all corners are inside the circular boundary
                    if self._is_inside_circle(x_start, y_start, x_end, y_end, center_x, center_y, radius):
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

        return (field_map, grid_size, field_length, field_width,
                monitor_location, vegetable_pos, np.array(bed_coordinates))

    def _is_inside_circle(self, x_start: int, y_start: int, x_end: int, y_end: int,
                          center_x: int, center_y: int, radius: int) -> bool:
        """Check if all corners of a bed are within the circular field."""
        corners = [
            (x_start, y_start), (x_end, y_start),
            (x_start, y_end), (x_end, y_end)
        ]
        for x, y in corners:
            if np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > radius - 2:
                return False
        return True


# Convenience function
def environment_generator_circular(field_length: float, field_width: float, 
                                   bed_width: float, bed_length: float, 
                                   furrow_width: float, grid_size: float, 
                                   dot_spacing: float) -> Tuple:
    generator = CircularFieldGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )
