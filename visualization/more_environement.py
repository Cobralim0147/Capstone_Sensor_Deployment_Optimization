"""
Enhanced Field Environment Generation Module

This module provides functionality for generating agricultural field environments
with raised beds, furrows, and vegetable placement. Now includes 4 different
field layouts for testing deployment algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random

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


class AsymmetricFieldGenerator1(FieldEnvironmentGenerator):
    """
    L-shaped field with clustered beds and offset monitoring station.
    """
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """Generate L-shaped field with clustered beds."""
        
        # Monitoring room in corner
        monitor_location = np.array([20, 20])
        
        # Create grid
        x_cells = int(round(field_length / grid_size))
        y_cells = int(round(field_width / grid_size))
        field_map = np.zeros((x_cells, y_cells))
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            furrow_width, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Create L-shaped field by masking part of the field
        # Block out upper-right quadrant
        field_map[x_cells//2:, y_cells//2:] = -1  # -1 = blocked area
        
        bed_coordinates = []
        vegetable_pos = []
        
        # Generate beds in clusters
        clusters = [
            (grid_params['pad_cells'] + 10, grid_params['pad_cells'] + 10),  # Bottom-left cluster
            (grid_params['pad_cells'] + 10, y_cells//2 - 30),  # Top-left cluster
            (x_cells//2 - 40, grid_params['pad_cells'] + 10),  # Bottom-right cluster
        ]
        
        for cluster_x, cluster_y in clusters:
            # Create 3x2 cluster of beds
            for i in range(3):
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
                            x_pos, y_pos, grid_params, grid_size
                        )
                        bed_coordinates.append(bed_coords)
                        
                        # Add vegetables with random density
                        self._add_vegetables_to_bed(
                            x_pos, y_pos, grid_params, grid_size, vegetable_pos, 
                            density_factor=random.uniform(0.5, 1.0)
                        )
        
        # Set blocked areas back to 0 for visualization
        field_map[field_map == -1] = 0
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
    def _add_vegetables_to_bed(self, x_pos: int, y_pos: int, grid_params: dict, 
                              grid_size: float, vegetable_pos: List, density_factor: float = 1.0):
        """Add vegetables to a bed with variable density."""
        x_center = x_pos + grid_params['bed_cells'] / 2
        
        spacing = max(1, int(grid_params['dot_spacing_cells'] / density_factor))
        
        for y in range(y_pos + 1, y_pos + grid_params['bed_length_cells'], spacing):
            # Add some randomness to vegetable positions
            x_offset = random.uniform(-0.5, 0.5)
            y_offset = random.uniform(-0.5, 0.5)
            
            x_veg = (x_center + x_offset) * grid_size
            y_veg = (y + y_offset) * grid_size
            vegetable_pos.append([x_veg, y_veg])


class AsymmetricFieldGenerator2(FieldEnvironmentGenerator):
    """
    Circular field with radial bed arrangement.
    """
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """Generate circular field with radial beds."""
        
        # Monitoring room at edge
        monitor_location = np.array([field_length * 0.8, field_width * 0.1])
        
        # Create grid
        x_cells = int(round(field_length / grid_size))
        y_cells = int(round(field_width / grid_size))
        field_map = np.zeros((x_cells, y_cells))
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            furrow_width, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Create circular field boundary
        center_x, center_y = x_cells // 2, y_cells // 2
        radius = min(x_cells, y_cells) // 2 - 10
        
        # Mask out areas outside circle
        for x in range(x_cells):
            for y in range(y_cells):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance > radius:
                    field_map[x, y] = -1
        
        bed_coordinates = []
        vegetable_pos = []
        
        # Create radial pattern of beds
        num_rings = 4
        for ring in range(1, num_rings + 1):
            ring_radius = ring * 20
            beds_in_ring = ring * 6  # More beds in outer rings
            
            for bed_idx in range(beds_in_ring):
                angle = (2 * np.pi * bed_idx) / beds_in_ring
                
                # Calculate bed position
                bed_center_x = center_x + int(ring_radius * np.cos(angle))
                bed_center_y = center_y + int(ring_radius * np.sin(angle))
                
                # Adjust bed orientation based on angle
                if abs(np.sin(angle)) > abs(np.cos(angle)):
                    # Vertical orientation
                    bed_width_cells = grid_params['bed_cells']
                    bed_length_cells = grid_params['bed_length_cells']
                else:
                    # Horizontal orientation
                    bed_width_cells = grid_params['bed_length_cells']
                    bed_length_cells = grid_params['bed_cells']
                
                x_start = bed_center_x - bed_width_cells // 2
                y_start = bed_center_y - bed_length_cells // 2
                
                # Check if bed fits and is within circle
                if (x_start > 0 and y_start > 0 and 
                    x_start + bed_width_cells < x_cells and 
                    y_start + bed_length_cells < y_cells):
                    
                    # Check if bed is within circular boundary
                    bed_in_circle = True
                    for x in range(x_start, x_start + bed_width_cells):
                        for y in range(y_start, y_start + bed_length_cells):
                            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            if distance > radius - 5:  # Small margin
                                bed_in_circle = False
                                break
                        if not bed_in_circle:
                            break
                    
                    if bed_in_circle:
                        # Create bed
                        field_map[x_start:x_start + bed_width_cells, 
                                 y_start:y_start + bed_length_cells] = 1
                        
                        # Store coordinates
                        bed_coords = [
                            x_start * grid_size,
                            y_start * grid_size,
                            (x_start + bed_width_cells) * grid_size,
                            (y_start + bed_length_cells) * grid_size
                        ]
                        bed_coordinates.append(bed_coords)
                        
                        # Add vegetables
                        self._add_vegetables_circular(
                            x_start, y_start, bed_width_cells, bed_length_cells,
                            grid_params, grid_size, vegetable_pos
                        )
        
        # Set blocked areas back to 0
        field_map[field_map == -1] = 0
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
    def _add_vegetables_circular(self, x_start: int, y_start: int, 
                                bed_width_cells: int, bed_length_cells: int,
                                grid_params: dict, grid_size: float, vegetable_pos: List):
        """Add vegetables to circular bed arrangement."""
        for x in range(x_start + 1, x_start + bed_width_cells, 
                      grid_params['dot_spacing_cells']):
            for y in range(y_start + 1, y_start + bed_length_cells, 
                          grid_params['dot_spacing_cells']):
                x_veg = x * grid_size
                y_veg = y * grid_size
                vegetable_pos.append([x_veg, y_veg])


class AsymmetricFieldGenerator3(FieldEnvironmentGenerator):
    """
    Irregular field with scattered beds and obstacles.
    """
    
    def generate_environment(self, field_length: float, field_width: float, 
                           bed_width: float, bed_length: float, 
                           furrow_width: float, grid_size: float, 
                           dot_spacing: float) -> Tuple:
        """Generate irregular field with scattered beds and obstacles."""
        
        # Monitoring room in random position
        monitor_location = np.array([field_length * 0.3, field_width * 0.8])
        
        # Create grid
        x_cells = int(round(field_length / grid_size))
        y_cells = int(round(field_width / grid_size))
        field_map = np.zeros((x_cells, y_cells))
        
        # Convert parameters to grid cells
        grid_params = self._convert_to_grid_cells(
            furrow_width, bed_width, bed_length, furrow_width, dot_spacing, grid_size
        )
        
        # Add random obstacles (trees, buildings, etc.)
        obstacles = [
            (50, 80, 20, 30),   # x, y, width, height in grid cells
            (120, 40, 15, 25),
            (80, 150, 25, 20),
            (160, 100, 18, 22),
        ]
        
        for obs_x, obs_y, obs_w, obs_h in obstacles:
            if obs_x + obs_w < x_cells and obs_y + obs_h < y_cells:
                field_map[obs_x:obs_x + obs_w, obs_y:obs_y + obs_h] = -1
        
        bed_coordinates = []
        vegetable_pos = []
        
        # Randomly place beds avoiding obstacles
        random.seed(42)  # For reproducibility
        attempted_beds = 0
        placed_beds = 0
        max_attempts = 200
        
        while attempted_beds < max_attempts and placed_beds < 30:
            # Random position
            x_pos = random.randint(grid_params['pad_cells'], 
                                  x_cells - grid_params['bed_cells'] - grid_params['pad_cells'])
            y_pos = random.randint(grid_params['pad_cells'], 
                                  y_cells - grid_params['bed_length_cells'] - grid_params['pad_cells'])
            
            # Random bed size variation
            bed_width_var = grid_params['bed_cells'] + random.randint(-2, 3)
            bed_length_var = grid_params['bed_length_cells'] + random.randint(-5, 10)
            
            bed_width_var = max(3, bed_width_var)
            bed_length_var = max(5, bed_length_var)
            
            # Check if area is free
            area_free = True
            if (x_pos + bed_width_var < x_cells and 
                y_pos + bed_length_var < y_cells):
                
                for x in range(x_pos - 2, x_pos + bed_width_var + 2):
                    for y in range(y_pos - 2, y_pos + bed_length_var + 2):
                        if (x >= 0 and x < x_cells and y >= 0 and y < y_cells and 
                            field_map[x, y] != 0):
                            area_free = False
                            break
                    if not area_free:
                        break
            else:
                area_free = False
            
            if area_free:
                # Create bed
                field_map[x_pos:x_pos + bed_width_var, 
                         y_pos:y_pos + bed_length_var] = 1
                
                # Store coordinates
                bed_coords = [
                    x_pos * grid_size,
                    y_pos * grid_size,
                    (x_pos + bed_width_var) * grid_size,
                    (y_pos + bed_length_var) * grid_size
                ]
                bed_coordinates.append(bed_coords)
                
                # Add vegetables with irregular spacing
                self._add_vegetables_irregular(
                    x_pos, y_pos, bed_width_var, bed_length_var,
                    grid_params, grid_size, vegetable_pos
                )
                
                placed_beds += 1
            
            attempted_beds += 1
        
        # Set blocked areas back to 0
        field_map[field_map == -1] = 0
        
        return (field_map, grid_size, field_length, field_width, 
                monitor_location, vegetable_pos, np.array(bed_coordinates))
    
    def _add_vegetables_irregular(self, x_pos: int, y_pos: int, 
                                 bed_width: int, bed_length: int,
                                 grid_params: dict, grid_size: float, vegetable_pos: List):
        """Add vegetables with irregular spacing."""
        # Random spacing variation
        for x in range(x_pos + 1, x_pos + bed_width):
            for y in range(y_pos + 1, y_pos + bed_length):
                # Random chance of placing vegetable
                if random.random() < 0.3:  # 30% chance
                    # Add some randomness to position
                    x_offset = random.uniform(-0.3, 0.3)
                    y_offset = random.uniform(-0.3, 0.3)
                    
                    x_veg = (x + x_offset) * grid_size
                    y_veg = (y + y_offset) * grid_size
                    vegetable_pos.append([x_veg, y_veg])


def plot_field(field_map: np.ndarray, grid_size: float, field_length: float, 
               field_width: float, monitor_location: np.ndarray, 
               vegetable_pos: List, title: str = "Field Layout") -> None:
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
    title : str
        Title for the plot
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


# Convenience functions for each field type
def environment_generator(field_length: float, field_width: float, 
                         bed_width: float, bed_length: float, 
                         furrow_width: float, grid_size: float, 
                         dot_spacing: float) -> Tuple:
    """Original symmetric field generator."""
    generator = FieldEnvironmentGenerator()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )


def environment_generator_l_shaped(field_length: float, field_width: float, 
                                  bed_width: float, bed_length: float, 
                                  furrow_width: float, grid_size: float, 
                                  dot_spacing: float) -> Tuple:
    """L-shaped field with clustered beds."""
    generator = AsymmetricFieldGenerator1()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )


def environment_generator_circular(field_length: float, field_width: float, 
                                  bed_width: float, bed_length: float, 
                                  furrow_width: float, grid_size: float, 
                                  dot_spacing: float) -> Tuple:
    """Circular field with radial bed arrangement."""
    generator = AsymmetricFieldGenerator2()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )


def environment_generator_irregular(field_length: float, field_width: float, 
                                   bed_width: float, bed_length: float, 
                                   furrow_width: float, grid_size: float, 
                                   dot_spacing: float) -> Tuple:
    """Irregular field with scattered beds and obstacles."""
    generator = AsymmetricFieldGenerator3()
    return generator.generate_environment(
        field_length, field_width, bed_width, bed_length,
        furrow_width, grid_size, dot_spacing
    )


# # Example usage and testing
# if __name__ == "__main__":
#     # Test parameters
#     field_length = 200
#     field_width = 200
#     bed_width = 8
#     bed_length = 20
#     furrow_width = 4
#     grid_size = 1
#     dot_spacing = 2
    
#     # Generate and plot all four field types
#     generators = [
#         (environment_generator, "Original Symmetric Field"),
#         (environment_generator_l_shaped, "L-Shaped Field with Clustered Beds"),
#         (environment_generator_circular, "Circular Field with Radial Beds"),
#         (environment_generator_irregular, "Irregular Field with Scattered Beds")
#     ]
    
#     for generator_func, title in generators:
#         print(f"\nGenerating {title}...")
        
#         result = generator_func(
#             field_length, field_width, bed_width, bed_length,
#             furrow_width, grid_size, dot_spacing
#         )
        
#         field_map, grid_size, field_length, field_width, monitor_location, vegetable_pos, bed_coordinates = result
        
#         print(f"Generated field with {len(bed_coordinates)} beds and {len(vegetable_pos)} vegetables")
#         print(f"Base station location: {monitor_location}")
        
#         # Plot the field
#         plot_field(field_map, grid_size, field_length, field_width, 
#                   monitor_location, vegetable_pos, title)


# # Re-export for backward compatibility
# __all__ = [
#     'environment_generator', 
#     'environment_generator_l_shaped',
#     'environment_generator_circular', 
#     'environment_generator_irregular',
#     'plot_field', 
#     'FieldEnvironmentGenerator',
#     'AsymmetricFieldGenerator1',
#     'AsymmetricFieldGenerator2', 
#     'AsymmetricFieldGenerator3'
# ]