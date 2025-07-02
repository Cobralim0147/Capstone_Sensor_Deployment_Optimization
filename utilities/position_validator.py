import numpy as np
import os
import sys

class PositionValidator:
    """Utility class for validating sensor positions."""
    
    def __init__(self, bed_coords: np.ndarray):
        self.bed_coords = self._validate_bed_coords(bed_coords)
    
    def _validate_bed_coords(self, bed_coords: np.ndarray) -> np.ndarray:
        """Validate and clean bed coordinates."""
        if bed_coords.size == 0:
            raise ValueError("Bed coordinates cannot be empty")
        
        if bed_coords.ndim != 2 or bed_coords.shape[1] != 4:
            raise ValueError("Bed coordinates must be a 2D array with 4 columns [x_min, y_min, x_max, y_max]")
        
        # Ensure x_min <= x_max and y_min <= y_max
        for i in range(bed_coords.shape[0]):
            x_min, y_min, x_max, y_max = bed_coords[i]
            if x_min > x_max:
                bed_coords[i, 0], bed_coords[i, 2] = x_max, x_min
            if y_min > y_max:
                bed_coords[i, 1], bed_coords[i, 3] = y_max, y_min
        
        return bed_coords
    
    def point_in_any_bed(self, x: float, y: float) -> bool:
        """Check if point (x,y) is inside any bed."""
        try:
            for i in range(self.bed_coords.shape[0]):
                x_min, y_min, x_max, y_max = self.bed_coords[i]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
            return False
        except Exception as e:
            print(f"Error checking point in bed: {e}")
            return False
    
    def build_allowed_positions(self, grid_spacing: float) -> np.ndarray:
        """
        Build a list of (x,y) coordinates where sensors can be placed.
        Only positions within raised beds are allowed.
        """
        try:
            # Find the bounds of all beds
            min_x = np.min(self.bed_coords[:, 0])
            max_x = np.max(self.bed_coords[:, 2])
            min_y = np.min(self.bed_coords[:, 1])
            max_y = np.max(self.bed_coords[:, 3])
            
            # Validate grid spacing
            if grid_spacing <= 0:
                raise ValueError("Grid spacing must be positive")
            
            positions = []
            
            # Create a grid of potential positions
            x_positions = np.arange(min_x, max_x + grid_spacing, grid_spacing)
            y_positions = np.arange(min_y, max_y + grid_spacing, grid_spacing)
            
            for x in x_positions:
                for y in y_positions:
                    if self.point_in_any_bed(x, y):
                        positions.append((x, y))
            
            if not positions:
                raise ValueError("No valid positions found within bed boundaries")
            
            return np.array(positions)
            
        except Exception as e:
            print(f"Error building allowed positions: {e}")
            raise