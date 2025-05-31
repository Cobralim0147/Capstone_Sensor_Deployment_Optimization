"""
Vegetation Sensor Placement Problem for Multi-Objective Optimization

This module defines a multi-objective optimization problem for placing sensors
in agricultural fields to maximize coverage while minimizing cost and redundancy.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from pymoo.core.problem import Problem
except ImportError as e:
    raise ImportError(f"Required package 'pymoo' not found: {e}")


@dataclass
class SensorMetrics:
    """Container for sensor placement metrics."""
    n_sensors: int
    sensor_positions: np.ndarray
    count_rate: float
    coverage_rate: float
    over_coverage_rate: float
    placement_rate: float
    connectivity_rate: float


class CoverageCalculator:
    """Utility class for coverage calculations."""
    
    @staticmethod
    def calculate_coverage_rate(sensors: np.ndarray, plants: np.ndarray, sensor_range: float) -> float:
        """Calculate the percentage of plants covered by at least one sensor."""
        if sensors.size == 0 or plants.size == 0:
            return 0.0
        
        try:
            # Calculate distances from each sensor to each plant
            distances = np.linalg.norm(
                sensors[:, None, :] - plants[None, :, :],
                axis=2
            )
            
            # Check which plants are covered by at least one sensor
            covered_plants = (distances <= sensor_range).any(axis=0)
            coverage_rate = 100.0 * covered_plants.sum() / plants.shape[0]
            
            return float(coverage_rate)
            
        except Exception as e:
            logging.error(f"Error calculating coverage rate: {e}")
            return 0.0
    
    @staticmethod
    def calculate_over_coverage_rate(sensors: np.ndarray, plants: np.ndarray, sensor_range: float) -> float:
        """Calculate the percentage of plants covered by multiple sensors."""
        if sensors.size == 0 or plants.size == 0:
            return 0.0
        
        try:
            # Calculate distances from each sensor to each plant
            distances = np.linalg.norm(
                sensors[:, None, :] - plants[None, :, :],
                axis=2
            )
            
            # Count how many sensors cover each plant
            coverage_count = (distances <= sensor_range).sum(axis=0)
            over_covered_plants = (coverage_count > 1).sum()
            over_coverage_rate = 100.0 * over_covered_plants / plants.shape[0]
            
            return float(over_coverage_rate)
            
        except Exception as e:
            logging.error(f"Error calculating over-coverage rate: {e}")
            return 0.0
    
    @staticmethod
    def calculate_connectivity_rate(sensors: np.ndarray, comm_range: float) -> float:
        """Calculate the percentage of sensors that are connected to the network."""
        if sensors.size == 0:
            return 0.0
        if sensors.shape[0] == 1:
            return 100.0
        
        try:
            # Calculate distance matrix between sensors
            distances = np.linalg.norm(
                sensors[:, None, :] - sensors[None, :, :],
                axis=2
            )
            
            # Create adjacency matrix (excluding self-connections)
            adjacency = (distances <= comm_range) & (distances > 0)
            
            # Count sensors with at least one connection
            connected_sensors = (adjacency.sum(axis=1) > 0).sum()
            connectivity_rate = 100.0 * connected_sensors / sensors.shape[0]
            
            return float(connectivity_rate)
            
        except Exception as e:
            logging.error(f"Error calculating connectivity rate: {e}")
            return 0.0


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


class VegSensorProblem(Problem):
    """
    Multi-objective optimization problem for sensor placement in agricultural fields.
    
    This class defines a multi-objective optimization problem that aims to:
    1. Minimize sensor count (cost)
    2. Maximize plant coverage
    3. Minimize over-coverage (redundancy)
    4. Maximize valid placements
    5. Maximize network connectivity
    """
    
    def __init__(self,
                 veg_x: np.ndarray,
                 veg_y: np.ndarray,
                 bed_coords: np.ndarray,
                 sensor_range: float,
                 max_sensors: int,
                 comm_range: float,
                 grid_spacing: float):
        """
        Initialize the sensor placement optimization problem.
        
        Parameters:
        -----------
        veg_x, veg_y : np.ndarray
            1D arrays of coordinates for every vegetable plant
        bed_coords : np.ndarray
            (N_beds, 4) array of [x_min, y_min, x_max, y_max] for each raised bed
        sensor_range : float
            Maximum sensing range of sensors (must be positive)
        max_sensors : int
            Maximum number of sensors allowed (must be positive)
        comm_range : float
            Communication range between sensors (must be positive)
        grid_spacing : float
            Spacing between possible sensor positions (must be positive)
        """
        
        # Validate and store parameters
        self._validate_parameters(veg_x, veg_y, bed_coords, sensor_range, 
                                max_sensors, comm_range, grid_spacing)
        
        # Store plant positions
        self.veg_pts = self._create_plant_array(veg_x, veg_y)
        self.bed_coords = bed_coords
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.max_sensors = max_sensors
        self.grid_spacing = grid_spacing
        
        # Initialize utility classes
        self.position_validator = PositionValidator(bed_coords)
        
        # Build allowed positions for sensor placement
        self.possible_positions = self.position_validator.build_allowed_positions(grid_spacing)
        
        # Problem dimensions
        n_var = len(self.possible_positions)
        n_obj = 5  # count, coverage, over-coverage, placement, connectivity
        n_constr = 1  # max sensor constraint
        
        # Binary decision variables (0 or 1 for each possible position)
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            type_var=int,
            n_constr=n_constr
        )
        
        print(f"Problem initialized: {n_var} variables, {n_obj} objectives, "
                        f"{len(self.possible_positions)} possible positions")
    
    def _validate_parameters(self, veg_x: np.ndarray, veg_y: np.ndarray, 
                           bed_coords: np.ndarray, sensor_range: float,
                           max_sensors: int, comm_range: float, grid_spacing: float) -> None:
        """Validate all input parameters."""
        # Validate vegetable coordinates
        if not isinstance(veg_x, np.ndarray) or not isinstance(veg_y, np.ndarray):
            raise TypeError("Vegetation coordinates must be numpy arrays")
        
        if veg_x.shape != veg_y.shape:
            raise ValueError("veg_x and veg_y must have the same shape")
        
        if veg_x.ndim != 1:
            raise ValueError("Vegetation coordinates must be 1D arrays")
        
        # Validate bed coordinates
        if not isinstance(bed_coords, np.ndarray):
            raise TypeError("Bed coordinates must be a numpy array")
        
        # Validate numeric parameters
        if sensor_range <= 0:
            raise ValueError("Sensor range must be positive")
        
        if max_sensors <= 0:
            raise ValueError("Maximum sensors must be positive")
        
        if comm_range <= 0:
            raise ValueError("Communication range must be positive")
        
        if grid_spacing <= 0:
            raise ValueError("Grid spacing must be positive")
    
    def _create_plant_array(self, veg_x: np.ndarray, veg_y: np.ndarray) -> np.ndarray:
        """Create a 2D array of plant positions."""
        if veg_x.size == 0 or veg_y.size == 0:
            print("No vegetation positions provided")
            return np.empty((0, 2))
        
        return np.column_stack([veg_x, veg_y])
    
    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        """
        Evaluate the multi-objective function for a population of solutions.
        
        Objectives (all to be minimized):
        1. Sensor count rate (percentage of max sensors used)
        2. Negative coverage rate (to maximize coverage)
        3. Over-coverage rate (minimize redundant coverage)  
        4. Negative placement rate (to maximize valid placements)
        5. Negative connectivity rate (to maximize network connectivity)
        
        Parameters:
        -----------
        X : np.ndarray
            Population of solutions, shape (pop_size, n_var)
        out : dict
            Output dictionary to store objectives and constraints
        """
        try:
            pop_size = X.shape[0]
            F = np.zeros((pop_size, self.n_obj))
            G = np.zeros((pop_size, 1))  # Constraints
            
            for i, chromosome in enumerate(X):
                # Evaluate single solution
                objectives, constraint = self._evaluate_single_solution(chromosome)
                F[i] = objectives
                G[i, 0] = constraint
            
            # Store results
            out["F"] = F
            out["G"] = G
            
        except Exception as e:
            print(f"Error in population evaluation: {e}")
            # Return worst-case objectives for safety
            out["F"] = np.full((X.shape[0], self.n_obj), 1e6)
            out["G"] = np.full((X.shape[0], 1), 1e6)
    
    def _evaluate_single_solution(self, chromosome: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate a single solution chromosome.
        
        Returns:
        --------
        objectives : np.ndarray
            Array of objective values
        constraint : float
            Constraint violation value
        """
        try:
            # Get selected sensor indices
            selected_indices = np.where(chromosome == 1)[0]
            n_sensors = len(selected_indices)
            
            # Constraint: number of sensors <= max_sensors
            constraint = n_sensors - self.max_sensors
            
            # Handle case with no sensors
            if n_sensors == 0:
                return np.array([0.0, 0.0, 0.0, -100.0, 0.0]), constraint
            
            # Get actual sensor positions
            selected_sensors = self.possible_positions[selected_indices]
            
            # Calculate objectives
            objectives = self._calculate_objectives(selected_sensors, n_sensors)
            
            return objectives, constraint
            
        except Exception as e:
            print(f"Error evaluating single solution: {e}")
            # Return worst-case scenario
            return np.array([100.0, 0.0, 100.0, -0.0, 0.0]), float(self.max_sensors)
    
    def _calculate_objectives(self, sensors: np.ndarray, n_sensors: int) -> np.ndarray:
        """Calculate all objective values for given sensor placement."""
        try:
            # Objective 1: Sensor count rate (minimize usage)
            count_rate = 100.0 * n_sensors / self.max_sensors
            
            # Objective 2: Coverage rate (maximize plants covered)
            coverage_rate = CoverageCalculator.calculate_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )
            
            # Objective 3: Over-coverage rate (minimize redundant coverage)
            over_coverage_rate = CoverageCalculator.calculate_over_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )
            
            # Objective 4: Placement rate (all selected positions are valid by construction)
            placement_rate = 100.0
            
            # Objective 5: Connectivity rate (maximize network connectivity)
            connectivity_rate = CoverageCalculator.calculate_connectivity_rate(
                sensors, self.comm_range
            )
            
            # Return objectives (all to be minimized)
            return np.array([
                count_rate,
                -coverage_rate,      # Negative to minimize (maximize coverage)
                over_coverage_rate,
                -placement_rate,     # Negative to minimize (maximize placement)
                -connectivity_rate   # Negative to minimize (maximize connectivity)
            ])
            
        except Exception as e:
            print(f"Error calculating objectives: {e}")
            return np.array([100.0, 0.0, 100.0, -0.0, 0.0])
    
    def get_sensor_positions(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Extract actual sensor positions from a chromosome.
        
        Parameters:
        -----------
        chromosome : np.ndarray
            Binary array indicating which positions are selected
            
        Returns:
        --------
        np.ndarray
            Array of selected sensor positions, shape (n_selected, 2)
        """
        try:
            if chromosome.size != len(self.possible_positions):
                raise ValueError("Chromosome size does not match number of possible positions")
            
            selected_indices = chromosome.astype(bool)
            return self.possible_positions[selected_indices].copy()
            
        except Exception as e:
            print(f"Error extracting sensor positions: {e}")
            return np.empty((0, 2))
    
    def evaluate_solution(self, chromosome: np.ndarray) -> SensorMetrics:
        """
        Evaluate a single solution and return detailed metrics.
        
        Parameters:
        -----------
        chromosome : np.ndarray
            Binary array indicating which positions are selected
            
        Returns:
        --------
        SensorMetrics
            Detailed metrics for the solution
        """
        try:
            sensors = self.get_sensor_positions(chromosome)
            n_sensors = sensors.shape[0]
            
            # Calculate all metrics
            count_rate = 100.0 * n_sensors / self.max_sensors if self.max_sensors > 0 else 0.0
            coverage_rate = CoverageCalculator.calculate_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )
            over_coverage_rate = CoverageCalculator.calculate_over_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )
            connectivity_rate = CoverageCalculator.calculate_connectivity_rate(
                sensors, self.comm_range
            )
            
            return SensorMetrics(
                n_sensors=n_sensors,
                sensor_positions=sensors,
                count_rate=count_rate,
                coverage_rate=coverage_rate,
                over_coverage_rate=over_coverage_rate,
                placement_rate=100.0,  # All positions are valid by construction
                connectivity_rate=connectivity_rate
            )
            
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return SensorMetrics(
                n_sensors=0,
                sensor_positions=np.empty((0, 2)),
                count_rate=0.0,
                coverage_rate=0.0,
                over_coverage_rate=0.0,
                placement_rate=0.0,
                connectivity_rate=0.0
            )
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get summary information about the problem setup."""
        return {
            'n_variables': self.n_var,
            'n_objectives': self.n_obj,
            'n_constraints': self.n_constr,
            'possible_positions': len(self.possible_positions),
            'n_plants': self.veg_pts.shape[0],
            'n_beds': self.bed_coords.shape[0],
            'sensor_range': self.sensor_range,
            'comm_range': self.comm_range,
            'max_sensors': self.max_sensors,
            'grid_spacing': self.grid_spacing
        }
    
    def validate_solution(self, chromosome: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate a solution chromosome.
        
        Returns:
        --------
        is_valid : bool
            Whether the solution is valid
        errors : List[str]
            List of validation error messages
        """
        errors = []
        
        try:
            # Check chromosome shape
            if chromosome.shape[0] != self.n_var:
                errors.append(f"Chromosome length {chromosome.shape[0]} != {self.n_var}")
            
            # Check binary values
            if not np.all(np.isin(chromosome, [0, 1])):
                errors.append("Chromosome must contain only 0s and 1s")
            
            # Check sensor count constraint
            n_sensors = np.sum(chromosome)
            if n_sensors > self.max_sensors:
                errors.append(f"Too many sensors: {n_sensors} > {self.max_sensors}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    
    # Sample vegetable positions
    veg_x = np.random.uniform(0, 20, 100)
    veg_y = np.random.uniform(0, 20, 100)
    
    # Sample bed coordinates
    bed_coords = np.array([
        [0, 0, 10, 5],
        [0, 7, 10, 12],
        [12, 0, 22, 5],
        [12, 7, 22, 12]
    ])
    
    try:
        # Create problem instance
        problem = VegSensorProblem(
            veg_x=veg_x,
            veg_y=veg_y,
            bed_coords=bed_coords,
            sensor_range=3.0,
            max_sensors=20,
            comm_range=8.0,
            grid_spacing=1.0
        )
        
        print("Problem created successfully!")
        print("Problem info:", problem.get_problem_info())
        
        # Test with a random solution
        test_chromosome = np.random.choice([0, 1], size=problem.n_var, p=[0.9, 0.1])
        
        # Validate solution
        is_valid, errors = problem.validate_solution(test_chromosome)
        print(f"Solution valid: {is_valid}")
        if errors:
            print("Errors:", errors)
        
        # Evaluate solution
        metrics = problem.evaluate_solution(test_chromosome)
        print(f"Solution metrics: {metrics}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()