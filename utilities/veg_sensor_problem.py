import numpy as np
from typing import Tuple
import os 
import sys

from utilities.position_validator import PositionValidator
from utilities.coverage_calculator import CoverageCalculator
from configurations.data_structure import SensorMetrics

class VegSensorProblem:
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
        self.n_var = len(self.possible_positions)
        self.n_obj = 5  # count, coverage, over-coverage, placement, connectivity
        self.n_constr = 1  # max sensor constraint
        
        # Binary decision variables (0 or 1 for each possible position)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        
        print(f"Problem initialized: {self.n_var} variables, {self.n_obj} objectives, "
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
        
        # #print the column stack of veg_x and veg_y
        # vege_merged = np.column_stack([veg_x, veg_y])
        # for x in range(len(vege_merged)):
        #     print(vege_merged[x])
        return np.column_stack([veg_x, veg_y])
    
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

            # Penalize infeasible solutions
            if n_sensors > self.max_sensors:
                penalty = 1e6
                # Large penalty for all objectives
                return np.array([penalty, penalty, penalty, penalty, penalty]), constraint

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
    
    
