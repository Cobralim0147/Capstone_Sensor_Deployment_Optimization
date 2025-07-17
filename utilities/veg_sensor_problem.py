import numpy as np
from typing import Tuple
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilities.position_validator import PositionValidator
from utilities.coverage_calculator import CoverageCalculator
from configurations.data_structure import SensorMetrics

class VegSensorProblem:
    """
    Defines the multi-objective optimization problem for sensor placement.

    This class encapsulates all the logic and data related to the specific
    problem of placing sensors in an agricultural field. It is responsible for:
    1. Defining the search space (all possible sensor locations).
    2. Evaluating a given solution (a chromosome) and calculating its
       performance across multiple objectives.
    3. Providing utility functions to interpret solutions.

    The primary objectives to be optimized are:
    - Minimize the number of sensors used.
    - Maximize the coverage of vegetable plants.
    - Minimize redundant sensor coverage (over-coverage).
    - Maximize the validity of sensor placements.
    - Maximize the communication connectivity of the sensor network.
    """

    # =====================================================================================
    # 1. Initialization and Setup
    # =====================================================================================

    def __init__(self,
                 veg_x: np.ndarray,
                 veg_y: np.ndarray,
                 bed_coords: np.ndarray,
                 sensor_range: float,
                 max_sensors: int,
                 comm_range: float,
                 grid_spacing: float):
        """
        Initializes the sensor placement optimization problem definition.

        Parameters:
        -----------
        veg_x, veg_y : np.ndarray
            1D arrays of coordinates for every vegetable plant.
        bed_coords : np.ndarray
            (N, 4) array of [x_min, y_min, x_max, y_max] for each bed.
        sensor_range : float
            The sensing radius of a single sensor.
        max_sensors : int
            The maximum number of sensors allowed in a valid solution.
        comm_range : float
            The communication radius between sensors for network connectivity.
        grid_spacing : float
            The spacing used to create a grid of possible sensor positions.
        """
        self._validate_parameters(veg_x, veg_y, bed_coords, sensor_range,
                                max_sensors, comm_range, grid_spacing)

        self.veg_pts = self._create_plant_array(veg_x, veg_y)
        self.bed_coords = bed_coords
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.max_sensors = max_sensors
        self.grid_spacing = grid_spacing
        self.position_validator = PositionValidator(bed_coords)
        self.possible_positions = self.position_validator.build_allowed_positions(grid_spacing)

        self.n_var = len(self.possible_positions)  
        self.n_obj = 5 
        self.n_constr = 1  
        self.xl = np.zeros(self.n_var)  
        self.xu = np.ones(self.n_var)  

        print(f"Problem initialized: {self.n_var} variables, {self.n_obj} objectives, "
              f"{len(self.possible_positions)} possible positions")

    # =====================================================================================
    # 2. Core Evaluation Logic
    # =====================================================================================

    def _evaluate_single_solution(self, chromosome: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluates a single solution (chromosome) for the optimization algorithm.

        This is the primary function called by the optimizer (e.g., SPEA2)
        to get the fitness/objective values for a potential solution. It
        translates a binary chromosome into a set of objective scores and a
        constraint violation value.

        Returns:
        --------
        objectives : np.ndarray
            An array of the five objective values for the algorithm to minimize.
        constraint : float
            The constraint violation value (positive if violated, <= 0 otherwise).
        """
        try:
            selected_indices = np.where(chromosome == 1)[0]
            n_sensors = len(selected_indices)

            constraint = max(0, n_sensors - self.max_sensors)

            if constraint > 0:
                penalty = 1e6  
                return np.full(self.n_obj, penalty), constraint

            if n_sensors == 0:
                return np.array([0.0, 0.0, 0.0, -100.0, 0.0]), constraint

            selected_sensors = self.possible_positions[selected_indices]
            objectives = self._calculate_objectives(selected_sensors, n_sensors)

            return objectives, constraint

        except Exception as e:
            print(f"Error evaluating single solution: {e}")
            return np.array([100.0, 0.0, 100.0, 0.0, 0.0]), float(self.max_sensors)

    def _calculate_objectives(self, sensors: np.ndarray, n_sensors: int) -> np.ndarray:
        """
        Calculates all objective values for a given set of sensor positions.

        Note: The optimization algorithm (SPEA2) is designed to MINIMIZE all
        objective values. Therefore, objectives we want to maximize (like
        coverage) are returned as negative numbers.
        """
        try:
            count_rate = 100.0 * n_sensors / self.max_sensors

            coverage_rate = CoverageCalculator.calculate_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )

            over_coverage_rate = CoverageCalculator.calculate_over_coverage_rate(
                sensors, self.veg_pts, self.sensor_range
            )

            placement_rate = 100.0

            connectivity_rate = CoverageCalculator.calculate_connectivity_rate(
                sensors, self.comm_range
            )

            return np.array([
                count_rate,
                -coverage_rate,      
                over_coverage_rate,
                -placement_rate,     
                -connectivity_rate   
            ])

        except Exception as e:
            print(f"Error calculating objectives: {e}")
            return np.array([100.0, 0.0, 100.0, 0.0, 0.0]) 

    # =====================================================================================
    # 3. Public Interface & Utilities
    # =====================================================================================

    def get_sensor_positions(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Translates a binary chromosome into an array of sensor (x, y) coordinates.
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
        Evaluates a solution and returns a detailed, human-readable metrics object.

        This is a user-facing function, typically used after the optimization is
        complete to analyze the final results. It provides positive, intuitive
        percentages (e.g., coverage is 95%, not -95.0).
        """
        try:
            sensors = self.get_sensor_positions(chromosome)
            n_sensors = sensors.shape[0]

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
                placement_rate=100.0,  
                connectivity_rate=connectivity_rate
            )

        except Exception as e:
            print(f"Error evaluating solution for metrics: {e}")
            return SensorMetrics() 
        
    def create_greedy_heuristic_solution(self) -> np.ndarray:
        """
        Creates a single, high-quality solution using a greedy heuristic.

        This method iteratively adds the best possible sensor to cover the most
        uncovered plants, providing a strong starting point for the optimization.

        Returns:
        --------
        np.ndarray
            A binary chromosome representing the greedily constructed solution.
        """
        print("Generating a high-quality initial solution using a greedy heuristic...")
        
        num_veg_pts = self.veg_pts.shape[0]
        num_possible_pos = self.possible_positions.shape[0]

        # A boolean mask of vegetable points that are not yet covered.
        uncovered_veg_mask = np.ones(num_veg_pts, dtype=bool)
        
        # The chromosome for our new individual. Starts with no sensors.
        solution_chromosome = np.zeros(num_possible_pos, dtype=int)
        
        # Pre-calculate coverage for each possible sensor position to avoid re-computation.
        # This is a big performance win.
        coverage_cache = [
            np.linalg.norm(self.veg_pts - pos, axis=1) <= self.sensor_range
            for pos in self.possible_positions
        ]

        while np.any(uncovered_veg_mask) and np.sum(solution_chromosome) < self.max_sensors:
            best_candidate_idx = -1
            max_newly_covered_count = 0
            
            # Iterate through all possible sensor positions that we haven't used yet.
            for i in range(num_possible_pos):
                if solution_chromosome[i] == 1:
                    continue # This sensor is already in our solution.

                # Find the intersection of this sensor's coverage and the currently uncovered points.
                potential_new_coverage = coverage_cache[i]
                newly_covered_points = potential_new_coverage & uncovered_veg_mask
                num_newly_covered = np.sum(newly_covered_points)

                # If this candidate is the best one we've seen so far, save it.
                if num_newly_covered > max_newly_covered_count:
                    max_newly_covered_count = num_newly_covered
                    best_candidate_idx = i
            
            # If we found a sensor that helps, add it to our solution.
            if best_candidate_idx != -1 and max_newly_covered_count > 0:
                solution_chromosome[best_candidate_idx] = 1
                # Update the mask of uncovered vegetables.
                best_coverage_mask = coverage_cache[best_candidate_idx]
                uncovered_veg_mask[best_coverage_mask] = False # Mark these as covered.
            else:
                # If no sensor can cover any more new points, we're done.
                break
                
        num_sensors_used = np.sum(solution_chromosome)
        coverage_rate = (num_veg_pts - np.sum(uncovered_veg_mask)) / num_veg_pts * 100
        print(f"Greedy solution created with {num_sensors_used} sensors, achieving {coverage_rate:.2f}% coverage.")
        
        return solution_chromosome

    # =====================================================================================
    # 4. Private Helper Methods
    # =====================================================================================

    def _validate_parameters(self, *args) -> None:
        """Validates the input parameters during initialization."""
        (veg_x, veg_y, bed_coords, sensor_range,
         max_sensors, comm_range, grid_spacing) = args

        if not isinstance(veg_x, np.ndarray) or not isinstance(veg_y, np.ndarray):
            raise TypeError("Vegetation coordinates must be numpy arrays.")
        if veg_x.shape != veg_y.shape or veg_x.ndim != 1:
            raise ValueError("veg_x and veg_y must be matching 1D arrays.")
        if not isinstance(bed_coords, np.ndarray):
            raise TypeError("Bed coordinates must be a numpy array.")
        if not all(p > 0 for p in [sensor_range, max_sensors, comm_range, grid_spacing]):
            raise ValueError("Ranges, counts, and spacing must be positive.")

    def _create_plant_array(self, veg_x: np.ndarray, veg_y: np.ndarray) -> np.ndarray:
        """Converts separate x and y vegetation arrays into a single (N, 2) array."""
        if veg_x.size == 0 or veg_y.size == 0:
            print("Warning: No vegetation positions provided.")
            return np.empty((0, 2))
        return np.column_stack([veg_x, veg_y])
    
    def count_unique_coverage(self, sensor_idx: int, chromosome: np.ndarray) -> int:
        """
        Counts how many vegetation points are uniquely covered by a specific sensor.
        """
        active_indices = np.where(chromosome == 1)[0]
        if sensor_idx not in active_indices:
            return 0

        sensor_positions = self.get_sensor_positions(chromosome)
        veg_points = self.veg_pts
        sensor_range = self.sensor_range

        # Precompute which veg points are covered by which sensors
        coverage_matrix = []
        for i in active_indices:
            sensor_pos = self.possible_positions[i]
            distances = np.linalg.norm(veg_points - sensor_pos, axis=1)
            covered = distances <= sensor_range
            coverage_matrix.append(covered)

        # Convert to a matrix: rows = sensors, cols = veg points
        coverage_matrix = np.array(coverage_matrix)

        # Sum across sensors to count how many times each veg point is covered
        total_coverage = np.sum(coverage_matrix, axis=0)

        # Now get the coverage from this specific sensor
        idx = list(active_indices).index(sensor_idx)
        sensor_coverage = coverage_matrix[idx]

        # A veg point is uniquely covered by this sensor if it's covered only once
        unique_coverage = np.logical_and(sensor_coverage, total_coverage == 1)

        return int(np.sum(unique_coverage))
