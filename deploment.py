"""
Vegetation Sensor Placement Problem using SPEA2 Algorithm

This module defines a multi-objective optimization problem for placing sensors
in agricultural fields using the SPEA2 algorithm.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import logging
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod

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

class Individual:
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome
        self.objectives = None
        self.strength = 0
        self.raw_fitness = 0
        self.density = 0
        self.fitness = 0

class SPEA2:
    def __init__(self, problem, pop_size=100, archive_size=100, max_generations=100):
        self.problem = problem
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        
    def initialize_population(self) -> List[Individual]:
        population = []
        for _ in range(self.pop_size):
            # Create random binary chromosome
            chromosome = np.random.choice([0, 1], size=self.problem.n_var, p=[0.9, 0.1])
            ind = Individual(chromosome)
            objectives, _ = self.problem._evaluate_single_solution(chromosome)
            ind.objectives = objectives
            population.append(ind)
        return population

    def calculate_strength(self, population: List[Individual], archive: List[Individual]):
        combined = population + archive
        for ind in combined:
            ind.strength = sum(1 for other in combined 
                             if self.dominates(ind.objectives, other.objectives))

    def calculate_raw_fitness(self, population: List[Individual], archive: List[Individual]):
        combined = population + archive
        for ind in combined:
            ind.raw_fitness = sum(other.strength for other in combined 
                                if self.dominates(other.objectives, ind.objectives))

    def calculate_density(self, population: List[Individual], archive: List[Individual]):
        combined = population + archive
        k = int(np.sqrt(len(combined)))
        
        for ind in combined:
            distances = []
            for other in combined:
                if other != ind:
                    dist = np.linalg.norm(ind.objectives - other.objectives)
                    distances.append(dist)
            distances.sort()
            ind.density = 1 / (distances[k] + 2) if len(distances) > k else 0

    def calculate_fitness(self, population: List[Individual], archive: List[Individual]):
        self.calculate_strength(population, archive)
        self.calculate_raw_fitness(population, archive)
        self.calculate_density(population, archive)
        
        combined = population + archive
        for ind in combined:
            ind.fitness = ind.raw_fitness + ind.density

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return (np.all(obj1 <= obj2) and np.any(obj1 < obj2))

    def environmental_selection(self, population: List[Individual], 
                             archive: List[Individual]) -> List[Individual]:
        combined = population + archive
        # Select non-dominated individuals
        new_archive = [ind for ind in combined 
                      if ind.fitness < 1.0]
        
        if len(new_archive) == self.archive_size:
            return new_archive
        
        if len(new_archive) < self.archive_size:
            # Fill with best dominated solutions
            combined.sort(key=lambda x: x.fitness)
            new_archive.extend(combined[len(new_archive):self.archive_size])
            return new_archive[:self.archive_size]
        
        # Truncate archive using density information
        while len(new_archive) > self.archive_size:
            # Find individual with minimum distance to another individual
            min_dist = float('inf')
            to_remove = None
            
            for i, ind1 in enumerate(new_archive):
                distances = []
                for j, ind2 in enumerate(new_archive):
                    if i != j:
                        dist = np.linalg.norm(ind1.objectives - ind2.objectives)
                        distances.append(dist)
                min_local = min(distances)
                if min_local < min_dist:
                    min_dist = min_local
                    to_remove = i
            
            if to_remove is not None:
                new_archive.pop(to_remove)
        
        return new_archive

    def tournament_selection(self, population: List[Individual], size=2) -> Individual:
        tournament = random.sample(population, size)
        return min(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        mask = np.random.random(len(parent1.chromosome)) < 0.5
        child_chromosome = np.where(mask, parent1.chromosome, parent2.chromosome)
        child = Individual(child_chromosome)
        objectives, _ = self.problem._evaluate_single_solution(child_chromosome)
        child.objectives = objectives
        return child
    
    def _should_remove_sensor(self, sensor_idx: int, chromosome: np.ndarray) -> bool:
        """
        Determine if a sensor should be removed based on its contribution.
        
        Parameters:
        -----------
        sensor_idx : int
            Index of the sensor position to consider removing
        chromosome : np.ndarray
            Current chromosome configuration
            
        Returns:
        --------
        bool
            True if sensor should be removed, False otherwise
        """
        try:
            # Get current sensor positions
            current_sensors = self.problem.get_sensor_positions(chromosome)
            
            if current_sensors.shape[0] <= 1:
                return False  # Don't remove if only one or no sensors
            
            # Calculate current coverage
            current_coverage = CoverageCalculator.calculate_coverage_rate(
                current_sensors, self.problem.veg_pts, self.problem.sensor_range
            )
            
            # Simulate removal of this sensor
            test_chromosome = chromosome.copy()
            test_chromosome[sensor_idx] = 0
            test_sensors = self.problem.get_sensor_positions(test_chromosome)
            
            # Calculate coverage without this sensor
            if test_sensors.shape[0] == 0:
                new_coverage = 0.0
            else:
                new_coverage = CoverageCalculator.calculate_coverage_rate(
                    test_sensors, self.problem.veg_pts, self.problem.sensor_range
                )
            
            # Remove sensor if coverage loss is minimal (less than 5%)
            coverage_loss = current_coverage - new_coverage
            
            # Also consider connectivity - don't remove if it would disconnect the network
            if test_sensors.shape[0] > 1:
                new_connectivity = CoverageCalculator.calculate_connectivity_rate(
                    test_sensors, self.problem.comm_range
                )
                if new_connectivity < 50.0:  # Maintain minimum connectivity
                    return False
            
            return coverage_loss < 5.0
            
        except Exception as e:
            print(f"Error in _should_remove_sensor: {e}")
            return False

    def _should_add_sensor(self, sensor_idx: int, chromosome: np.ndarray) -> bool:
        """
        Determine if a sensor should be added based on coverage improvement.
        
        Parameters:
        -----------
        sensor_idx : int
            Index of the sensor position to consider adding
        chromosome : np.ndarray
            Current chromosome configuration
            
        Returns:
        --------
        bool
            True if sensor should be added, False otherwise
        """
        try:
            # Check if we're at maximum sensor limit
            current_sensor_count = np.sum(chromosome)
            if current_sensor_count >= self.problem.max_sensors:
                return False
            
            # Get current sensor positions
            current_sensors = self.problem.get_sensor_positions(chromosome)
            
            # Calculate current coverage
            if current_sensors.shape[0] == 0:
                current_coverage = 0.0
            else:
                current_coverage = CoverageCalculator.calculate_coverage_rate(
                    current_sensors, self.problem.veg_pts, self.problem.sensor_range
                )
            
            # Simulate adding this sensor
            test_chromosome = chromosome.copy()
            test_chromosome[sensor_idx] = 1
            test_sensors = self.problem.get_sensor_positions(test_chromosome)
            
            # Calculate coverage with new sensor
            new_coverage = CoverageCalculator.calculate_coverage_rate(
                test_sensors, self.problem.veg_pts, self.problem.sensor_range
            )
            
            # Add sensor if it improves coverage significantly (more than 2%)
            coverage_improvement = new_coverage - current_coverage
            
            # Also check if the new position improves connectivity
            connectivity_bonus = False
            if current_sensors.shape[0] > 0:
                # Check if new sensor connects to existing network
                new_sensor_pos = self.problem.possible_positions[sensor_idx:sensor_idx+1]
                distances_to_existing = np.linalg.norm(
                    current_sensors[:, None, :] - new_sensor_pos[None, :, :],
                    axis=2
                )
                min_distance = np.min(distances_to_existing)
                if min_distance <= self.problem.comm_range:
                    connectivity_bonus = True
            
            # Add sensor if it provides good coverage improvement or connectivity benefit
            return coverage_improvement > 2.0 or (coverage_improvement > 0.5 and connectivity_bonus)
            
        except Exception as e:
            print(f"Error in _should_add_sensor: {e}")
            return False
        
    # def mutation(self, individual: Individual, mutation_rate=0.01) -> Individual: # original version
    #     mutated = individual.chromosome.copy()
    #     for i in range(len(mutated)):
    #         if random.random() < mutation_rate:
    #             mutated[i] = 1 - mutated[i]
        
    #     new_ind = Individual(mutated)
    #     objectives, _ = self.problem._evaluate_single_solution(mutated)
    #     new_ind.objectives = objectives
    #     return new_ind

    def mutation(self, individual: Individual, mutation_rate=0.01) -> Individual: #improved version
        mutated = individual.chromosome.copy()
        
        # Adaptive mutation rate based on fitness
        adaptive_rate = mutation_rate * (1 + individual.fitness / 10)
        
        for i in range(len(mutated)):
            if random.random() < adaptive_rate:
                if mutated[i] == 1:
                    # Smart removal: remove sensors with low contribution
                    if self._should_remove_sensor(i, mutated):
                        mutated[i] = 0
                else:
                    # Smart addition: add sensors in uncovered areas
                    if self._should_add_sensor(i, mutated):
                        mutated[i] = 1
        
        new_ind = Individual(mutated)
        objectives, _ = self.problem._evaluate_single_solution(mutated)
        new_ind.objectives = objectives
        return new_ind

    def run(self) -> Tuple[List[Individual], List[List[Individual]]]:
        population = self.initialize_population()
        archive = []
        history = []
        
        for gen in range(self.max_generations):
            self.calculate_fitness(population, archive)
            archive = self.environmental_selection(population, archive)
            history.append(archive.copy())
            
            if gen == self.max_generations - 1:
                break
            
            # Create new population
            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(archive if archive else population)
                parent2 = self.tournament_selection(archive if archive else population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                new_population.append(offspring)
            
            population = new_population
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Archive size = {len(archive)}")
        
        return archive, history

# Add after PositionValidator and before SPEA2

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

# Modify VegSensorProblem class to remove pymoo inheritance
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
            out["F"] = np.full((X.shape[0], self.n_obj), 100)
            out["G"] = np.full((X.shape[0], 1), 0)
    
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

