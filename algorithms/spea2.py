import numpy as np
from typing import Tuple, Dict, Any, List
import logging
import random
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from configurations.data_structure import SensorMetrics, Individual
from configurations.config_file import OptimizationConfig
from utilities.position_validator import PositionValidator
from utilities.coverage_calculator import CoverageCalculator
from utilities.veg_sensor_problem import VegSensorProblem

class SPEA2:
    """
    Implements the Strength Pareto Evolutionary Algorithm 2 (SPEA2) for solving
    multi-objective optimization problems. This class is specifically configured
    for a sensor placement problem.
    """

    # =====================================================================================
    # 1. Initialization
    # =====================================================================================

    def __init__(self, problem):
        """
        Initializes the SPEA2 algorithm solver.

        This constructor sets up the configuration parameters, validator, and
        the problem definition that the algorithm will solve.

        Parameters:
        -----------
        problem : VegSensorProblem
            An object representing the optimization problem, containing details
            like evaluation functions, constraints, and problem-specific data.
        """
        self.config = OptimizationConfig()
        self.field_validator = PositionValidator(problem.bed_coords)
        self.problem = problem
        self.pop_size = self.config.deployment_pop_size
        self.archive_size = self.config.deployment_archive_size
        self.max_generations = self.config.deployment_generations

    # =====================================================================================
    # 2. Main Execution Flow
    # =====================================================================================

    
    def run(self) -> Tuple[List[Individual], List[List[Individual]]]:
        """
        Executes the main evolutionary loop of the SPEA2 algorithm.

        This version is modified to run until at least one solution with 100%
        coverage is found, or until the maximum number of generations is reached.
        """
        population = self.initialize_population()
        archive = []
        history = []
        
        found_full_coverage = False
        gen = 0
        
        print(f"Starting SPEA2. Will run for max {self.max_generations} generations or until 100% coverage is found.")
        
        while not found_full_coverage and gen < self.max_generations:
            self.calculate_fitness(population, archive)
            archive = self.environmental_selection(population, archive)
            history.append(archive.copy())

            for individual in archive:
                if individual.objectives[1] < -99.99:
                    print(f"\nSUCCESS: Found a solution with 100% coverage at Generation {gen}!")
                    found_full_coverage = True
            
            if found_full_coverage or gen == self.max_generations - 1:
                break
            
            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(archive if archive else population)
                parent2 = self.tournament_selection(archive if archive else population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                new_population.append(offspring)
            
            population = new_population
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Archive size = {len(archive)}. Still searching for 100% coverage...")
                
            gen += 1 

        if not found_full_coverage:
            print(f"\nWARNING: Algorithm finished after {self.max_generations} generations "
                f"but did not find a solution with 100% coverage.")
        
        if found_full_coverage:
            final_archive = [ind for ind in archive if ind.objectives[1] < -99.99]
            print(f"Returning {len(final_archive)} solutions that meet the 100% coverage criteria.")
            return final_archive, history

        return archive, history

    # =====================================================================================
    # 3. Core SPEA2 Fitness Assignment and Selection
    # =====================================================================================

    def calculate_fitness(self, population: List[Individual], archive: List[Individual]):
        """
        Calculates the final fitness for each individual in the combined
        population and archive.

        This is the main fitness assignment step in SPEA2, which combines
        strength, raw fitness, and density into a single fitness value. It calls
        the helper methods to compute each component.
        """
        self.calculate_strength(population, archive)
        self.calculate_raw_fitness(population, archive)
        self.calculate_density(population, archive)

        combined = population + archive
        for ind in combined:
            ind.fitness = ind.raw_fitness + ind.density

    def calculate_strength(self, population: List[Individual], archive: List[Individual]):
        """
        Calculates the strength value (S) for each individual.

        An individual's strength is the number of other individuals in the
        combined population and archive that it dominates. A higher strength
        value indicates a more dominant solution.
        """
        combined = population + archive
        for ind in combined:
            ind.strength = sum(1 for other in combined
                             if self.dominates(ind.objectives, other.objectives))

    def calculate_raw_fitness(self, population: List[Individual], archive: List[Individual]):
        """
        Calculates the raw fitness (R) for each individual.

        An individual's raw fitness is the sum of the strength values of all
        individuals that dominate it. A lower raw fitness value is better, with
        a value of 0 indicating a non-dominated solution.
        """
        combined = population + archive
        for ind in combined:
            ind.raw_fitness = sum(other.strength for other in combined
                                if self.dominates(other.objectives, ind.objectives))

    def calculate_density(self, population: List[Individual], archive: List[Individual]):
        """
        Calculates the density (D) for each individual.

        Density is estimated using a k-nearest neighbor approach. It measures
        the proximity of an individual to its neighbors in the objective space.
        The value is the inverse of the distance to the k-th nearest neighbor,
        used to promote diversity in the solutions.
        """
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

    def environmental_selection(self, population: List[Individual],
                             archive: List[Individual]) -> List[Individual]:
        """
        Selects the individuals that will form the archive for the next generation.

        The process is as follows:
        1. All non-dominated individuals (fitness < 1) are copied to the new archive.
        2. If the archive is too small, it's filled with the best-dominated
           individuals (lowest fitness values).
        3. If the archive is too large, individuals are removed one by one based
           on their density (the one with the smallest distance to a neighbor is
           removed) until the archive reaches its target size.
        """
        combined = population + archive
        # Select non-dominated individuals (raw fitness is 0, so fitness < 1)
        new_archive = [ind for ind in combined
                      if ind.fitness < 1.0]

        if len(new_archive) == self.archive_size:
            return new_archive

        if len(new_archive) < self.archive_size:
            # Fill archive with best dominated solutions if under capacity
            combined.sort(key=lambda x: x.fitness)
            new_archive.extend(combined[len(new_archive):self.archive_size])
            return new_archive[:self.archive_size]

        # Truncate archive using density information if over capacity
        while len(new_archive) > self.archive_size:
            # Find individual with the minimum distance to another individual
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

    # =====================================================================================
    # 4. Genetic Operators (Selection, Crossover, Mutation)
    # =====================================================================================

    def initialize_population(self) -> List[Individual]:
        """
        Creates the initial population of solutions.

        Each individual is created with a random binary chromosome, where the
        probability of a gene being '1' is low to encourage sparse solutions.
        The objectives for each new individual are evaluated immediately.
        """
        population = []
        heuristic_chromosome = self.problem.create_greedy_heuristic_solution()
        
        ind = Individual(heuristic_chromosome)
        objectives, _ = self.problem._evaluate_single_solution(heuristic_chromosome)
        ind.objectives = objectives
        population.append(ind)
        
        num_random_to_create = self.pop_size - 1
        print(f"Generating {num_random_to_create} random individuals to complete the population...")
        
        for _ in range(num_random_to_create):
            # Create random binary chromosome
            chromosome = np.random.choice([0, 1], size=self.problem.n_var, p=[0.9, 0.1])
            ind = Individual(chromosome)
            objectives, _ = self.problem._evaluate_single_solution(chromosome)
            ind.objectives = objectives
            population.append(ind)
        
        return population

    def tournament_selection(self, population: List[Individual], size=2) -> Individual:
        """
        Selects a single individual from a population using binary tournament selection.

        Two individuals are randomly chosen from the population, and the one
        with the better (lower) fitness value is selected as the winner.
        This method is used to select parents for crossover.
        """
        tournament = random.sample(population, size)
        return min(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Creates a child individual by performing uniform crossover on two parents.

        A random binary mask is generated. The child's chromosome takes genes
        from parent1 where the mask is 1 and from parent2 where the mask is 0.
        The new child's objectives are then evaluated.
        """
        mask = np.random.random(len(parent1.chromosome)) < 0.5
        child_chromosome = np.where(mask, parent1.chromosome, parent2.chromosome)
        child = Individual(child_chromosome)
        objectives, _ = self.problem._evaluate_single_solution(child_chromosome)
        child.objectives = objectives
        return child

    def mutation(self, individual: Individual, mutation_rate=0.01) -> Individual:
        """
        Applies a "smart" mutation to an individual's chromosome.

        This mutation operator has an adaptive rate based on the individual's
        fitness. For each gene, it decides whether to flip it (0 to 1 or 1 to 0)
        based on problem-specific heuristics provided by the `_should_add_sensor`
        and `_should_remove_sensor` helper methods.
        """
        mutated = individual.chromosome.copy()

        # Adaptive mutation rate increases for less-fit individuals
        adaptive_rate = mutation_rate * (1 + individual.fitness / 10)

        for i in range(len(mutated)):
            if random.random() < adaptive_rate:
                if mutated[i] == 1:
                    # Try to remove an existing sensor if it has low contribution
                    if self._should_remove_sensor(i, mutated):
                        mutated[i] = 0
                else:
                    # Try to add a new sensor if it provides a good benefit
                    if self._should_add_sensor(i, mutated):
                        mutated[i] = 1

        new_ind = Individual(mutated)
        objectives, _ = self.problem._evaluate_single_solution(mutated)
        new_ind.objectives = objectives
        return new_ind

    # =====================================================================================
    # 5. Helper & Utility Functions
    # =====================================================================================

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Checks if solution 1 dominates solution 2 (Pareto dominance).

        For minimization problems, a solution `obj1` dominates `obj2` if it is
        better or equal in all objectives and strictly better in at least one.
        """
        return (np.all(obj1 <= obj2) and np.any(obj1 < obj2))

    def _should_remove_sensor(self, sensor_idx: int, chromosome: np.ndarray) -> bool:
        """
        Improved logic: Remove a sensor if it contributes very little unique coverage,
        doesn't reduce total coverage significantly, and maintains network connectivity.
        """
        try:
            current_sensors = self.problem.get_sensor_positions(chromosome)
            if current_sensors.shape[0] <= 1:
                return False

            current_coverage = CoverageCalculator.calculate_coverage_rate(
                current_sensors, self.problem.veg_pts, self.problem.sensor_range
            )

            test_chromosome = chromosome.copy()
            test_chromosome[sensor_idx] = 0
            test_sensors = self.problem.get_sensor_positions(test_chromosome)

            # Don't remove if it disconnects the entire network
            if test_sensors.shape[0] == 0:
                return False

            new_coverage = CoverageCalculator.calculate_coverage_rate(
                test_sensors, self.problem.veg_pts, self.problem.sensor_range
            )

            coverage_loss = current_coverage - new_coverage

            # Check connectivity
            new_connectivity = CoverageCalculator.calculate_connectivity_rate(
                test_sensors, self.problem.comm_range
            )
            if new_connectivity < 50.0:
                return False

            # check unique contribution
            unique_veg_points = self.problem.count_unique_coverage(sensor_idx, chromosome)
            if unique_veg_points < 5 and coverage_loss < 1.0:
                if self.problem.current_coverage > 95 and unique_veg_points < 5:
                    return True
                # return True

            return coverage_loss < 2.0 and unique_veg_points < 3

        except Exception as e:
            print(f"Error in _should_remove_sensor: {e}")
            return False


    def _should_add_sensor(self, sensor_idx: int, chromosome: np.ndarray) -> bool:
        """
        Determines if a sensor should be added based on potential improvement.

        A sensor is considered for addition if it significantly improves
        vegetation coverage (e.g., more than 2% gain) or provides a meaningful
        connectivity benefit to the existing sensor network.
        """
        try:
            # Don't add if we are already at the sensor limit
            current_sensor_count = np.sum(chromosome)
            if current_sensor_count >= self.problem.max_sensors:
                return False

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

            # Calculate coverage with the new sensor
            new_coverage = CoverageCalculator.calculate_coverage_rate(
                test_sensors, self.problem.veg_pts, self.problem.sensor_range
            )

            coverage_improvement = new_coverage - current_coverage

            # Check if adding this sensor improves network connectivity
            connectivity_bonus = False
            if current_sensors.shape[0] > 0:
                new_sensor_pos = self.problem.possible_positions[sensor_idx:sensor_idx+1]
                distances_to_existing = np.linalg.norm(
                    current_sensors[:, None, :] - new_sensor_pos[None, :, :],
                    axis=2
                )
                min_distance = np.min(distances_to_existing)
                if min_distance <= self.problem.comm_range:
                    connectivity_bonus = True

            # Add if coverage gain is good or if it helps connectivity
            return coverage_improvement > 2.0 or (coverage_improvement > 0.5 and connectivity_bonus)

        except Exception as e:
            print(f"Error in _should_add_sensor: {e}")
            return False
        
    