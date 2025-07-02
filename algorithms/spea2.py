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
    def __init__(self, problem):
        self.config = OptimizationConfig()
        self.field_validator = PositionValidator(problem.bed_coords)
        self.problem = problem
        self.pop_size = self.config.deployment_pop_size
        self.archive_size = self.config.deployment_archive_size
        self.max_generations = self.config.deployment_generations
        
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
