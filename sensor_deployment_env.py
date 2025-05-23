import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os

class SensorDeploymentSPEA2:
    def __init__(self, field_length, field_width, sensor_range, furrow_width, bed_width, max_sensors):
        """
        Initialize sensor deployment system
        
        Args:
            field_length: Length of the agricultural field
            field_width: Width of the agricultural field  
            sensor_range: Detection range of each sensor
            furrow_width: Width of furrows (where sensors should NOT be placed)
            bed_width: Width of raised beds (where vegetables grow)
        """
        self.field_length = field_length
        self.field_width = field_width
        self.sensor_range = sensor_range
        self.furrow_width = furrow_width
        self.bed_width = bed_width
        self.total_pattern_width = furrow_width + bed_width
        self.max_sensors = max_sensors
        
        # Grid resolution for placement (1m x 1m cells)
        self.grid_size = 1.0
        self.grid_rows = int(field_length / self.grid_size)
        self.grid_cols = int(field_width / self.grid_size)
        
    def is_valid_sensor_position(self, x, y):
        """Check if sensor position is valid (not in furrow center)"""
        # Calculate which pattern section we're in
        pattern_position = x % self.total_pattern_width
        
        # Sensor should not be in the center of furrows
        furrow_center_start = self.bed_width
        furrow_center_end = self.bed_width + self.furrow_width
        
        return not (furrow_center_start <= pattern_position < furrow_center_end)
    
    def calculate_coverage(self, sensor_positions):
        """Calculate coverage matrix for given sensor positions"""
        coverage_matrix = np.zeros((self.grid_rows, self.grid_cols))
        
        for sensor_x, sensor_y in sensor_positions:
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    grid_x = j * self.grid_size
                    grid_y = i * self.grid_size
                    
                    distance = math.sqrt((sensor_x - grid_x)**2 + (sensor_y - grid_y)**2)
                    if distance <= self.sensor_range:
                        coverage_matrix[i, j] += 1
                        
        return coverage_matrix
    
    def calculate_spatial_distribution(self, sensor_positions):
        """Calculate minimum distance between sensors (higher is better)"""
        if len(sensor_positions) < 2:
            return 100  # Perfect if only one sensor
        
        min_distance = float('inf')
        for i in range(len(sensor_positions)):
            for j in range(i+1, len(sensor_positions)):
                dist = math.sqrt((sensor_positions[i][0] - sensor_positions[j][0])**2 + 
                            (sensor_positions[i][1] - sensor_positions[j][1])**2)
                min_distance = min(min_distance, dist)
        
        return min_distance

    def calculate_connectivity(self, sensor_positions):
        """Calculate connectivity between sensors"""
        n_sensors = len(sensor_positions)
        if n_sensors == 0:
            return 0
            
        connectivity_matrix = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                distance = math.sqrt((sensor_positions[i][0] - sensor_positions[j][0])**2 + 
                                   (sensor_positions[i][1] - sensor_positions[j][1])**2)
                if distance <= 2 * self.sensor_range:  # Communication range
                    connectivity_matrix[i, j] = 1
                    connectivity_matrix[j, i] = 1
        
        # Check if all sensors are connected (simple connectivity check)
        connected_count = 0
        for i in range(n_sensors):
            if np.sum(connectivity_matrix[i, :]) > 0:
                connected_count += 1
                
        return connected_count / n_sensors if n_sensors > 0 else 0

    def decode_chromosome(self, chromosome):
        """Convert binary chromosome to sensor positions"""
        sensor_positions = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                idx = i * self.grid_cols + j
                if idx < len(chromosome) and chromosome[idx] == 1:
                    x = j * self.grid_size
                    y = i * self.grid_size
                    if self.is_valid_sensor_position(x, y):
                        sensor_positions.append((x, y))
        return sensor_positions

    def evaluate_individual(self, chromosome):
        """Evaluate individual based on 5 criteria"""
        sensor_positions = self.decode_chromosome(chromosome)
        n_sensors = len(sensor_positions)
        
         # **PENALTY FOR EXCEEDING MAX SENSORS**
        if n_sensors > self.max_sensors:
            # Heavy penalty for exceeding sensor limit
            penalty_factor = (n_sensors - self.max_sensors) * 100
            return np.array([penalty_factor, -1, penalty_factor, -1, -1])
        
        # 1. Sensor node count rate (minimize)
        total_cells = self.grid_rows * self.grid_cols
        sensor_count_rate = (n_sensors / total_cells) * 100
        
        # 2. Coverage rate (maximize)
        coverage_matrix = self.calculate_coverage(sensor_positions)
        covered_cells = np.sum(coverage_matrix > 0)
        coverage_rate = (covered_cells / total_cells) * 100
        
        # 3. Over-coverage rate (minimize)
        overlap_intensity = np.sum(np.maximum(coverage_matrix - 1, 0))  # Sum of excess coverage
        over_coverage_rate = (overlap_intensity / total_cells) * 100
        
        # 4. Valid placement rate (maximize - sensors not in furrow centers)
        valid_placements = sum(1 for pos in sensor_positions if self.is_valid_sensor_position(pos[0], pos[1]))
        placement_rate = (valid_placements / max(n_sensors, 1)) * 100
        
        # 5. Connectivity rate (maximize)
        connectivity_rate = self.calculate_connectivity(sensor_positions) * 100

        # 6. Spatial distribution (maximize minimum distance)
        spatial_dist = self.calculate_spatial_distribution(sensor_positions)

        return np.array([sensor_count_rate, -coverage_rate, over_coverage_rate, 
                        -placement_rate, -connectivity_rate, -spatial_dist])
        
        return np.array([sensor_count_rate, -coverage_rate, over_coverage_rate, -placement_rate, -connectivity_rate])

    def initial_population(self, population_size=50):
        """Generate initial population of sensor deployments"""
        chromosome_length = self.grid_rows * self.grid_cols
        population = []
        
        for _ in range(population_size):
            # Create sparse chromosome (low sensor density)
            chromosome = np.zeros(chromosome_length, dtype=int)
            n_sensors = random.randint(1, min(self.max_sensors, chromosome_length // 20))
            
            # Randomly place sensors
            positions = random.sample(range(chromosome_length), n_sensors)
            for pos in positions:
                chromosome[pos] = 1
                
            # Evaluate the individual
            objectives = self.evaluate_individual(chromosome)
            individual = np.concatenate([chromosome, objectives])
            population.append(individual)
            
        return np.array(population)

    def dominance_function(self, solution_1, solution_2, num_objectives=5):
        """Check if solution_1 dominates solution_2"""
        objectives_1 = solution_1[-num_objectives:]
        objectives_2 = solution_2[-num_objectives:]
        
        dominates = True
        for i in range(num_objectives):
            if objectives_1[i] > objectives_2[i]:  # Assuming minimization
                dominates = False
                break
                
        return dominates and not np.array_equal(objectives_1, objectives_2)

    def find_non_dominated_solutions(self, population, num_objectives=5):
        """Find non-dominated solutions in population"""
        non_dominated_indices = []
        
        for i in range(population.shape[0]):
            is_dominated = False
            for j in range(population.shape[0]):
                if i != j:
                    if self.dominance_function(population[j], population[i], num_objectives):
                        is_dominated = True
                        break
            
            if not is_dominated:
                non_dominated_indices.append(i)
                
        return non_dominated_indices

    def euclidean_distance_objectives(self, obj1, obj2):
        """Calculate Euclidean distance between objective vectors"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(obj1, obj2)))

    def truncation_operator(self, archive, archive_size, num_objectives=5):
        """**TRUNCATION OPERATOR - REPLACES CROWDING DISTANCE**"""
        if archive.shape[0] <= archive_size:
            return archive
        
        current_archive = np.copy(archive)
        
        while current_archive.shape[0] > archive_size:
            n = current_archive.shape[0]
            distances = np.zeros((n, n))
            
            # Calculate distances between all pairs in objective space
            for i in range(n):
                for j in range(n):
                    if i != j:
                        obj_i = current_archive[i, -num_objectives:]
                        obj_j = current_archive[j, -num_objectives:]
                        distances[i, j] = self.euclidean_distance_objectives(obj_i, obj_j)
                    else:
                        distances[i, j] = float('inf')  # Distance to self is infinite
            
            # Find minimum distances for each solution
            min_distances = np.min(distances, axis=1)
            
            # Remove solution with smallest minimum distance (least diverse)
            remove_idx = np.argmin(min_distances)
            current_archive = np.delete(current_archive, remove_idx, axis=0)
        
        return current_archive

    def spea2_environmental_selection(self, population, archive_size, num_objectives=5):
        """**MODIFIED SPEA-II environmental selection using TRUNCATION**"""
        # Find non-dominated solutions
        non_dominated_indices = self.find_non_dominated_solutions(population, num_objectives)
        
        if len(non_dominated_indices) <= archive_size:
            # Not enough non-dominated solutions, add dominated ones
            archive = population[non_dominated_indices]
            remaining = archive_size - len(non_dominated_indices)
            
            if remaining > 0:
                dominated_indices = [i for i in range(population.shape[0]) if i not in non_dominated_indices]
                if dominated_indices:
                    dominated_pop = population[dominated_indices]
                    # Sort by first objective (sensor count)
                    sorted_indices = np.argsort(dominated_pop[:, -num_objectives])
                    additional = dominated_pop[sorted_indices[:remaining]]
                    archive = np.vstack([archive, additional])
        else:
            # Too many non-dominated solutions, use TRUNCATION instead of crowding distance
            non_dominated_pop = population[non_dominated_indices]
            archive = self.truncation_operator(non_dominated_pop, archive_size, num_objectives)
            
        return archive

    def tournament_selection(self, population, num_objectives=5):
        """Binary tournament selection"""
        idx1 = random.randint(0, population.shape[0] - 1)
        idx2 = random.randint(0, population.shape[0] - 1)
        
        # Compare based on first objective (sensor count)
        obj1 = population[idx1, -num_objectives]
        obj2 = population[idx2, -num_objectives]
        
        return idx1 if obj1 <= obj2 else idx2

    def crossover_and_mutation(self, population, mutation_rate=0.1, num_objectives=5):
        """Single-point crossover and bit-flip mutation"""
        chromosome_length = population.shape[1] - num_objectives
        offspring = []
        
        for i in range(0, population.shape[0], 2):
            # Select parents
            parent1_idx = self.tournament_selection(population, num_objectives)
            parent2_idx = self.tournament_selection(population, num_objectives)
            
            parent1 = population[parent1_idx, :chromosome_length].astype(int)
            parent2 = population[parent2_idx, :chromosome_length].astype(int)
            
            # Single-point crossover
            crossover_point = random.randint(1, chromosome_length - 1)
            
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
            # Mutation
            for child in [child1, child2]:
                for j in range(chromosome_length):
                    if random.random() < mutation_rate:
                        child[j] = 1 - child[j]  # Bit flip

                 # **ENFORCE SENSOR LIMIT AFTER MUTATION**
                sensor_count = np.sum(child)
                if sensor_count > self.max_sensors:
                    # Randomly remove excess sensors
                    sensor_indices = np.where(child == 1)[0]
                    remove_count = sensor_count - self.max_sensors
                    remove_indices = random.sample(list(sensor_indices), remove_count)
                    child[remove_indices] = 0
                
                # Evaluate child
                objectives = self.evaluate_individual(child)
                child_with_objectives = np.concatenate([child, objectives])
                offspring.append(child_with_objectives)
                
        return np.array(offspring[:population.shape[0]])  # Maintain population size

    def run_spea2(self, population_size=50, archive_size=50, generations=100, mutation_rate=0.1):
        """Run SPEA-II algorithm for sensor deployment optimization"""
        print("Initializing SPEA-II for sensor deployment...")
        
        # Initialize population and archive
        population = self.initial_population(population_size)
        archive = self.initial_population(archive_size)
        
        for gen in range(generations):
            print(f"Generation {gen + 1}/{generations}")
            
            # Combine population and archive
            combined = np.vstack([population, archive])
            
            # Environmental selection to form new archive
            archive = self.spea2_environmental_selection(combined, archive_size)
            
            # Generate new population
            mating_pool = np.vstack([population, archive])
            population = self.crossover_and_mutation(mating_pool, mutation_rate)
            
        return archive

    def visualize_deployment(self, solution, title="Sensor Deployment"):
        """Visualize the sensor deployment solution"""
        chromosome_length = self.grid_rows * self.grid_cols
        chromosome = solution[:chromosome_length].astype(int)
        sensor_positions = self.decode_chromosome(chromosome)
        coverage_matrix = self.calculate_coverage(sensor_positions)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Sensor positions and field layout
        ax1.set_xlim(0, self.field_width)
        ax1.set_ylim(0, self.field_length)
        ax1.set_xlabel('Width (m)')
        ax1.set_ylabel('Length (m)')
        ax1.set_title(f'{title} - Sensor Positions')
        
        # Draw furrows and beds
        for x in range(0, int(self.field_width), int(self.total_pattern_width)):
            # Bed area
            bed_rect = plt.Rectangle((x, 0), self.bed_width, self.field_length, 
                                   facecolor='lightgreen', alpha=0.3, label='Bed' if x == 0 else "")
            ax1.add_patch(bed_rect)
            
            # Furrow area
            furrow_rect = plt.Rectangle((x + self.bed_width, 0), self.furrow_width, self.field_length,
                                      facecolor='brown', alpha=0.3, label='Furrow' if x == 0 else "")
            ax1.add_patch(furrow_rect)
        
        # Plot sensors
        for x, y in sensor_positions:
            circle = plt.Circle((x, y), self.sensor_range, fill=False, color='red', alpha=0.5)
            ax1.add_patch(circle)
            ax1.plot(x, y, 'ro', markersize=8)
        
        ax1.legend()
        # ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage heatmap
        im = ax2.imshow(coverage_matrix, cmap='YlOrRd', origin='lower', 
                       extent=[0, self.field_width, 0, self.field_length])
        ax2.set_xlabel('Width (m)')
        ax2.set_ylabel('Length (m)')
        ax2.set_title(f'{title} - Coverage Map')
        plt.colorbar(im, ax=ax2, label='Coverage Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        objectives = solution[-5:]
        print(f"\nSolution Metrics:")
        print(f"Sensor Count Rate: {objectives[0]:.2f}%")
        print(f"Coverage Rate: {-objectives[1]:.2f}%") 
        print(f"Over-coverage Rate: {objectives[2]:.2f}%")
        print(f"Valid Placement Rate: {-objectives[3]:.2f}%")
        print(f"Connectivity Rate: {-objectives[4]:.2f}%")
        print(f"Number of Sensors: {len(sensor_positions)}")

# Example usage
if __name__ == "__main__":
    # Field parameters (in meters)
    field_length = 100
    field_width = 100
    sensor_range = 5
    furrow_width = 2
    bed_width = 3
    max_sensors = 20
    
    # Create sensor deployment system
    sensor_system = SensorDeploymentSPEA2(
        field_length=field_length,
        field_width=field_width,
        sensor_range=sensor_range,
        furrow_width=furrow_width,
        bed_width=bed_width,
        max_sensors=max_sensors
    )
    
    # Run SPEA-II optimization
    final_archive = sensor_system.run_spea2(
        population_size= 20,
        archive_size=10,
        generations=10,
        mutation_rate=0.1
    )
    
    # Visualize best solutions
    print("\n" + "="*50)
    print("SPEA-II Sensor Deployment Results")
    print("="*50)
    
    # Show the solution with best coverage
    best_coverage_idx = np.argmin(final_archive[:, -4])  # -coverage_rate (negative, so min is best)
    print("\nBest Coverage Solution:")
    sensor_system.visualize_deployment(final_archive[best_coverage_idx], "Best Coverage")
    
    # Show the solution with minimum sensors
    min_sensors_idx = np.argmin(final_archive[:, -5])  # sensor_count_rate
    print("\nMinimum Sensors Solution:")
    sensor_system.visualize_deployment(final_archive[min_sensors_idx], "Minimum Sensors")