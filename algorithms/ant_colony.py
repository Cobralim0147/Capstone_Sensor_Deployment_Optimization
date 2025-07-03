# Add these imports at the top
import random
from typing import *
import numpy as np
import os
import sys
from configurations.config_file import OptimizationConfig

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class AntColonyOptimizer:
    """Ant Colony Optimization for mobile sink trajectory planning."""
    
    def __init__(self, cluster_heads: np.ndarray):
        self.config = OptimizationConfig()
        cluster_heads = np.array(cluster_heads)
        if len(cluster_heads.shape) != 2 or cluster_heads.shape[1] != 2:
            raise ValueError(f"cluster_heads must be (N, 2) array, got shape {cluster_heads.shape}")
        
        self.cluster_heads = cluster_heads
        
        # Convert base station to numpy array
        self.base_station = np.array(self.config.environment_base_station)
        if len(self.base_station.shape) != 1 or self.base_station.shape[0] != 2:
            raise ValueError(f"base_station must be (2,) array, got shape {self.base_station.shape}")
        
        # Create rendezvous nodes (cluster heads + base station)
        self.rendezvous_nodes = np.vstack([
            self.base_station.reshape(1, -1),  # (1, 2)
            cluster_heads                      
        ])
        self.n_nodes = len(self.rendezvous_nodes)
        
        # Initialize distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes)) * 0.1
        
        # Best solution tracking
        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_history = []
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all rendezvous nodes."""
        n = len(self.rendezvous_nodes)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate Euclidean distance between nodes i and j
                    node_i = self.rendezvous_nodes[i]
                    node_j = self.rendezvous_nodes[j]
                    
                    # Ensure this returns a scalar, not an array
                    distance = np.linalg.norm(node_i - node_j)
                    distances[i, j] = distance
                else:
                    distances[i, j] = 0.0
        
        return distances
    
    def _calculate_heuristic(self, i: int, j: int) -> float:
        """Calculate heuristic function η_ij(t) = Q / d(i,j)"""
        if i == j:
            return 0.0
        
        # Get distance from distance matrix
        distance = self.distance_matrix[i, j]
        
        # Convert to scalar safely
        if isinstance(distance, np.ndarray):
            if distance.size == 1:
                distance = distance.item()  # Single element array
            else:
                # Multiple elements - take the first or compute norm
                print(f"Warning: Expected scalar distance but got array of size {distance.size}")
                print(f"Distance array: {distance}")
                distance = float(distance.flat[0])  # Take first element
        
        # Ensure it's a number
        if not isinstance(distance, (int, float)):
            print(f"Warning: Distance is not a number: {type(distance)}, value: {distance}")
            distance = float(distance)
        

        return self.config.collection_Q / distance

    
    def _calculate_probability(self, current_node: int, available_nodes: List[int]) -> np.ndarray:
        """Calculate probability for ant to choose next node."""
        if not available_nodes:
            return np.array([])
        
        probabilities = np.zeros(len(available_nodes))
        
        # Calculate numerator for each available node
        numerators = []
        for node in available_nodes:
            pheromone = self.pheromone_matrix[current_node, node] ** self.config.collection_alpha
            heuristic = self._calculate_heuristic(current_node, node) ** self.config.collection_beta
            numerators.append(pheromone * heuristic)
        
        # Calculate denominator (sum of all numerators)
        denominator = sum(numerators)
        
        # Calculate probabilities
        if denominator > 0:
            probabilities = np.array(numerators) / denominator
        else:
            # If denominator is 0, use uniform probability
            probabilities = np.ones(len(available_nodes)) / len(available_nodes)
        
        return probabilities
    
    def _construct_ant_solution(self) -> Tuple[List[int], float]:
        """Construct a solution for a single ant."""
        # Start from base station (node 0)
        current_node = 0
        path = [current_node]
        total_distance = 0.0
        
        # Available nodes (all cluster heads)
        available_nodes = list(range(1, self.n_nodes))
        
        # Visit all cluster heads
        while available_nodes:
            # Calculate probabilities for next node selection
            probabilities = self._calculate_probability(current_node, available_nodes)
            
            # Select next node using roulette wheel selection
            if len(probabilities) > 0:
                next_node_idx = np.random.choice(len(available_nodes), p=probabilities)
                next_node = available_nodes[next_node_idx]
            else:
                # Fallback: choose randomly
                next_node = random.choice(available_nodes)
                next_node_idx = available_nodes.index(next_node)
            
            # Move to next node
            total_distance += self.distance_matrix[current_node, next_node]
            path.append(next_node)
            available_nodes.remove(next_node)
            current_node = next_node
        
        # Return to base station
        total_distance += self.distance_matrix[current_node, 0]
        path.append(0)
        
        return path, total_distance
    
    def _update_local_pheromones(self, path: List[int], path_length: float):
        """Update local pheromone trails for a specific path."""
        delta_pheromone = 1.0 / path_length
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Update pheromone: τ_ij(t+1) = (1-ρ)τ_ij(t) + Δτ_ij(t)
            self.pheromone_matrix[current, next_node] = (
                (1 - self.config.collection_rho) * self.pheromone_matrix[current, next_node] + 
                delta_pheromone
            )
    
    def _update_global_pheromones(self):
        """Update global pheromone trails using the best solution."""
        if self.best_path is None:
            return
        
        # Global evaporation
        self.pheromone_matrix *= (1 - self.config.collection_gamma)
        
        # Reinforce best path
        delta_pheromone_best = 1.0 / self.best_distance
        
        for i in range(len(self.best_path) - 1):
            current = self.best_path[i]
            next_node = self.best_path[i + 1]
            
            self.pheromone_matrix[current, next_node] += (
                self.config.collection_gamma * delta_pheromone_best
            )
    
    def optimize(self) -> Tuple[List[int], float, List[float]]:
        """Run the ACO algorithm to find optimal path."""
        print(f"Starting ACO optimization with {self.config.collection_n_ants} ants, {self.config.collection_n_iterations} iterations")
        print(f"Optimizing path through {self.n_nodes} nodes ({len(self.cluster_heads)} cluster heads + base station)")
        
        for iteration in range(self.config.collection_n_iterations):
            iteration_best_distance = float('inf')
            iteration_best_path = None
            
            # Deploy ants
            for ant in range(self.config.collection_n_ants):
                path, distance = self._construct_ant_solution()
                
                # Update local pheromones
                self._update_local_pheromones(path, distance)
                
                # Track iteration best
                if distance < iteration_best_distance:
                    iteration_best_distance = distance
                    iteration_best_path = path.copy()
                
                # Track global best
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Update global pheromones
            self._update_global_pheromones()
            
            # Record convergence
            self.convergence_history.append(self.best_distance)
            
            # Print progress every 20 iterations
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best distance = {self.best_distance:.2f}m")
        
        print(f"ACO optimization completed. Best path length: {self.best_distance:.2f}m")
        return self.best_path, self.best_distance, self.convergence_history
    
    def get_path_coordinates(self, path: List[int]) -> np.ndarray:
        """Convert path indices to actual coordinates."""
        return np.array([self.rendezvous_nodes[i] for i in path])
    
