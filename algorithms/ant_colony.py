# Add these imports at the top
import random
from typing import *
import numpy as np
import os
import sys
from configurations.config_file import OptimizationConfig

# Ensure the parent directory is in the system path to find custom modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class AntColonyOptimizer:
    """
    Implements the Ant Colony Optimization (ACO) algorithm to find an optimal
    trajectory for a mobile sink to visit a set of cluster heads and return
    to a base station, while avoiding defined obstacles (e.g., crop beds).
    """

    # =========================================================================
    # 1. Initialization
    # =========================================================================

    def __init__(self, cluster_heads: np.ndarray, field_data: dict = None):
        """
        Initializes the Ant Colony Optimizer.

        Args:
            cluster_heads (np.ndarray): An (N, 2) array of XY coordinates for each cluster head.
            field_data (dict, optional): A dictionary containing field dimensions and obstacle
                                         coordinates. Defaults to None.
        """
        self.config = OptimizationConfig()
        
        # Validate and store cluster heads
        cluster_heads = np.array(cluster_heads)
        if len(cluster_heads.shape) != 2 or cluster_heads.shape[1] != 2:
            raise ValueError(f"cluster_heads must be (N, 2) array, got shape {cluster_heads.shape}")
        self.cluster_heads = cluster_heads
        
        # Validate and store base station coordinates
        self.base_station = np.array(self.config.environment_base_station)
        if len(self.base_station.shape) != 1 or self.base_station.shape[0] != 2:
            raise ValueError(f"base_station must be (2,) array, got shape {self.base_station.shape}")

        # Store field data for obstacle avoidance
        self.field_data = field_data
        
        # Combine base station (node 0) and cluster heads into a single node list
        self.rendezvous_nodes = np.vstack([
            self.base_station.reshape(1, -1),
            cluster_heads                      
        ])
        self.n_nodes = len(self.rendezvous_nodes)
        
        # Initialize matrices for distance and pheromones
        self.distance_matrix = self._calculate_distance_matrix()
        self.pheromone_matrix = np.ones((self.n_nodes, self.n_nodes)) * 0.1
        
        # Variables to track the best solution found so far
        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_history = []

    # =========================================================================
    # 2. Public Interface
    # =========================================================================

    def optimize(self) -> Tuple[List[int], float, List[float]]:
        """
        Runs the main ACO algorithm loop to find the optimal path.

        Returns:
            Tuple[List[int], float, List[float]]: A tuple containing:
                - The best path found (list of node indices).
                - The length of the best path.
                - The convergence history (best distance at each iteration).
        """
        print(f"Starting ACO optimization with {self.config.collection_n_ants} ants, {self.config.collection_n_iterations} iterations")
        print(f"Optimizing path through {self.n_nodes} nodes ({len(self.cluster_heads)} cluster heads + base station)")
        
        for iteration in range(self.config.collection_n_iterations):
            # In each iteration, ants build their solutions
            for ant in range(self.config.collection_n_ants):
                path, distance = self._construct_ant_solution()
                
                # Update pheromones based on the ant's path (local update)
                self._update_local_pheromones(path, distance)
                
                # Check if this ant found a new global best path
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # After all ants have finished, update pheromones based on the best path (global update)
            self._update_global_pheromones()
            
            # Record the best distance found so far for plotting convergence
            self.convergence_history.append(self.best_distance)
            
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best distance = {self.best_distance:.2f}m")
        
        print(f"ACO optimization completed. Best path length: {self.best_distance:.2f}m")
        return self.best_path, self.best_distance, self.convergence_history

    def get_path_coordinates(self, path: List[int]) -> np.ndarray:
        """
        Converts a path of node indices into an array of their XY coordinates.
        Note: This does NOT include detour waypoints.

        Args:
            path (List[int]): A list of node indices representing the path.

        Returns:
            np.ndarray: An (M, 2) array of XY coordinates for the path.
        """
        return np.array([self.rendezvous_nodes[i] for i in path])

    def get_full_path_for_plotting(self, path: List[int]) -> np.ndarray:
        """
        Converts a path of node indices into a detailed list of XY coordinates,
        including all intermediate waypoints generated for obstacle avoidance.

        Args:
            path (List[int]): A list of node indices representing the best path.

        Returns:
            np.ndarray: A detailed (P, 2) array of coordinates for plotting the full trajectory.
        """
        if not path:
            return np.array([])

        full_path_coords = []
        start_node_coords = self.rendezvous_nodes[path[0]]
        full_path_coords.append(start_node_coords)

        # Iterate through each segment of the path (e.g., node 1 to node 5)
        for i in range(len(path) - 1):
            start_coords = self.rendezvous_nodes[path[i]]
            end_coords = self.rendezvous_nodes[path[i+1]]
            
            # Find the actual waypoints (straight or detour) between these two nodes
            segment_waypoints = self._find_simple_detour_waypoints(start_coords, end_coords)
            
            # Add the waypoints to the full path, skipping the first point to avoid duplication
            full_path_coords.extend(segment_waypoints[1:])

        for x in full_path_coords:
            print(f"{x} -> ")

        return np.array(full_path_coords)

    # =========================================================================
    # 3. Core ACO Algorithm Logic
    # =========================================================================

    def _construct_ant_solution(self) -> Tuple[List[int], float]:
        """
        Constructs a complete tour for a single ant.
        The ant starts at the base station, visits all cluster heads, and returns.

        Returns:
            Tuple[List[int], float]: The path taken by the ant and its total distance.
        """
        # Start every tour from the base station (node 0)
        current_node = 0
        path = [current_node]
        total_distance = 0.0
        
        # Create a list of unvisited nodes (all cluster heads)
        available_nodes = list(range(1, self.n_nodes))
        
        while available_nodes:
            # Decide which node to visit next based on pheromones and distance
            probabilities = self._calculate_probability(current_node, available_nodes)
            
            # Select the next node using a weighted random choice
            if len(probabilities) > 0:
                next_node_idx = np.random.choice(len(available_nodes), p=probabilities)
                next_node = available_nodes[next_node_idx]
            else:
                next_node = random.choice(available_nodes) # Fallback
            
            # "Move" the ant to the next node
            total_distance += self.distance_matrix[current_node, next_node]
            path.append(next_node)
            available_nodes.remove(next_node)
            current_node = next_node
        
        # After visiting all cluster heads, return to the base station
        total_distance += self.distance_matrix[current_node, 0]
        path.append(0)
        
        return path, total_distance

    def _calculate_probability(self, current_node: int, available_nodes: List[int]) -> np.ndarray:
        """
        Calculates the probability for an ant to choose each of the available next nodes.
        Probability is proportional to: (pheromone^alpha) * (heuristic^beta).

        Args:
            current_node (int): The ant's current location (node index).
            available_nodes (List[int]): A list of unvisited node indices.

        Returns:
            np.ndarray: An array of probabilities corresponding to each available node.
        """
        if not available_nodes:
            return np.array([])
        
        numerators = []
        for node in available_nodes:
            pheromone = self.pheromone_matrix[current_node, node] ** self.config.collection_alpha
            heuristic = self._calculate_heuristic(current_node, node) ** self.config.collection_beta
            numerators.append(pheromone * heuristic)
        
        denominator = sum(numerators)
        
        if denominator > 0:
            probabilities = np.array(numerators) / denominator
        else:
            # If all paths have zero desirability, choose uniformly
            probabilities = np.ones(len(available_nodes)) / len(available_nodes)
        
        return probabilities

    def _update_local_pheromones(self, path: List[int], path_length: float):
        """
        Performs a local pheromone update. This is done after each ant completes its tour.
        It slightly reduces pheromones on the traversed path to encourage exploration.
        Formula: τ_ij = (1-ρ)τ_ij + Δτ_ij, where Δτ_ij is based on path length.
        """
        delta_pheromone = 1.0 / path_length
        
        for i in range(len(path) - 1):
            current, next_node = path[i], path[i+1]
            self.pheromone_matrix[current, next_node] = (
                (1 - self.config.collection_rho) * self.pheromone_matrix[current, next_node] + 
                delta_pheromone
            )

    def _update_global_pheromones(self):
        """
        Performs a global pheromone update. This is done after all ants in an
        iteration have completed their tours. It reinforces the best path found so far.
        """
        if self.best_path is None:
            return
        
        # Evaporate pheromone on all paths
        self.pheromone_matrix *= (1 - self.config.collection_gamma)
        
        # Add new pheromone to the best path
        delta_pheromone_best = 1.0 / self.best_distance
        for i in range(len(self.best_path) - 1):
            current, next_node = self.best_path[i], self.best_path[i+1]
            self.pheromone_matrix[current, next_node] += (
                self.config.collection_gamma * delta_pheromone_best
            )

    def _calculate_heuristic(self, i: int, j: int) -> float:
        """
        Calculates the heuristic value (η_ij) for moving from node i to node j.
        This value represents the desirability of the move, typically inverse to the distance.
        Heuristic: η_ij = Q / d(i, j), where Q is a constant.
        """
        if i == j:
            return 0.0
        
        distance = self.distance_matrix[i, j]
        
        # Sanitize input to ensure it's a scalar float
        if isinstance(distance, np.ndarray):
            distance = float(distance.item()) if distance.size == 1 else float(distance.flat[0])
        if not isinstance(distance, (int, float)):
            distance = float(distance)
        
        return self.config.collection_Q / distance

    # =========================================================================
    # 4. Distance Calculation & Obstacle Avoidance
    # =========================================================================

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculates and stores the distance between every pair of nodes.
        Crucially, it uses an obstacle-avoiding distance function.
        """
        n = len(self.rendezvous_nodes)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n): # Symmetric matrix, so only calculate one way
                node_i = self.rendezvous_nodes[i]
                node_j = self.rendezvous_nodes[j]
                
                # Use the special function to get distance around obstacles
                distance = self._calculate_obstacle_avoiding_distance(node_i, node_j)
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances

    def _is_path_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        """
        Checks if the direct line-of-sight path between two points is clear of obstacles
        by sampling points along the line.
        """
        if self.field_data is None:
            return True
        
        path_length = np.linalg.norm(end - start)
        if path_length < self.config.collection_sample_distance:
            return not self._point_in_bed_area((start + end) / 2) # Check midpoint for short paths
        
        n_samples = int(path_length / self.config.collection_sample_distance) + 1
        
        for i in range(n_samples + 1):
            t = i / n_samples
            point = start + t * (end - start)
            if self._point_in_bed_area(point):
                return False # Collision detected
        
        return True # Path is clear

    def _point_in_bed_area(self, point: np.ndarray) -> bool:
        """
        Checks if a given point falls within any of the defined bed obstacle areas,
        including a safety buffer zone.
        """
        buffer = self.config.collection_buffer_distance
        if self.field_data is None or 'bed_coords' not in self.field_data:
            return False
        
        bed_coords = self.field_data['bed_coords']
        if bed_coords is None or len(bed_coords) == 0:
            return False

        for i, bed in enumerate(bed_coords):
            if len(bed) >= 4:  # bed format: [x1, y1, x2, y2]
                # Define bed boundaries with a safety buffer
                min_x = min(bed[0], bed[2]) - buffer
                max_x = max(bed[0], bed[2]) + buffer
                min_y = min(bed[1], bed[3]) - buffer
                max_y = max(bed[1], bed[3]) + buffer
                
                if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                    # print(f"Point {point} blocked by bed {i} (buffer={buffer})")
                    # print(f"  Bed bounds: [{min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f}]")
                    return True
        
        return False
    
    def _find_best_path_segment(self, start: np.ndarray, end: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        A single, authoritative function to find the best path between two points.
        It tries multiple strategies (direct, zigzag, boundary) and returns the shortest one.

        Returns:
            A tuple containing:
            - The minimum distance of the path (float).
            - The list of waypoints for that path (List[np.ndarray]).
        """
        # Strategy 1: Check for a clear direct path.
        if self._is_path_clear(start, end):
            distance = np.linalg.norm(end - start)
            return distance, [start, end]

        # --- If direct path is blocked, find and compare all possible detours ---
        detour_options = []  # This will store all valid detours as (distance, waypoints) tuples

        # Strategy 2: Search for detours along field boundaries.
        if self.field_data and 'environment_field_width' in self.field_data:
            field_width = self.field_data['environment_field_width']
            for offset in [1,2,3,4,5,6,7,8,9,10]: 
                # Bottom boundary detour
                wp1_b = np.array([start[0], offset])
                wp2_b = np.array([end[0], offset])
                if (self._is_path_clear(start, wp1_b) and
                    self._is_path_clear(wp1_b, wp2_b) and
                    self._is_path_clear(wp2_b, end)):
                    dist = (np.linalg.norm(wp1_b - start) + np.linalg.norm(wp2_b - wp1_b) + np.linalg.norm(end - wp2_b))
                    detour_options.append((dist, [start, wp1_b, wp2_b, end]))

                # Top boundary detour
                wp1_t = np.array([start[0], field_width - offset])
                wp2_t = np.array([end[0], field_width - offset])
                if (self._is_path_clear(start, wp1_t) and
                    self._is_path_clear(wp1_t, wp2_t) and
                    self._is_path_clear(wp2_t, end)):
                    dist = (np.linalg.norm(wp1_t - start) + np.linalg.norm(wp2_t - wp1_t) + np.linalg.norm(end - wp2_t))
                    detour_options.append((dist, [start, wp1_t, wp2_t, end]))

        # Strategy 3: Search for paths between beds.
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        # Search a grid of potential intermediate waypoints
        for y_offset in np.linspace(-15, 15, 7): # A wider search range
            for x_offset in np.linspace(-15, 15, 7):
                waypoint = np.array([mid_x + x_offset, mid_y + y_offset])
                if (not self._point_in_bed_area(waypoint) and
                    self._is_path_clear(start, waypoint) and
                    self._is_path_clear(waypoint, end)):
                    
                    dist = np.linalg.norm(waypoint - start) + np.linalg.norm(end - waypoint)
                    detour_options.append((dist, [start, waypoint, end]))

        # Select the best detour option
        if detour_options:
            # If we found one or more valid detours, return the one with the shortest distance.
            best_distance, best_waypoints = min(detour_options, key=lambda item: item[0])
            return best_distance, best_waypoints
        else:
            # Fallback/Penalty: If NO valid detour was found, return a heavily penalized distance
            # and the (blocked) direct path. The high penalty will discourage the optimizer
            # from ever choosing this segment.
            penalty_distance = np.linalg.norm(end - start) * 5.0 
            return penalty_distance, [start, end]
    
    # =========================================================================
    # Strategy for Detour Distance Calculation  
    # =========================================================================

    def _calculate_obstacle_avoiding_distance(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Calculates the travel distance between two points by finding the shortest
        possible path (direct or detour). This now calls the master pathfinding function.
        """
        distance, _ = self._find_best_path_segment(start, end)
        return distance

    def get_full_path_for_plotting(self, path: List[int]) -> np.ndarray:
        """
        Converts a path of node indices into a detailed list of XY coordinates,
        including all intermediate waypoints generated for obstacle avoidance.
        
        MODIFIED TO CALL THE MASTER FUNCTION via a helper.
        """
        if not path:
            return np.array([])

        full_path_coords = []
        start_node_coords = self.rendezvous_nodes[path[0]]
        full_path_coords.append(start_node_coords)

        # Iterate through each segment of the path (e.g., node 1 to node 5)
        for i in range(len(path) - 1):
            start_coords = self.rendezvous_nodes[path[i]]
            end_coords = self.rendezvous_nodes[path[i+1]]
            
            # Get the waypoints from the master pathfinding function
            _, segment_waypoints_full = self._find_best_path_segment(start_coords, end_coords)
            
            # Add the waypoints to the full path, skipping the first point to avoid duplication
            full_path_coords.extend(segment_waypoints_full[1:])

        # This print loop might be excessive, you can comment it out if you wish
        # for x in full_path_coords:
        #     print(f"{x} -> ")

        return np.array(full_path_coords)