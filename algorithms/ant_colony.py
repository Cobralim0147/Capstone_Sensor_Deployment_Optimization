# Add these imports at the top
import random
from typing import *
import numpy as np
import os
import sys
from collections import deque
from configurations.config_file import OptimizationConfig

# Ensure the parent directory is in the system path to find custom modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class AntColonyOptimizer:
    """
    Implements a grid-based Ant Colony Optimization (ACO) algorithm to find an optimal
    trajectory for a mobile sink to visit a set of cluster heads and return
    to a base station, while avoiding defined obstacles (e.g., crop beds).
    
    The ants can only move on a predefined grid and are restricted to 4-directional
    movement (up, down, left, right - no diagonals).
    """

    # =========================================================================
    # 1. Initialization
    # =========================================================================

    def __init__(self, cluster_heads: np.ndarray, field_data: dict = None):
        """
        Initializes the Grid-Based Ant Colony Optimizer.

        Args:
            cluster_heads (np.ndarray): An (N, 2) array of XY coordinates for each cluster head.
            field_data (dict, optional): A dictionary containing field dimensions and obstacle
                                         coordinates. Defaults to None.
            grid_resolution (float): The spacing between grid points in meters. Defaults to 1.0.
        """
        self.config = OptimizationConfig()
        self.grid_resolution = self.config.collection_grid_resolution
        
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
        
        # Initialize the grid system
        self._initialize_grid()
        
        # Map cluster heads and base station to grid coordinates
        self._map_nodes_to_grid()

        # Create collection zones for cluster heads
        self._create_collection_zones()
        
        # Initialize matrices for distance and pheromones
        self.distance_matrix = self._calculate_distance_matrix()
        self.pheromone_matrix = np.ones((len(self.grid_nodes), len(self.grid_nodes))) * 0.1
        
        # Variables to track the best solution found so far
        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_history = []

    def _initialize_grid(self):
        """
        Creates a grid of valid travel points, excluding obstacles.
        """
        # Determine field boundaries
        if self.field_data:
            field_length = self.field_data.get('environment_field_length', 100)
            field_width = self.field_data.get('environment_field_width', 100)
        else:
            # Default field size if not specified
            field_length = 100
            field_width = 100
        
        # Create grid points
        x_coords = np.arange(0, field_length + self.grid_resolution, self.grid_resolution)
        y_coords = np.arange(0, field_width + self.grid_resolution, self.grid_resolution)
        
        # Generate all grid points
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                point = np.array([x, y])
                # Only add points that are not in obstacle areas
                if not self._point_in_bed_area(point):
                    grid_points.append(point)
        
        self.grid_points = np.array(grid_points)
        print(f"Generated {len(self.grid_points)} valid grid points")
        
        # Create a mapping from grid coordinates to indices for faster lookup
        self.grid_coord_to_index = {}
        for i, point in enumerate(self.grid_points):
            key = (round(point[0], 2), round(point[1], 2))
            self.grid_coord_to_index[key] = i

    def _map_nodes_to_grid(self):
        """
        Maps the base station and cluster heads to the nearest valid grid points.
        """
        all_nodes = np.vstack([self.base_station.reshape(1, -1), self.cluster_heads])
        
        self.grid_nodes = []
        self.node_to_grid_index = {}
        
        # Keep track of grid indices that have been used
        used_grid_indices = {}

        for i, node in enumerate(all_nodes):
            distances = np.linalg.norm(self.grid_points - node, axis=1)
            nearest_grid_idx = np.argmin(distances)
            
            # Detect duplications
            if nearest_grid_idx in used_grid_indices:
                original_node_idx = used_grid_indices[nearest_grid_idx]
                print(f"WARNING: Node {i} at {node} and Node {original_node_idx} are both mapped to the same grid point index {nearest_grid_idx}.")
                print("This can cause division by zero. Consider increasing grid_resolution or adjusting node positions.")
            
            used_grid_indices[nearest_grid_idx] = i

            nearest_grid_point = self.grid_points[nearest_grid_idx]
            
            self.grid_nodes.append(nearest_grid_point)
            self.node_to_grid_index[i] = nearest_grid_idx
            
            print(f"Node {i} at {node} mapped to grid point {nearest_grid_point} (grid index {nearest_grid_idx})")
        
        self.grid_nodes = np.array(self.grid_nodes)
        self.n_nodes = len(self.grid_nodes)

    def _create_collection_zones(self):
        """
        For each cluster head, identifies all valid grid points within the communication range.
        These points form the "collection zone" that the rover can travel to.
        """
        print("Creating collection zones for each cluster head...")
        self.ch_collection_zones = {}
        
        # The first node (index 0) is the base station, so we loop from 1 to n_nodes
        for node_idx in range(1, self.n_nodes):
            # Get the original (not grid-mapped) position of the cluster head
            ch_position = self.cluster_heads[node_idx - 1]
            
            # Find all grid points within the collection range
            distances = np.linalg.norm(self.grid_points - ch_position, axis=1)
            nearby_grid_indices = np.where(distances <= self.config.collection_range_limit)[0]
            
            if len(nearby_grid_indices) == 0:
                # Fallback: if no grid points are in range, use the single nearest grid point
                # This can happen if the CH is far from any valid path.
                nearest_grid_idx = self.node_to_grid_index[node_idx]
                self.ch_collection_zones[node_idx] = [nearest_grid_idx]
                print(f"  WARNING: No valid grid points in collection range for CH {node_idx-1}. Using nearest point only.")
            else:
                self.ch_collection_zones[node_idx] = nearby_grid_indices.tolist()

        print("Collection zones created.")

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
        print(f"Starting Grid-Based ACO optimization with {self.config.collection_n_ants} ants, {self.config.collection_n_iterations} iterations")
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
        
        print(f"Grid-Based ACO optimization completed. Best path length: {self.best_distance:.2f}m")
        return self.best_path, self.best_distance, self.convergence_history

    def get_full_path_for_plotting(self, path: List[int]) -> np.ndarray:
        """
        Converts a path of node indices into a detailed list of XY coordinates,
        including all intermediate waypoints generated by A* pathfinding.
        """
        if not path:
            return np.array([])

        full_path_coords = []
        
        # Add the starting point
        full_path_coords.append(self.grid_nodes[path[0]])

        # Iterate through each segment of the path
        for i in range(len(path) - 1):
            start_node_idx = path[i]
            end_node_idx = path[i + 1]
            
            # Get the detailed path between these two nodes using A*
            segment_path = self._find_grid_path(start_node_idx, end_node_idx)
            
            # Add the waypoints to the full path, skipping the first point to avoid duplication
            if len(segment_path) > 1:
                for j in range(1, len(segment_path)):
                    grid_idx = segment_path[j]
                    full_path_coords.append(self.grid_points[grid_idx])

        return np.array(full_path_coords)

    # =========================================================================
    # 3. Core ACO Algorithm Logic
    # =========================================================================

    def _construct_ant_solution(self) -> Tuple[List[int], float]:
        """
        Constructs a complete tour for a single ant using grid-based movement.
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
                next_node = random.choice(available_nodes)  # Fallback
            
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
        Performs a local pheromone update after each ant completes its tour.
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
        Performs a global pheromone update based on the best path found so far.
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
        Calculates the heuristic value for moving from node i to node j.
        """
        if i == j:
            return 0.0
        
        distance = self.distance_matrix[i, j]
        epsilon = 1e-9  # A very small number to prevent division by zero

        if not isinstance(distance, (int, float)):
             distance = float(distance.item()) if distance.size == 1 else float(distance.flat[0])
        
        return self.config.collection_Q / (distance + epsilon)


    # =========================================================================
    # 4. Grid-Based Distance Calculation & Pathfinding
    # =========================================================================

# In algorithms/ant_colony.py

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculates the distance matrix between all pairs of nodes.
        - For cluster heads, distance is calculated to the CLOSEST point in their collection zone.
        - For the base station, distance is to its single mapped grid point.
        """
        print("Calculating distance matrix with collection zones... (This may take a moment)")
        n = self.n_nodes
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Get the grid index for the start node
                start_grid_idx = self.node_to_grid_index[i]

                # Determine the target grid indices for the end node
                if j == 0: # If the destination is the base station
                    target_grid_indices = [self.node_to_grid_index[j]]
                else: # If the destination is a cluster head
                    target_grid_indices = self.ch_collection_zones[j]

                # Find the shortest path from the start node to ANY of the target indices
                min_dist_to_zone = float('inf')
                for target_idx in target_grid_indices:
                    path = self._astar_pathfinding(start_grid_idx, target_idx)
                    dist = self._calculate_path_distance(path)
                    if dist < min_dist_to_zone:
                        min_dist_to_zone = dist
                
                # Store the minimum distance found
                distances[i, j] = min_dist_to_zone

                # Get start and target indices for the reverse path
                start_grid_idx_rev = self.node_to_grid_index[j]
                if i == 0: # If the new destination is the base station
                    target_grid_indices_rev = [self.node_to_grid_index[i]]
                else: # If the new destination is a cluster head
                    target_grid_indices_rev = self.ch_collection_zones[i]

                min_dist_to_zone_rev = float('inf')
                for target_idx in target_grid_indices_rev:
                    path = self._astar_pathfinding(start_grid_idx_rev, target_idx)
                    dist = self._calculate_path_distance(path)
                    if dist < min_dist_to_zone_rev:
                        min_dist_to_zone_rev = dist
                
                distances[j, i] = min_dist_to_zone_rev

            if i % 2 == 0:
                print(f"  Processed distances for node {i}/{n-1}")

        print("Distance matrix calculation complete.")
        return distances

    def _find_grid_path(self, start_node_idx: int, end_node_idx: int) -> List[int]:
        """
        Uses A* algorithm to find the shortest path between two nodes on the grid.
        Returns a list of grid point indices representing the path.
        """
        start_grid_idx = self.node_to_grid_index[start_node_idx]
        end_grid_idx = self.node_to_grid_index[end_node_idx]
        
        return self._astar_pathfinding(start_grid_idx, end_grid_idx)

    def _astar_pathfinding(self, start_idx: int, goal_idx: int) -> List[int]:
        """
        A* pathfinding algorithm implementation for grid-based navigation.
        """
        if start_idx == goal_idx:
            return [start_idx]
        
        # Initialize data structures
        open_set = [start_idx]
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self._heuristic_distance(start_idx, goal_idx)}
        
        while open_set:
            # Find the node with the lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal_idx:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                return path[::-1]
            
            open_set.remove(current)
            
            # Check all neighbors (4-directional movement only)
            for neighbor_idx in self._get_grid_neighbors(current):
                tentative_g_score = g_score[current] + self.grid_resolution
                
                if neighbor_idx not in g_score or tentative_g_score < g_score[neighbor_idx]:
                    came_from[neighbor_idx] = current
                    g_score[neighbor_idx] = tentative_g_score
                    f_score[neighbor_idx] = tentative_g_score + self._heuristic_distance(neighbor_idx, goal_idx)
                    
                    if neighbor_idx not in open_set:
                        open_set.append(neighbor_idx)
        
        # No path found, return direct path (should not happen in a well-designed grid)
        return [start_idx, goal_idx]

    def _get_grid_neighbors(self, grid_idx: int) -> List[int]:
        """
        Returns the valid neighboring grid points for 4-directional movement.
        """
        current_point = self.grid_points[grid_idx]
        neighbors = []
        
        # Define 4-directional movements (up, down, left, right)
        directions = [
            np.array([0, self.grid_resolution]),   # up
            np.array([0, -self.grid_resolution]),  # down
            np.array([-self.grid_resolution, 0]),  # left
            np.array([self.grid_resolution, 0])    # right
        ]
        
        for direction in directions:
            neighbor_point = current_point + direction
            neighbor_key = (round(neighbor_point[0], 2), round(neighbor_point[1], 2))
            
            if neighbor_key in self.grid_coord_to_index:
                neighbors.append(self.grid_coord_to_index[neighbor_key])
        
        return neighbors

    def _heuristic_distance(self, idx1: int, idx2: int) -> float:
        """
        Manhattan distance heuristic for A* pathfinding.
        """
        point1 = self.grid_points[idx1]
        point2 = self.grid_points[idx2]
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def _calculate_path_distance(self, path: List[int]) -> float:
        """
        Calculates the total distance of a path given as a list of grid indices.
        """
        if len(path) <= 1:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            point1 = self.grid_points[path[i]]
            point2 = self.grid_points[path[i + 1]]
            total_distance += np.linalg.norm(point2 - point1)
        
        return total_distance

    # =========================================================================
    # 5. Utility Methods
    # =========================================================================

    def get_grid_points(self) -> np.ndarray:
        """
        Returns the valid grid points for visualization.
        """
        return self.grid_points

    def get_node_positions(self) -> np.ndarray:
        """
        Returns the positions of the mapped nodes (base station + cluster heads).
        """
        return self.grid_nodes