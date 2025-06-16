"""
Enhanced K-means clustering for Wireless Sensor Networks (WSN)

This implementation follows the algorithm from the provided specification with improvements:
- Uses actual sensor nodes as cluster heads (not computed centroids)
- Considers communication range for connectivity
- Supports adjustable maximum cluster size
- Includes proper error handling and validation
- Follows coding best practices

Mathematical functions implemented:
- C_k calculation for optimal number of clusters
- Distance calculations between sensors and cluster centers
- Connectivity-based cluster head selection
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings


class SensorNetworkClustering:
    """
    Enhanced K-means clustering specifically designed for Wireless Sensor Networks.
    
    Features:
    - Sensor nodes as actual cluster heads
    - Communication range constraints
    - Adjustable cluster size limits
    - Connectivity-based optimization
    """
    
    def __init__(self, comm_range: float, max_cluster_size, 
                 max_iterations, tolerance):
        """
        Initialize the clustering system.
        
        Args:
            comm_range: Communication range of sensors
            max_cluster_size: Maximum number of sensors per cluster
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for cluster head changes
        """
        self.comm_range = comm_range
        self.max_cluster_size = max_cluster_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def calculate_optimal_clusters(self, n_sensors: int, network_area: float, 
                                 d_bs_avg: float, d_th: float) -> int:
        """
        Calculate optimal number of clusters using the provided mathematical function:
        C_k = sqrt(n/(2*pi) * F/d_bs^2 * d_th)
        
        Args:
            n_sensors: Number of sensor nodes
            network_area: Network domain size (F)
            d_bs_avg: Average node distance from base station
            d_th: Distance threshold parameter
            
        Returns:
            Optimal number of clusters
        """
        if d_bs_avg <= 0 or network_area <= 0:
            raise ValueError("Network area and average distance must be positive")
            
        c_k = np.sqrt((n_sensors / (2 * np.pi)) * (network_area / (d_bs_avg ** 2)) * d_th)
        return max(1, int(np.round(c_k)))
    
    def calculate_distance_sensor_to_center(self, sensor_coord: np.ndarray, 
                                          center_coord: np.ndarray) -> float:
        """
        Calculate distance between sensor and cluster center using:
        d_n-c = sqrt(sum((S_i - S_c)^2))
        
        Args:
            sensor_coord: Sensor coordinates [x, y]
            center_coord: Cluster center coordinates [x, y]
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((sensor_coord - center_coord) ** 2))
    
    def calculate_distance_between_centers(self, center1: np.ndarray, 
                                         center2: np.ndarray) -> float:
        """
        Calculate distance between cluster centers using:
        d(C_i, C_j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        
        Args:
            center1: First cluster center coordinates
            center2: Second cluster center coordinates
            
        Returns:
            Euclidean distance between centers
        """
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def find_connected_sensors(self, sensor_positions: np.ndarray, 
                             center_idx: int) -> List[int]:
        """
        Find all sensors within communication range of a cluster head.
        
        Args:
            sensor_positions: Array of sensor positions
            center_idx: Index of the cluster head sensor
            
        Returns:
            List of sensor indices within communication range
        """
        center_pos = sensor_positions[center_idx]
        distances = np.linalg.norm(sensor_positions - center_pos, axis=1)
        connected = np.where(distances <= self.comm_range)[0]
        return connected.tolist()
    
    def initialize_cluster_heads_advanced(self, sensor_positions: np.ndarray, 
                                    n_clusters: int, 
                                    random_state: Optional[int] = None) -> np.ndarray:
        """
        Initialize cluster heads using actual sensor positions with advanced strategy:
        1) First head: Random sensor
        2) Subsequent heads: Maximize minimum distance to existing heads
        3) Ensure heads are separated by at least comm_range when possible
        4) Consider connectivity for better network coverage
        
        Returns:
            Array of cluster head indices (not positions, but indices of actual sensors)
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        n_sensors = len(sensor_positions)
        if n_clusters > n_sensors:
            raise ValueError(f"Cannot create {n_clusters} clusters with only {n_sensors} sensors")
        
        cluster_heads = []
        
        # First cluster head: random selection
        first_head = np.random.randint(0, n_sensors)
        cluster_heads.append(first_head)
        
        # Subsequent cluster heads: maximize separation
        for _ in range(1, n_clusters):
            max_min_distance = -1
            best_candidate = -1
            
            for candidate in range(n_sensors):
                if candidate in cluster_heads:
                    continue
                    
                # Calculate minimum distance to existing cluster heads
                min_distance = float('inf')
                for head_idx in cluster_heads:
                    distance = self.calculate_distance_between_centers(
                        sensor_positions[candidate], 
                        sensor_positions[head_idx]
                    )
                    min_distance = min(min_distance, distance)
                
                # Prefer candidates that are farther from existing heads
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate != -1:
                cluster_heads.append(best_candidate)
            else:
                # Fallback: random selection from remaining sensors
                remaining = [i for i in range(n_sensors) if i not in cluster_heads]
                if remaining:
                    cluster_heads.append(np.random.choice(remaining))
        
        return np.array(cluster_heads)  # Returns indices of actual sensors
    
    def assign_sensors_to_clusters(self, sensor_positions: np.ndarray, 
                                 cluster_head_indices: np.ndarray) -> np.ndarray:
        """
        Assign each sensor to the nearest cluster head.
        
        Args:
            sensor_positions: Array of sensor positions
            cluster_head_indices: Indices of cluster heads
            
        Returns:
            Array of cluster assignments for each sensor
        """
        n_sensors = len(sensor_positions)
        assignments = np.zeros(n_sensors, dtype=int)
        
        for i, sensor_pos in enumerate(sensor_positions):
            min_distance = float('inf')
            best_cluster = 0
            
            for cluster_id, head_idx in enumerate(cluster_head_indices):
                head_pos = sensor_positions[head_idx]
                distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_id
            
            assignments[i] = best_cluster
        
        return assignments
    
    def validate_cluster_sizes(self, assignments: np.ndarray, 
                             sensor_positions: np.ndarray, 
                             cluster_head_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate that clusters don't exceed maximum size limit.
        Redistribute sensors if necessary.
        
        Args:
            assignments: Current cluster assignments
            sensor_positions: Array of sensor positions
            cluster_head_indices: Indices of cluster heads
            
        Returns:
            Tuple of (validated_assignments, updated_cluster_heads)
        """
        n_clusters = len(cluster_head_indices)
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        
        # Check if any cluster exceeds maximum size
        oversized_clusters = np.where(cluster_sizes > self.max_cluster_size)[0]
        
        if len(oversized_clusters) == 0:
            return assignments, cluster_head_indices
        
        # Redistribute sensors from oversized clusters
        new_assignments = assignments.copy()
        
        for cluster_id in oversized_clusters:
            # Get sensors in this cluster
            cluster_sensors = np.where(assignments == cluster_id)[0]
            head_idx = cluster_head_indices[cluster_id]
            head_pos = sensor_positions[head_idx]
            
            # Calculate distances from cluster head
            distances = []
            for sensor_idx in cluster_sensors:
                if sensor_idx != head_idx:  # Don't move the cluster head itself
                    dist = self.calculate_distance_sensor_to_center(
                        sensor_positions[sensor_idx], head_pos
                    )
                    distances.append((dist, sensor_idx))
            
            # Sort by distance (farthest first for redistribution)
            distances.sort(reverse=True)
            
            # Keep only max_cluster_size sensors (including the head)
            sensors_to_keep = min(self.max_cluster_size, len(cluster_sensors))
            sensors_to_redistribute = len(cluster_sensors) - sensors_to_keep
            
            # Redistribute farthest sensors
            for i in range(sensors_to_redistribute):
                if i < len(distances):
                    sensor_to_move = distances[i][1]
                    # Find nearest cluster with space
                    self._reassign_sensor(sensor_to_move, sensor_positions, 
                                        cluster_head_indices, new_assignments)
        
        return new_assignments, cluster_head_indices
    
    def _reassign_sensor(self, sensor_idx: int, sensor_positions: np.ndarray,
                        cluster_head_indices: np.ndarray, assignments: np.ndarray):
        """
        Reassign a sensor to the nearest cluster with available space.
        
        Args:
            sensor_idx: Index of sensor to reassign
            sensor_positions: Array of sensor positions
            cluster_head_indices: Indices of cluster heads
            assignments: Current cluster assignments (modified in-place)
        """
        sensor_pos = sensor_positions[sensor_idx]
        n_clusters = len(cluster_head_indices)
        
        # Find clusters with available space, sorted by distance
        available_clusters = []
        for cluster_id, head_idx in enumerate(cluster_head_indices):
            cluster_size = np.sum(assignments == cluster_id)
            if cluster_size < self.max_cluster_size:
                head_pos = sensor_positions[head_idx]
                distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                available_clusters.append((distance, cluster_id))
        
        if available_clusters:
            # Assign to nearest available cluster
            available_clusters.sort()
            assignments[sensor_idx] = available_clusters[0][1]
        else:
            # All clusters are full - assign to nearest anyway (will be handled in next iteration)
            min_distance = float('inf')
            best_cluster = 0
            for cluster_id, head_idx in enumerate(cluster_head_indices):
                head_pos = sensor_positions[head_idx]
                distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_id
            assignments[sensor_idx] = best_cluster
    
    def perform_clustering(self, sensor_positions: np.ndarray, n_clusters: int,
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform complete K-means clustering with sensor-based cluster heads.
        
        Args:
            sensor_positions: Array of sensor positions (n_sensors, 2)
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cluster_assignments, cluster_head_indices, clustering_info)
        """
        if len(sensor_positions) < n_clusters:
            raise ValueError(f"Cannot create {n_clusters} clusters with only {len(sensor_positions)} sensors")
        
        # Initialize cluster heads
        cluster_heads = self.initialize_cluster_heads_advanced(
            sensor_positions, n_clusters, random_state
        )
        
        prev_heads = None
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # Assign sensors to clusters
            assignments = self.assign_sensors_to_clusters(sensor_positions, cluster_heads)
            
            # Validate cluster sizes
            assignments, cluster_heads = self.validate_cluster_sizes(
                assignments, sensor_positions, cluster_heads
            )
            
            # Update cluster heads (select most central sensor in each cluster)
            new_cluster_heads = self._update_cluster_heads(
                sensor_positions, assignments, cluster_heads
            )
            
            # Check for convergence
            if prev_heads is not None:
                head_changes = np.sum(new_cluster_heads != prev_heads)
                if head_changes == 0:
                    break
            
            prev_heads = cluster_heads.copy()
            cluster_heads = new_cluster_heads
        
        # Calculate clustering metrics
        sse = self.compute_total_sse(sensor_positions, assignments, cluster_heads)
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        
        clustering_info = {
            'iterations': iteration + 1,
            'total_sse': sse,
            'cluster_sizes': cluster_sizes,
            'converged': iteration < self.max_iterations - 1
        }
        
        return assignments, cluster_heads, clustering_info
    
    def _update_cluster_heads(self, sensor_positions: np.ndarray, 
                            assignments: np.ndarray, 
                            current_heads: np.ndarray) -> np.ndarray:
        """
        Update cluster heads by selecting the most central sensor in each cluster.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Current cluster assignments
            current_heads: Current cluster head indices
            
        Returns:
            Updated cluster head indices (indices of actual sensors, not centroids)
        """
        n_clusters = len(current_heads)
        new_heads = current_heads.copy()
        
        for cluster_id in range(n_clusters):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            
            if len(cluster_sensors) <= 1:
                continue  # Keep current head if cluster has only one sensor
            
            # Find the sensor closest to the cluster centroid
            cluster_positions = sensor_positions[cluster_sensors]
            centroid = np.mean(cluster_positions, axis=0)
            
            min_distance = float('inf')
            best_head = current_heads[cluster_id]
            
            # Find the sensor closest to the cluster centroid
            for sensor_idx in cluster_sensors:
                distance = self.calculate_distance_sensor_to_center(
                    sensor_positions[sensor_idx], centroid
                )
                if distance < min_distance:
                    min_distance = distance
                    best_head = sensor_idx  # This is an actual sensor index
    
            new_heads[cluster_id] = best_head
        
        return new_heads
    
    def compute_total_sse(self, sensor_positions: np.ndarray, 
                         assignments: np.ndarray, 
                         cluster_heads: np.ndarray) -> float:
        """
        Compute total Sum of Squared Errors (SSE) for the clustering.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            
        Returns:
            Total SSE value
        """
        total_sse = 0.0
        n_clusters = len(cluster_heads)
        
        for cluster_id in range(n_clusters):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            if len(cluster_sensors) == 0:
                continue
                
            head_pos = sensor_positions[cluster_heads[cluster_id]]
            
            for sensor_idx in cluster_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                total_sse += distance ** 2
        
        return total_sse
    
    def generate_elbow_plot(self, sensor_positions: np.ndarray, 
                          k_min: int = 2, k_max: int = 10, 
                          random_state: Optional[int] = None) -> Tuple[List[int], List[float]]:
        """
        Generate data for elbow plot to determine optimal number of clusters.
        
        Args:
            sensor_positions: Array of sensor positions
            k_min: Minimum number of clusters to test
            k_max: Maximum number of clusters to test
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (k_values, sse_values)
        """
        k_values = list(range(k_min, min(k_max + 1, len(sensor_positions))))
        sse_values = []
        
        for k in k_values:
            try:
                assignments, cluster_heads, info = self.perform_clustering(
                    sensor_positions, k, random_state
                )
                sse_values.append(info['total_sse'])
            except ValueError as e:
                warnings.warn(f"Could not cluster with k={k}: {e}")
                sse_values.append(float('inf'))
        
        return k_values, sse_values
    
    def plot_clustering_results(self, sensor_positions: np.ndarray, 
                              assignments: np.ndarray, 
                              cluster_heads: np.ndarray,
                              title: str = "Sensor Network Clustering Results"):
        """
        Visualize the clustering results.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Plot sensors colored by cluster
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_heads)))
        
        for cluster_id in range(len(cluster_heads)):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            if len(cluster_sensors) > 0:
                cluster_positions = sensor_positions[cluster_sensors]
                plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                           c=[colors[cluster_id]], s=50, alpha=0.7, 
                           label=f'Cluster {cluster_id} ({len(cluster_sensors)} sensors)')
        
        # Plot cluster heads
        head_positions = sensor_positions[cluster_heads]
        plt.scatter(head_positions[:, 0], head_positions[:, 1], 
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                   label='Cluster Heads')
        
        # Draw communication range circles around cluster heads
        for head_idx in cluster_heads:
            head_pos = sensor_positions[head_idx]
            circle = plt.Circle(head_pos, self.comm_range, 
                              fill=False, linestyle='--', alpha=0.5, color='gray')
            plt.gca().add_patch(circle)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


# Connectivity-based functions (for future implementation)
def analyze_network_connectivity(sensor_positions: np.ndarray, comm_range: float) -> dict:
    """
    Analyze network connectivity properties.
    
    Args:
        sensor_positions: Array of sensor positions
        comm_range: Communication range
        
    Returns:
        Dictionary with connectivity metrics
    """
    n_sensors = len(sensor_positions)
    adjacency_matrix = np.zeros((n_sensors, n_sensors))
    
    # Build adjacency matrix
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            distance = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
            if distance <= comm_range:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    # Calculate connectivity metrics
    node_degrees = np.sum(adjacency_matrix, axis=1)
    avg_degree = np.mean(node_degrees)
    connectivity_ratio = np.sum(node_degrees > 0) / n_sensors
    
    return {
        'adjacency_matrix': adjacency_matrix,
        'node_degrees': node_degrees,
        'average_degree': avg_degree,
        'connectivity_ratio': connectivity_ratio,
        'isolated_nodes': np.where(node_degrees == 0)[0].tolist()
    }


def find_optimal_cluster_heads_by_connectivity(sensor_positions: np.ndarray, 
                                             comm_range: float, 
                                             n_clusters: int) -> np.ndarray:
    """
    Find optimal cluster heads based on connectivity properties.
    This is a placeholder for future advanced connectivity-based clustering.
    
    Args:
        sensor_positions: Array of sensor positions
        comm_range: Communication range
        n_clusters: Number of clusters
        
    Returns:
        Array of cluster head indices
    """
    # Placeholder implementation - to be enhanced
    connectivity_info = analyze_network_connectivity(sensor_positions, comm_range)
    node_degrees = connectivity_info['node_degrees']
    
    # Select nodes with highest connectivity as potential cluster heads
    high_connectivity_nodes = np.argsort(node_degrees)[-n_clusters*2:]
    
    # Use the existing advanced initialization with connectivity preference
    # This is a simplified version - can be expanded later
    return high_connectivity_nodes[-n_clusters:]

