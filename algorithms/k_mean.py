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
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import Tuple, Optional, List
import warnings
from configurations.config_file import OptimizationConfig


class KMeansClustering:
    """
    Enhanced K-means clustering specifically designed for Wireless Sensor Networks.
    
    Features:
    - Sensor nodes as actual cluster heads
    - Communication range constraints
    - Adjustable cluster size limits
    - Connectivity-based optimization
    """
    
    def __init__(self):
        """
        Initialize the clustering system.
        
        Args:
            comm_range: Communication range of sensors
            max_cluster_size: Maximum number of sensors per cluster
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for cluster head changes
        """
        self.config = OptimizationConfig()
        self.comm_range = self.config.deployment_comm_range
        self.max_cluster_size = self.config.clustering_max_cluster_size
        self.max_iterations = self.config.clustering_max_iterations
        self.tolerance = self.config.clustering_tolerance
        
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
    
    def initialize_cluster_heads_plus_plus(self, sensor_positions: np.ndarray, 
                                     n_clusters: int, 
                                     random_state: Optional[int] = None) -> np.ndarray:
        """
        Initialize cluster heads using K-means++ algorithm for better initial placement.
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        n_sensors = len(sensor_positions)
        if n_clusters > n_sensors:
            raise ValueError(f"Cannot create {n_clusters} clusters with only {n_sensors} sensors")
        
        cluster_heads = []
        
        # Step 1: Choose first center randomly
        first_head = np.random.randint(0, n_sensors)
        cluster_heads.append(first_head)
        
        # Step 2: Choose remaining centers using weighted probability
        for _ in range(1, n_clusters):
            # Calculate squared distances from each point to nearest existing center
            distances_squared = np.full(n_sensors, np.inf)
            
            for i in range(n_sensors):
                if i in cluster_heads:
                    distances_squared[i] = 0
                    continue
                    
                min_dist_sq = np.inf
                for head_idx in cluster_heads:
                    dist = np.linalg.norm(sensor_positions[i] - sensor_positions[head_idx])
                    min_dist_sq = min(min_dist_sq, dist ** 2)
                distances_squared[i] = min_dist_sq
            
            # Choose next center with probability proportional to squared distance
            available_indices = [i for i in range(n_sensors) if i not in cluster_heads]
            available_distances = distances_squared[available_indices]
            
            if np.sum(available_distances) > 0:
                probabilities = available_distances / np.sum(available_distances)
                next_head = np.random.choice(available_indices, p=probabilities)
            else:
                next_head = np.random.choice(available_indices)
            
            cluster_heads.append(next_head)
        
        return np.array(cluster_heads)
        
    def assign_sensors_to_clusters(self, sensor_positions: np.ndarray, 
                                        cluster_head_indices: np.ndarray,
                                        assignment_strategy: str = "fallback") -> Tuple[np.ndarray, dict]:
        """
        Advanced sensor assignment with multiple strategies for handling range constraints.
        
        Args:
            sensor_positions: Array of sensor positions
            cluster_head_indices: Indices of cluster heads
            assignment_strategy: Strategy for handling out-of-range sensors
                - "strict": Only assign if within range, mark others as unassigned (-1)
                - "fallback": Assign to nearest head even if out of range (with warning)
                - "adaptive": Try to reassign unreachable sensors to less loaded clusters
                
        Returns:
            Tuple of (assignments, connectivity_report)
        """
        n_sensors = len(sensor_positions)
        assignments = np.full(n_sensors, -1, dtype=int)  # -1 = unassigned
        
        connectivity_report = {
            'total_sensors': n_sensors,
            'assigned_sensors': 0,
            'unreachable_sensors': [],
            'cluster_loads': {},
            'connectivity_rate': 0.0
        }
        
        # First pass: Assign sensors within communication range
        for i, sensor_pos in enumerate(sensor_positions):
            reachable_options = []
            
            for cluster_id, head_idx in enumerate(cluster_head_indices):
                head_pos = sensor_positions[head_idx]
                distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                
                if distance <= self.comm_range:
                    reachable_options.append((cluster_id, distance))
            
            if reachable_options:
                # Assign to nearest reachable cluster head
                reachable_options.sort(key=lambda x: x[1])  # Sort by distance
                assignments[i] = reachable_options[0][0]
                connectivity_report['assigned_sensors'] += 1
            else:
                # Mark as unreachable
                connectivity_report['unreachable_sensors'].append(i)
        
        # Handle unreachable sensors based on strategy
        if connectivity_report['unreachable_sensors']:
            print(f"\n⚠️  CONNECTIVITY ISSUES DETECTED ⚠️")
            print(f"Strategy: {assignment_strategy}")
            
            if assignment_strategy == "strict":
                print(f"{len(connectivity_report['unreachable_sensors'])} sensors remain UNASSIGNED (out of range)")
                
            elif assignment_strategy == "fallback":
                print(f"Assigning {len(connectivity_report['unreachable_sensors'])} out-of-range sensors to nearest heads...")
                
                for sensor_idx in connectivity_report['unreachable_sensors']:
                    sensor_pos = sensor_positions[sensor_idx]
                    min_distance = float('inf')
                    nearest_cluster = 0
                    
                    for cluster_id, head_idx in enumerate(cluster_head_indices):
                        head_pos = sensor_positions[head_idx]
                        distance = self.calculate_distance_sensor_to_center(sensor_pos, head_pos)
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_cluster = cluster_id
                    
                    assignments[sensor_idx] = nearest_cluster
                    # print(f"  Sensor {sensor_idx}: Assigned to cluster {nearest_cluster} "
                    #     f"(distance: {min_distance:.2f}m > range: {self.comm_range:.2f}m)")
                    
            # elif assignment_strategy == "adaptive":
            #     print(f"Attempting adaptive reassignment for {len(connectivity_report['unreachable_sensors'])} sensors...")
            #     assignments = self._adaptive_reassignment(
            #         sensor_positions, cluster_head_indices, assignments, 
            #         connectivity_report['unreachable_sensors']
            #     )
        
        # Calculate final connectivity metrics
        assigned_count = np.sum(assignments >= 0)
        connectivity_report['assigned_sensors'] = assigned_count
        connectivity_report['connectivity_rate'] = (assigned_count / n_sensors) * 100
        
        # Calculate cluster loads
        for cluster_id in range(len(cluster_head_indices)):
            cluster_sensors = np.sum(assignments == cluster_id)
            connectivity_report['cluster_loads'][cluster_id] = cluster_sensors
        
        # Print summary
        print(f"\n📊 CONNECTIVITY SUMMARY:")
        print(f"  Total sensors: {n_sensors}")
        print(f"  Successfully assigned: {assigned_count} ({connectivity_report['connectivity_rate']:.1f}%)")
        print(f"  Communication range: {self.comm_range:.2f}m")
        print(f"  Cluster loads: {list(connectivity_report['cluster_loads'].values())}")
    
        return assignments, connectivity_report

    
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
    
    def perform_clustering(self, sensor_positions: np.ndarray, 
                      n_clusters: int = None,
                      use_hybrid: bool = True,
                      assignment_strategy: str = "fallback",
                      random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform complete K-means clustering with hybrid cluster calculation.
        
        Args:
            sensor_positions: Array of sensor positions (n_sensors, 2)
            n_clusters: Number of clusters (if None, will auto-calculate)
            use_hybrid: Whether to use hybrid calculation method
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cluster_assignments, cluster_head_indices, clustering_info)
        """
        calculation_info = {}
        
        if n_clusters is None:
            if use_hybrid:
                # Use hybrid method (both formula and elbow)
                n_clusters, calculation_info = self.calculate_optimal_clusters_hybrid(
                    sensor_positions, random_state=random_state
                )
            else:
                # Use formula method only
                network_area = self.config.environment_field_length * self.config.environment_field_width
                base_station_pos = np.array([
                    self.config.environment_field_length / 2,
                    self.config.environment_field_width / 2
                ])
                d_bs_avg = np.mean(np.linalg.norm(sensor_positions - base_station_pos, axis=1))
                
                n_clusters = self.calculate_optimal_clusters(
                    len(sensor_positions), network_area, d_bs_avg, self.config.deployment_comm_range
                )
                
                calculation_info = {
                    'k_formula': n_clusters,
                    'k_elbow': None,
                    'k_final': n_clusters,
                    'method_used': 'formula_only'
                }
        
        if len(sensor_positions) < n_clusters:
            raise ValueError(f"Cannot create {n_clusters} clusters with only {len(sensor_positions)} sensors")
        
        # Initialize cluster heads using K-means++
        cluster_heads = self.initialize_cluster_heads_plus_plus(
            sensor_positions, n_clusters, random_state
        )
        
        prev_heads = None
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # Assign sensors to clusters
            assignments, connectivity_report = self.assign_sensors_to_clusters(
                sensor_positions, cluster_heads, assignment_strategy
            )
            
            # Validate cluster sizes
            assignments, cluster_heads = self.validate_cluster_sizes(
                assignments, sensor_positions, cluster_heads
            )
            
            # Update cluster heads
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
            'converged': iteration < self.max_iterations - 1,
            'connectivity_report': connectivity_report,
            'communication_range': self.comm_range,
            'assignment_strategy': assignment_strategy
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
    
    def calculate_optimal_clusters_hybrid(self, sensor_positions: np.ndarray, 
                                    network_area: float = None,
                                    base_station_pos: np.ndarray = None,
                                    d_th: float= 1.0,
                                    use_elbow: bool = True,
                                    random_state: Optional[int] = None) -> Tuple[int, dict]:
        """
        Calculate optimal clusters using both formula and elbow methods.
        
        Args:
            sensor_positions: Array of sensor positions
            network_area: Network domain size (auto-calculated if None)
            base_station_pos: Base station position (auto-calculated if None)
            d_th: Distance threshold parameter
            use_elbow: Whether to use elbow method
            random_state: Random seed
            
        Returns:
            Tuple of (optimal_k, calculation_info)
        """
        n_sensors = len(sensor_positions)
        
        # Calculate network area if not provided
        if network_area is None:
            network_area = self.config.environment_field_length * self.config.environment_field_width
        
        # Calculate base station position if not provided
        if base_station_pos is None:
            base_station_pos = np.array([
                self.config.environment_field_length / 2,
                self.config.environment_field_width / 2
            ])
        
        # Method 1: Mathematical formula
        distances_to_bs = np.linalg.norm(sensor_positions - base_station_pos, axis=1)
        d_bs_avg = np.mean(distances_to_bs)
        
        optimal_k_formula = self.calculate_optimal_clusters(
            n_sensors, network_area, d_bs_avg, d_th
        )
        
        # Method 2: Elbow method (if enough sensors and requested)
        optimal_k_elbow = None
        elbow_info = None
        
        if use_elbow and n_sensors >= 6:
            try:
                k_max = min(8, n_sensors // 2)
                optimal_k_elbow, k_values, sse_values = self.calculate_optimal_clusters_elbow(
                    sensor_positions, random_state=random_state
                )
                elbow_info = {
                    'k_values': k_values,
                    'sse_values': sse_values,
                    'method': 'elbow'
                }
            except Exception as e:
                print(f"Elbow method failed: {e}")
                optimal_k_elbow = None
        
        # Decision logic: Combine both methods
        if optimal_k_elbow is not None:
            # Both methods available - use hybrid approach
            difference = abs(optimal_k_formula - optimal_k_elbow)
            
            if difference <= 1:
                # Methods agree (within 1) - use elbow method
                final_k = optimal_k_elbow
                method_used = "elbow_preferred"
            elif difference <= 2:
                # Small difference - use average
                final_k = int((optimal_k_formula + optimal_k_elbow) / 2)
                method_used = "hybrid_average"
            else:
                # Large difference - prefer formula but limit by elbow
                if optimal_k_formula > optimal_k_elbow * 1.5:
                    final_k = int(((optimal_k_formula + optimal_k_elbow) / 2) + 1)
                    method_used = "hybrid_average"  
                else:
                    final_k = optimal_k_formula
                    method_used = "formula_preferred"
        else:
            # Only formula available
            final_k = optimal_k_formula
            method_used = "formula_only"
        
        # Ensure reasonable bounds
        final_k = max(1, min(final_k, n_sensors))
        
        calculation_info = {
            'k_formula': optimal_k_formula,
            'k_elbow': optimal_k_elbow,
            'k_final': final_k,
            'method_used': method_used,
            'n_sensors': n_sensors,
            'network_area': network_area,
            'd_bs_avg': d_bs_avg,
            'elbow_info': elbow_info
        }
        
        print(f"=== HYBRID CLUSTER CALCULATION ===")
        print(f"Formula method: {optimal_k_formula} clusters")
        print(f"Elbow method: {optimal_k_elbow} clusters" if optimal_k_elbow else "Elbow method: Not available")
        print(f"Final decision: {final_k} clusters (using {method_used})")
        
        return final_k, calculation_info

    def calculate_optimal_clusters_elbow(self, sensor_positions: np.ndarray, 
                                   random_state: Optional[int] = None) -> Tuple[int, List[int], List[float]]:
        """
        Use elbow method to find the optimal number of clusters.
        
        Args:
            sensor_positions: Array of sensor positions
            k_min: Minimum number of clusters to test
            k_max: Maximum number of clusters to test
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (optimal_k, k_values_tested, sse_values)
        """
        k_min = self.config.clustering_min_num_cluster
        k_max = self.config.clustering_max_num_cluster
        k_max = min(k_max, len(sensor_positions) - 1)
        if k_max < k_min:
            raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
        
        k_values = list(range(k_min, k_max + 1))
        sse_values = []
        
        print(f"Testing cluster counts from {k_min} to {k_max}...")
        
        for k in k_values:
            # Perform clustering multiple times and take best result
            best_sse = float('inf')
            
            for trial in range(3):  # 3 trials per k value
                try:
                    # Use K-means++ initialization
                    cluster_heads = self.initialize_cluster_heads_plus_plus(
                        sensor_positions, k, random_state
                    )
                    
                    # Run limited iterations for elbow test
                    assignments = None
                    for iteration in range(10):  # Limited iterations for elbow test
                        new_assignments = self.assign_sensors_to_clusters(sensor_positions, cluster_heads)
                        
                        # Validate cluster sizes
                        new_assignments, cluster_heads = self.validate_cluster_sizes(
                            new_assignments, sensor_positions, cluster_heads
                        )
                        
                        # Update cluster heads
                        new_heads = self._update_cluster_heads(sensor_positions, new_assignments, cluster_heads)
                        
                        # Check for convergence
                        if assignments is not None and np.array_equal(new_heads, cluster_heads):
                            break
                        
                        cluster_heads = new_heads
                        assignments = new_assignments
                    
                    # Calculate SSE for this trial
                    if assignments is not None:
                        sse = self.compute_total_sse(sensor_positions, assignments, cluster_heads)
                        best_sse = min(best_sse, sse)
                    
                except Exception as e:
                    print(f"Warning: Trial {trial+1} failed for k={k}: {e}")
                    continue
            
            if best_sse != float('inf'):
                sse_values.append(best_sse)
                print(f"k={k}: SSE={best_sse:.2f}")
            else:
                print(f"k={k}: All trials failed")
                sse_values.append(float('inf'))
        
        # Find elbow point using rate of change
        optimal_k = self._find_elbow_point(k_values, sse_values)
        
        return optimal_k, k_values, sse_values

    def _find_elbow_point(self, k_values: List[int], sse_values: List[float]) -> int:
        """
        Find the elbow point in the SSE curve using the rate of change method.
        
        Args:
            k_values: List of k values tested
            sse_values: List of corresponding SSE values
            
        Returns:
            Optimal k value based on elbow method
        """
        if len(sse_values) < 3:
            print("Not enough data points for elbow method, returning minimum k")
            return k_values[0] if k_values else 2
        
        # Filter out infinite values
        valid_indices = [i for i, sse in enumerate(sse_values) if sse != float('inf')]
        
        if len(valid_indices) < 3:
            print("Not enough valid SSE values for elbow method")
            return k_values[0] if k_values else 2
        
        valid_k_values = [k_values[i] for i in valid_indices]
        valid_sse_values = [sse_values[i] for i in valid_indices]
        
        # Calculate rate of change (slope) between consecutive points
        slopes = []
        for i in range(1, len(valid_sse_values)):
            slope = (valid_sse_values[i-1] - valid_sse_values[i]) / (valid_k_values[i] - valid_k_values[i-1])
            slopes.append(slope)
        
        # Find the point where slope change is maximum (elbow point)
        max_slope_change = 0
        elbow_idx = 0
        
        for i in range(1, len(slopes)):
            slope_change = slopes[i-1] - slopes[i]
            if slope_change > max_slope_change:
                max_slope_change = slope_change
                elbow_idx = i
        
        # The elbow point is at index elbow_idx + 1 in valid_k_values
        optimal_k = valid_k_values[min(elbow_idx + 1, len(valid_k_values) - 1)]
        
        print(f"Elbow analysis:")
        print(f"  Tested k values: {valid_k_values}")
        print(f"  SSE values: {[f'{sse:.2f}' for sse in valid_sse_values]}")
        print(f"  Slopes: {[f'{slope:.2f}' for slope in slopes]}")
        print(f"  Elbow point suggests k = {optimal_k}")
        
        return optimal_k

    def _calculate_coverage_metrics(self, sensor_positions, assignments, cluster_heads):
        """Calculate coverage and connectivity metrics"""
        total_sensors = len(sensor_positions)
        total_clusters = len(cluster_heads)
        
        # Calculate cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=total_clusters)
        
        # Calculate coverage rate (simplified)
        connected_sensors = 0
        for cluster_id, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            cluster_sensors = np.where(assignments == cluster_id)[0]
            
            for sensor_idx in cluster_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                distance = np.linalg.norm(sensor_pos - head_pos)
                if distance <= self.comm_range:
                    connected_sensors += 1
        
        coverage_rate = (connected_sensors / total_sensors) * 100 if total_sensors > 0 else 0
        
        return {
            'coverage_rate': coverage_rate,
            'total_sensors': total_sensors,
            'total_clusters': total_clusters,
            'cluster_sizes': cluster_sizes,
            'connected_sensors': connected_sensors
        }

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

    def analyze_cluster_connectivity(self, sensor_positions: np.ndarray, 
                                assignments: np.ndarray, 
                                cluster_heads: np.ndarray) -> dict:
        """
        Analyze connectivity within clusters and between cluster heads.
        
        Returns:
            Dictionary with connectivity analysis
        """
        connectivity_results = {}
        
        for cluster_id, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            cluster_sensors = np.where(assignments == cluster_id)[0]
            
            # Analyze intra-cluster connectivity
            connected_to_head = 0
            isolated_sensors = []
            
            for sensor_idx in cluster_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                distance = np.linalg.norm(sensor_pos - head_pos)
                
                if distance <= self.comm_range:
                    connected_to_head += 1
                else:
                    isolated_sensors.append(sensor_idx)
            
            connectivity_results[cluster_id] = {
                'cluster_size': len(cluster_sensors),
                'connected_to_head': connected_to_head,
                'connectivity_rate': connected_to_head / len(cluster_sensors) * 100,
                'isolated_sensors': isolated_sensors,
                'head_position': head_pos
            }
        
        return connectivity_results

    def _plot_field_environment(self, ax, field_data):
        """Plot the field environment (beds, vegetables, etc.)"""
        try:
            # Plot beds
            if 'bed_coords' in field_data:
                bed_coords = field_data['bed_coords']
                for i, bed in enumerate(bed_coords):
                    if len(bed) >= 4:  # Ensure we have at least 4 points
                        # Create polygon for bed
                        bed_polygon = patches.Polygon(bed, closed=True, 
                                                    facecolor='lightblue', 
                                                    edgecolor='blue', 
                                                    alpha=0.3, linewidth=1)
                        ax.add_patch(bed_polygon)
            
            # Plot vegetables if available
            if 'vegetable_positions' in field_data:
                veg_positions = field_data['vegetable_positions']
                ax.scatter(veg_positions[:, 0], veg_positions[:, 1], 
                        c='green', s=20, alpha=0.6, marker='s', label='Vegetables')
            
            # Plot monitor location (base station)
            if 'monitor_location' in field_data:
                monitor_pos = field_data['monitor_location']
                ax.scatter(monitor_pos[0], monitor_pos[1], 
                        c='red', s=150, marker='s', 
                        edgecolors='black', linewidth=2, label='Monitor')
        
        except Exception as e:
            print(f"Warning: Could not plot field environment: {e}")
            
    def plot_clustering_results(self, sensor_positions: np.ndarray, 
                            assignments: np.ndarray, 
                            cluster_heads: np.ndarray,
                            title: str = "Traditional K-means Clustering Results",
                            field_data: dict = None):
        """
        Visualize the clustering results with field environment and connectivity ranges.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            title: Plot title
            field_data: Field environment data (beds, vegetables, etc.)
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot field environment if available
        if field_data is not None:
            self._plot_field_environment(ax, field_data)
        
        # Define colors for clusters
        colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        cluster_colors = {}
        
        # Plot communication range circles for cluster heads first (behind sensors)
        head_positions = sensor_positions[cluster_heads]
        for i, (head_idx, head_pos) in enumerate(zip(cluster_heads, head_positions)):
            cluster_id = assignments[head_idx]
            color = colors[cluster_id % len(colors)]
            cluster_colors[cluster_id] = color
            
            # Draw communication range circle
            circle = plt.Circle(head_pos, self.comm_range, 
                            fill=False, linestyle='-', alpha=0.6, 
                            color=color, linewidth=2)
            ax.add_patch(circle)
        
        # Plot sensors colored by cluster
        for cluster_id in range(len(cluster_heads)):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            if len(cluster_sensors) > 0:
                cluster_positions = sensor_positions[cluster_sensors]
                color = colors[cluster_id % len(colors)]
                
                # Plot regular sensors
                regular_sensors = [idx for idx in cluster_sensors if idx not in cluster_heads]
                if regular_sensors:
                    regular_positions = sensor_positions[regular_sensors]
                    ax.scatter(regular_positions[:, 0], regular_positions[:, 1], 
                            c=color, s=60, alpha=0.8, marker='o',
                            label=f'Cluster {cluster_id + 1} ({len(cluster_sensors)} sensors)')
        
        # Plot cluster heads with special markers
        for i, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            cluster_id = assignments[head_idx]
            color = colors[cluster_id % len(colors)]
            
            # Plot cluster head with star marker
            ax.scatter(head_pos[0], head_pos[1], 
                    c='black', s=200, marker='*', 
                    edgecolors=color, linewidth=3,
                    zorder=5)
        
        # Calculate and display metrics
        coverage_info = self._calculate_coverage_metrics(sensor_positions, assignments, cluster_heads)
        
        # Update title with metrics
        title_with_metrics = f"{title}\nCoverage: {coverage_info['coverage_rate']:.1f}%, " \
                            f"Sensors: {len(sensor_positions)}, Clusters: {len(cluster_heads)}"
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title_with_metrics, fontsize=14, fontweight='bold')
        
        # Create custom legend
        legend_elements = []
        
        # Add cluster legends
        for cluster_id in range(len(cluster_heads)):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            color = colors[cluster_id % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8,
                                            label=f'Cluster {cluster_id + 1} ({len(cluster_sensors)} sensors)'))
        
        # Add cluster heads legend
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor='black', markersize=12,
                                        label='Cluster Heads'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

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