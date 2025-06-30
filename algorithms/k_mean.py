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
    - Strict cluster size 
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
        self.k_min = self.config.clustering_min_num_cluster     

    def calculate_optimal_clusters_mobile_rover(self, sensor_positions: np.ndarray) -> int:
        """
        Calculate optimal clusters for mobile rover scenario.
        
        This method considers:
        - Sensor density and distribution
        - Communication range constraints
        - Energy efficiency for rover movement
        - Maximum cluster size constraints
        
        Args:
            sensor_positions: Array of sensor positions
            
        Returns:
            Optimal number of clusters for mobile rover
        """
        n_sensors = len(sensor_positions)
        
        # Method 1: Based on max cluster size constraint
        min_clusters_by_size = int(np.ceil(n_sensors / self.max_cluster_size))
        
        # Method 2: Based on communication range and sensor density
        # Calculate average inter-sensor distance
        distances = []
        for i in range(min(100, n_sensors)):  # Sample to avoid O(n²) complexity
            for j in range(i + 1, min(100, n_sensors)):
                dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
                distances.append(dist)
        
        avg_inter_sensor_dist = np.mean(distances) if distances else self.comm_range
        
        # Estimate clusters based on communication coverage
        # Each cluster can cover approximately π * comm_range²
        # But we need to consider overlap and connectivity
        effective_coverage_per_cluster = np.pi * (self.comm_range * 0.7) ** 2  # 70% efficiency
        
        # Calculate field area (rough estimate)
        x_range = np.max(sensor_positions[:, 0]) - np.min(sensor_positions[:, 0])
        y_range = np.max(sensor_positions[:, 1]) - np.min(sensor_positions[:, 1])
        field_area = x_range * y_range
        
        min_clusters_by_coverage = max(1, int(np.ceil(field_area / effective_coverage_per_cluster)))
        
        # Method 3: Based on sensor distribution (clustering tendency)
        # Use a simple heuristic based on sensor spread
        sensor_spread = np.std(sensor_positions, axis=0)
        spread_factor = np.mean(sensor_spread) / self.comm_range
        min_clusters_by_spread = max(1, int(np.ceil(spread_factor)))
        
        # Combine methods with weights
        optimal_k = int(np.ceil(
            0.4 * min_clusters_by_size +           # 40% weight on size constraint
            0.3 * min_clusters_by_coverage +       # 30% weight on coverage
            0.3 * min_clusters_by_spread           # 30% weight on distribution
        ))
        
        # Ensure reasonable bounds
        optimal_k = max(min_clusters_by_size, optimal_k)  # Must satisfy size constraint
        optimal_k = min(optimal_k, n_sensors)             # Cannot exceed sensor count
        
        print(f"=== MOBILE ROVER CLUSTER CALCULATION ===")
        print(f"Min clusters by size constraint: {min_clusters_by_size}")
        print(f"Min clusters by coverage: {min_clusters_by_coverage}")
        print(f"Min clusters by spread: {min_clusters_by_spread}")
        print(f"Final decision: {optimal_k} clusters")
        
        return optimal_k
  
    def calculate_optimal_clusters_elbow(self, sensor_positions: np.ndarray, 
                                    random_state: Optional[int] = None) -> int:
        """
        Calculate optimal number of clusters using the actual elbow method.
        This performs k-means clustering for different k values and finds the elbow point.
        
        Args:
            sensor_positions: Array of sensor positions
            random_state: Random seed for reproducibility
            
        Returns:
            Optimal number of clusters based on elbow method
        """
        n_sensors = len(sensor_positions)
        
        # Get configuration values
        k_max = min(self.config.clustering_max_num_cluster, n_sensors - 1)
        k_max = min(k_max, max(10, n_sensors // 2))
        
        # Ensure we have enough sensors and a valid range
        if k_max < self.k_min:
            print(f"Warning: Not enough sensors ({n_sensors}) or invalid k range [{self.k_min}, {k_max}]")
            return max(1, int(np.ceil(n_sensors / self.max_cluster_size)))
        
        try:
            # Use the existing elbow method implementation
            optimal_k, k_values, sse_values = self._calculate_euclidean_distance(
                sensor_positions
            )
            
            print(f"=== ACTUAL ELBOW METHOD CLUSTER CALCULATION ===")
            print(f"Tested k values: {k_values}")
            print(f"SSE values: {[f'{sse:.2f}' if sse != float('inf') else 'Failed' for sse in sse_values]}")
            print(f"Elbow method suggests: {optimal_k} clusters")
            
            return optimal_k
            
        except Exception as e:
            print(f"Elbow method failed: {e}")
            print("Falling back to heuristic calculation...")
            
            # Fallback to size-based calculation
            fallback_k = max(1, int(np.ceil(n_sensors / self.max_cluster_size)))
            print(f"Using fallback: {fallback_k} clusters")
            return fallback_k
        
    def _calculate_euclidean_distance(self, sensor_positions: np.ndarray, 
                                random_state: Optional[int] = None) -> Tuple[int, List[int], List[float]]:
        """
        Use elbow method to find the optimal number of clusters.
        
        Args:
            sensor_positions: Array of sensor positions
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (optimal_k, k_values_tested, sse_values)
        """
        n_sensors = len(sensor_positions)
        k_max = min(self.config.clustering_max_num_cluster, n_sensors - 1)
        
        # Adjust k_max to be reasonable for elbow method
        k_max = min(k_max, max(8, n_sensors // 3))  # Don't test too many clusters
        
        if k_max < self.k_min:
            raise ValueError(f"k_max ({k_max}) must be >= k_min ({self.k_min})")
        
        k_values = list(range(self.k_min, k_max + 1))
        sse_values = []
        
        print(f"Testing cluster counts from {self.k_min} to {k_max}...........................................................")
        
        for k in k_values:
            # Perform clustering multiple times and take best result
            sse_trials = []
            
            for trial in range(10): 
                try:
                    # Initialize cluster heads
                    cluster_heads = self.initialize_cluster_heads_plus_plus(
                        sensor_positions, k, random_state=(random_state + trial) if random_state else None
                    )
                    
                    # Run clustering until convergence or max iterations
                    assignments = None
                    prev_assignments = None
                    
                    for iteration in range(self.max_iterations):  # Use full max_iterations
                        new_assignments, _ = self.assign_sensors_to_clusters(
                            sensor_positions, cluster_heads
                        )
                        
                        # Check if all sensors are assigned
                        if np.sum(new_assignments == -1) > 0:
                            break
                        
                        # Check for convergence
                        if prev_assignments is not None and np.array_equal(new_assignments, prev_assignments):
                            assignments = new_assignments
                            break
                        
                        # Update cluster heads
                        new_heads = self._update_cluster_heads(sensor_positions, new_assignments, cluster_heads)
                        cluster_heads = new_heads
                        prev_assignments = new_assignments.copy()
                        assignments = new_assignments
                    
                    # Calculate SSE for this trial if converged
                    if assignments is not None and np.sum(assignments == -1) == 0:
                        sse = self.compute_total_sse(sensor_positions, assignments, cluster_heads)
                        sse_trials.append(sse)
                    
                except Exception as e:
                    print(f"Warning: Trial {trial+1} failed for k={k}: {e}")
                    continue
            
            if sse_trials:
                # Take average SSE instead of minimum for more stable results
                avg_sse = np.mean(sse_trials)
                sse_values.append(avg_sse)
                print(f"k={k}: Average SSE={avg_sse:.2f} (from {len(sse_trials)} successful trials)")
            else:
                print(f"k={k}: All trials failed")
                sse_values.append(float('inf'))
        
        # Find elbow point using corrected method
        optimal_k = self._find_elbow_point(k_values, sse_values)
        return optimal_k, k_values, sse_values

    def _find_elbow_point(self, k_values: List[int], sse_values: List[float]) -> int:
        """
        Find the elbow point using the knee detection algorithm (Kneedle algorithm simplified).
        
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
        
        valid_k_values = np.array([k_values[i] for i in valid_indices])
        valid_sse_values = np.array([sse_values[i] for i in valid_indices])
        
        # Normalize the data to [0, 1] range for better knee detection
        norm_k = (valid_k_values - valid_k_values.min()) / (valid_k_values.max() - valid_k_values.min())
        norm_sse = (valid_sse_values - valid_sse_values.min()) / (valid_sse_values.max() - valid_sse_values.min())
        
        # Calculate the distance from each point to the line connecting first and last points
        # This is the core of the knee detection algorithm
        distances = []
        
        # Line from first to last point: y = mx + b
        x1, y1 = norm_k[0], norm_sse[0]
        x2, y2 = norm_k[-1], norm_sse[-1]
        
        # Calculate perpendicular distance from each point to the line
        for i in range(len(norm_k)):
            x0, y0 = norm_k[i], norm_sse[i]
            
            # Distance from point to line formula: |ax + by + c| / sqrt(a² + b²)
            # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            
            distance = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
            distances.append(distance)
        
        # The elbow point is where the distance is maximum
        elbow_idx = np.argmax(distances)
        optimal_k = valid_k_values[elbow_idx]
        
        print(f"Corrected Elbow analysis:")
        print(f"  Tested k values: {valid_k_values.tolist()}")
        print(f"  SSE values: {[f'{sse:.2f}' for sse in valid_sse_values]}")
        print(f"  Distances from line: {[f'{d:.4f}' for d in distances]}")
        print(f"  Maximum distance at k = {optimal_k} (elbow point)")
        
        return int(optimal_k)

    def calculate_euclidean_distance(self, coord1: np.ndarray, 
                                          coord2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between sensor and cluster center.
        """
        return np.sqrt(np.sum((coord1 - coord2) ** 2))
    
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
                                cluster_head_indices: np.ndarray) -> Tuple[np.ndarray, dict]:
        """        
        Args:
            sensor_positions: Array of sensor positions
            cluster_head_indices: Indices of cluster heads
                
        Returns:
            Tuple of (assignments, connectivity_report)
        """
        n_sensors = len(sensor_positions)
        n_clusters = len(cluster_head_indices)
        assignments = np.full(n_sensors, -1, dtype=int)
        
        # Initialize cluster capacities
        cluster_capacities = np.full(n_clusters, self.max_cluster_size, dtype=int)
        
        # Step 1: Assign cluster heads to their own clusters
        for cluster_id, head_idx in enumerate(cluster_head_indices):
            assignments[head_idx] = cluster_id
            cluster_capacities[cluster_id] -= 1
        
        # Step 2: Create list of unassigned sensors
        unassigned_sensors = [i for i in range(n_sensors) if assignments[i] == -1]
        
        # Step 3: Assign sensors in order of preference (respecting capacity limits)
        assignment_attempts = 0
        max_attempts = self.config.clustering_max_iteration_assignments 
        
        while unassigned_sensors and assignment_attempts < max_attempts:
            assignment_attempts += 1
            newly_assigned = []
            
            for sensor_idx in unassigned_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                
                # Find all clusters with available capacity
                available_clusters = []
                for cluster_id, head_idx in enumerate(cluster_head_indices):
                    if cluster_capacities[cluster_id] > 0:
                        head_pos = sensor_positions[head_idx]
                        distance = np.linalg.norm(sensor_pos - head_pos)  # Euclidean distance
                        available_clusters.append((distance, cluster_id))
                
                # Assign to nearest available cluster
                if available_clusters:
                    available_clusters.sort()  # Sort by distance
                    _, nearest_cluster = available_clusters[0]
                    
                    assignments[sensor_idx] = nearest_cluster
                    cluster_capacities[nearest_cluster] -= 1
                    newly_assigned.append(sensor_idx)
            
            # Remove newly assigned sensors from unassigned list
            for sensor_idx in newly_assigned:
                unassigned_sensors.remove(sensor_idx)
        
        # Step 4: FORCE assign remaining unassigned sensors to nearest cluster (ignoring capacity)
        if unassigned_sensors:
            # print(f"⚠️  Force assigning {len(unassigned_sensors)} sensors to nearest clusters (ignoring capacity limits)")
            
            for sensor_idx in unassigned_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                
                # Find nearest cluster regardless of capacity
                nearest_distances = []
                for cluster_id, head_idx in enumerate(cluster_head_indices):
                    head_pos = sensor_positions[head_idx]
                    distance = np.linalg.norm(sensor_pos - head_pos)  # Euclidean distance
                    nearest_distances.append((distance, cluster_id))
                
                # Sort by distance and assign to nearest
                nearest_distances.sort()
                _, nearest_cluster = nearest_distances[0]
                
                assignments[sensor_idx] = nearest_cluster
        
        # Calculate final cluster sizes for reporting
        final_cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        
        # Report final cluster sizes and any capacity violations
        capacity_violations = []
        for cluster_id, size in enumerate(final_cluster_sizes):
            if size > self.max_cluster_size:
                capacity_violations.append((cluster_id, size, self.max_cluster_size))
        
        if capacity_violations:
            # print(f"⚠️  Capacity violations after force assignment:")
            for cluster_id, actual_size, max_size in capacity_violations:
                # print(f"   Cluster {cluster_id}: {actual_size} sensors (limit: {max_size})")
                pass
        
        # Calculate connectivity report
        connectivity_report = self._calculate_connectivity_report(
            sensor_positions, assignments, cluster_head_indices
        )
        
        return assignments, connectivity_report

    def _calculate_connectivity_report(self, sensor_positions: np.ndarray, 
                                     assignments: np.ndarray, 
                                     cluster_head_indices: np.ndarray) -> dict:
        """Calculate detailed connectivity report."""
        n_sensors = len(sensor_positions)
        n_clusters = len(cluster_head_indices)
        
        within_range_count = 0
        out_of_range_sensors = []
        cluster_loads = {}
        
        for cluster_id, head_idx in enumerate(cluster_head_indices):
            head_pos = sensor_positions[head_idx]
            cluster_sensors = np.where(assignments == cluster_id)[0]
            cluster_loads[cluster_id] = len(cluster_sensors)
            
            for sensor_idx in cluster_sensors:
                sensor_pos = sensor_positions[sensor_idx]
                distance = self.calculate_euclidean_distance(sensor_pos, head_pos)
                
                if distance <= self.comm_range:
                    within_range_count += 1
                else:
                    out_of_range_sensors.append(sensor_idx)
        
        connectivity_rate = (within_range_count / n_sensors) * 100 if n_sensors > 0 else 0
        
        return {
            'total_sensors': n_sensors,
            'within_range_sensors': within_range_count,
            'out_of_range_sensors': out_of_range_sensors,
            'connectivity_rate': connectivity_rate,
            'cluster_loads': cluster_loads
        }
        
    def perform_clustering(self, sensor_positions: np.ndarray, 
                               n_clusters: int = None,
                               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform complete K-means clustering with proper error handling.
        
        Args:
            sensor_positions: Array of sensor positions (n_sensors, 2)
            n_clusters: Number of clusters (if None, will auto-calculate)
            use_mobile_rover_calc: Whether to use mobile rover calculation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cluster_assignments, cluster_head_indices, clustering_info)
        """
        n_sensors = len(sensor_positions)
        
        # Calculate optimal clusters
        if n_clusters is None:
            if self.config.clustering_use_mobile_rover_calc:
                n_clusters = self.calculate_optimal_clusters_mobile_rover(sensor_positions)
            else:
                # use the elbow method to determine the optimal number of clusters
                n_clusters = self.calculate_optimal_clusters_elbow(sensor_positions, random_state=random_state)
        
        restart_count = 0
        max_restarts = self.config.clustering_max_restart_attempts 
        
        while restart_count <= max_restarts:
            try:
                print(f"\n=== CLUSTERING ATTEMPT {restart_count + 1} ===")
                print(f"Trying with {n_clusters} clusters...")
                
                # Initialize cluster heads
                cluster_heads = self.initialize_cluster_heads_plus_plus(
                    sensor_positions, n_clusters, random_state
                )
                
                # Perform clustering iterations
                for iteration in range(self.max_iterations):
                    # Assign sensors to clusters with strict size limits
                    assignments, connectivity_report = self.assign_sensors_to_clusters(
                        sensor_positions, cluster_heads
                    )
                    
                    # Check if all sensors are assigned
                    unassigned_count = np.sum(assignments == -1)
                    if unassigned_count > 0:
                        print(f"⚠️  {unassigned_count} sensors unassigned")
                        break
                    
                    # Check cluster size violations
                    cluster_sizes = np.bincount(assignments, minlength=n_clusters)
                    max_cluster_size = np.max(cluster_sizes)
                    
                    if max_cluster_size > self.max_cluster_size:
                        print(f"⚠️  Cluster size violation: max={max_cluster_size}, limit={self.max_cluster_size}")
                        break
                    
                    # Update cluster heads
                    new_cluster_heads = self._update_cluster_heads(
                        sensor_positions, assignments, cluster_heads
                    )
                    
                    # Check for convergence
                    if np.array_equal(new_cluster_heads, cluster_heads):
                        print(f"✅ Converged after {iteration + 1} iterations")
                        
                        # Final validation
                        final_assignments, final_report = self.assign_sensors_to_clusters(
                            sensor_positions, new_cluster_heads
                        )
                        
                        # Calculate final metrics
                        sse = self.compute_total_sse(sensor_positions, final_assignments, new_cluster_heads)
                        
                        clustering_info = {
                            'iterations': iteration + 1,
                            'total_sse': sse,
                            'cluster_sizes': np.bincount(final_assignments, minlength=n_clusters),
                            'converged': True,
                            'connectivity_report': final_report,
                            'n_clusters_final': n_clusters,
                            'restart_count': restart_count
                        }
                        
                        print(f"✅ CLUSTERING SUCCESSFUL!")
                        print(f"Final cluster sizes: {clustering_info['cluster_sizes']}")
                        print(f"Connectivity rate: {final_report['connectivity_rate']:.1f}%")
                        
                        return final_assignments, new_cluster_heads, clustering_info
                    
                    cluster_heads = new_cluster_heads
                
                # If we reach here, clustering failed for this n_clusters
                print(f"❌ Clustering failed with {n_clusters} clusters")
                
            except Exception as e:
                print(f"❌ Error in clustering attempt: {e}")
            
            # Increase cluster count and retry
            restart_count += 1
            if restart_count <= max_restarts:
                n_clusters += 1
                print(f"🔄 Restarting with {n_clusters} clusters...")
        
        # If all attempts failed, return fallback assignment instead of None
        print(f"\n⚠️  ========== CLUSTERING IMPOSSIBLE ==========")
        print(f"❌ CRITICAL: Unable to cluster {n_sensors} sensors after {max_restarts + 1} attempts")
        print(f"❌ REASON: Clustering constraints cannot be satisfied")
        print(f"❌ CONSTRAINTS:")
        print(f"   - Max cluster size: {self.max_cluster_size}")
        print(f"   - Communication range: {self.comm_range}")
        print(f"   - Max iterations: {self.max_iterations}")
        print(f"❌ RECOMMENDATION: Consider adjusting:")
        print(f"   - Increase max_cluster_size")
        print(f"   - Increase communication range")
        print(f"   - Reduce sensor density")
        print(f"   - Use different clustering algorithm")
        print(f"⚠️  ============================================")
        
        # Return fallback assignment where each sensor is its own cluster
        fallback_assignments = np.arange(n_sensors)  # Each sensor is its own cluster
        fallback_heads = np.arange(n_sensors)        # Each sensor is a cluster head
        fallback_info = {
            'iterations': 0,
            'total_sse': float('inf'),
            'cluster_sizes': np.ones(n_sensors, dtype=int),  # Each cluster has 1 sensor
            'converged': False,
            'connectivity_report': {
                'connectivity_rate': 100.0,  # Each sensor connected to itself
                'isolated_sensors': [],
                'avg_cluster_connectivity': 100.0
            },
            'n_clusters_final': n_sensors,
            'restart_count': restart_count,
            'clustering_failed': True,
            'fallback_used': True
        }
        
        print(f"🔧 FALLBACK: Using individual sensor assignment ({n_sensors} clusters)")
        
        return fallback_assignments, fallback_heads, fallback_info
    
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
                distance = self.calculate_euclidean_distance(
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
                distance = self.calculate_euclidean_distance(sensor_pos, head_pos)
                total_sse += distance ** 2
        
        return total_sse
    
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

    def analyze_cluster_connectivity(self, sensor_positions: np.ndarray, # NOT USED ----------------------------
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
            
    def plot_clustering_results(self, sensor_positions: np.ndarray, # NOT USED ----------------------------
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
