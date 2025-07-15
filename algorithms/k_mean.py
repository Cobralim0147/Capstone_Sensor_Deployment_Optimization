import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, List
from configurations.config_file import OptimizationConfig

# Ensure the parent directory is in the system path to find custom modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class KMeansClustering:
    """
    Enhanced K-means clustering for Wireless Sensor Networks (WSN).

    This implementation provides a robust, self-correcting clustering algorithm
    that ensures all sensors are assigned to a cluster while respecting size
    and connectivity constraints. It uses actual sensor nodes as cluster heads.

    Key Features:
    - Guarantees all sensors are clustered by dynamically increasing cluster count.
    - Uses K-means++ for smarter initial head placement.
    - Enforces a maximum cluster size.
    - Uses actual sensor nodes as cluster heads, not abstract centroids.
    - Includes methods for calculating an optimal initial number of clusters.
    - Provides detailed visualization of the results.
    """

    # =========================================================================
    # 1. Initialization
    # =========================================================================

    def __init__(self):
        """
        Initializes the clustering system by loading configuration parameters.
        This method is the constructor for the class and sets up all necessary
        parameters from the configuration file before any clustering is performed.
        """
        self.config = OptimizationConfig()
        
        # Core clustering parameters
        self.comm_range = self.config.deployment_comm_range
        self.max_cluster_size = self.config.clustering_max_cluster_size
        self.max_iterations = self.config.clustering_max_iterations
        self.tolerance = self.config.clustering_tolerance
        
        # Parameters for determining the number of clusters (k)
        self.k_min = self.config.clustering_min_num_cluster
        self.k_max_config = self.config.clustering_max_num_cluster
        self.use_mobile_rover_calc = self.config.clustering_use_mobile_rover_calc
        self.n_init = self.config.clustering_max_restart_attempts
        self.use_coverage_seed = self.config.clustering_use_coverage_seed

    # =========================================================================
    # 2. Public-Facing Method
    # =========================================================================

    def perform_clustering(self, sensor_positions: np.ndarray,
                        n_clusters: Optional[int] = None,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Performs the complete clustering process.
        
        This is the main public method. It orchestrates the entire process:
        1. Determines the initial number of clusters (k) if not provided.
        2. Iteratively tries to find a valid clustering solution, increasing 'k' if necessary.
        3. For each 'k', it runs the algorithm multiple times (`n_init`) to find the best result.
        4. Returns the final cluster assignments, head indices, and metadata.
        """
        n_sensors = len(sensor_positions)
        rng = np.random.default_rng(random_state)
        
        # Determine the initial 'k' to start searching from
        if n_clusters is None:
            if self.use_mobile_rover_calc:
                print("Calculating optimal clusters using mobile rover method...")
                current_k = self.calculate_optimal_clusters_mobile_rover(sensor_positions)
            else:
                print("Calculating optimal clusters using elbow method...")
                current_k = self.calculate_optimal_clusters_elbow(sensor_positions, random_state)
        else:
            current_k = n_clusters
            
        print(f"\n{'='*25}\nStarting Advanced Clustering Process...\n{'='*25}")

        while current_k <= n_sensors:
            print(f"\n---> Searching for a valid solution with k = {current_k}...")
        
            best_solution_for_k = None
            lowest_sse_so_far = float('inf')
            lowest_orphan_count = float('inf')
            
            # Run `n_init` times to find the BEST solution for this k
            for attempt in range(self.n_init):
                attempt_seed = rng.integers(1, 1e6)
                attempt_rng = np.random.default_rng(attempt_seed)
                result = self._run_single_kmeans_instance(sensor_positions, current_k, attempt_rng)
                
                if result is not None:
                    assignments, heads, orphan_count = result
                    
                    if orphan_count == 0:  
                        sse = self.compute_total_sse(sensor_positions, assignments, heads)
                        if sse < lowest_sse_so_far:
                            lowest_sse_so_far = sse
                            best_solution_for_k = (assignments, heads)
                            
                    elif orphan_count > 0:  
                        lowest_orphan_count = min(lowest_orphan_count, orphan_count)
                        
            if best_solution_for_k is not None:
                print(f"\n[✓] SUCCESS: Found optimal configuration for k = {current_k} with a best SSE of {lowest_sse_so_far:.2f}.")
                final_assignments, final_heads = best_solution_for_k
                clustering_info = {'converged': True, 'n_clusters_final': len(final_heads), 'best_sse': lowest_sse_so_far}
                return final_assignments, final_heads, clustering_info
            else:
                if lowest_orphan_count != float('inf'):
                    print(f"  [!] k={current_k} is insufficient. Best attempt had {int(lowest_orphan_count)} orphaned sensors.")
                else:
                    print(f"  [!] k={current_k} is insufficient. No valid clustering found after {self.n_init} attempts.")
                current_k += 1

        print(f"❌ CRITICAL FAILURE: Could not find a valid clustering for any k up to {n_sensors}.")
        return np.array([]), np.array([]), {'clustering_failed': True, 'reason': 'Exhausted all k values.'}

    # =========================================================================
    # 3. Core Clustering Workflow (The K-Means Steps)
    # (These private methods execute the main algorithm logic)
    # =========================================================================
    
    def _run_single_kmeans_instance(self, sensor_positions: np.ndarray, k: int, 
                                    rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Runs one full K-means attempt from initialization to convergence.
        
        It handles the initialization of cluster heads and then iterates through
        the assignment and update steps until the cluster heads no longer change
        or the maximum number of iterations is reached.
        
        Returns:
            A tuple of (assignments, heads, orphan_count). Orphan count is 0 on
            success, >0 if sensors were unassigned, and -1 on convergence failure.
        """
        if self.use_coverage_seed:
            cluster_heads = self._initialize_heads_by_coverage(sensor_positions, k, rng)
        else:
            cluster_heads = self._initialize_cluster_heads_plus_plus(sensor_positions, k, rng)

        for iteration in range(self.max_iterations):
            assignments = self._assign_sensors_to_clusters(sensor_positions, cluster_heads)
            
            if np.any(assignments == -1):
                num_orphan = np.sum(assignments == -1)
                return None, None, num_orphan  
                
            updated_heads = self._update_cluster_heads(sensor_positions, assignments, cluster_heads)
            
            # Check for convergence
            if np.array_equal(np.sort(updated_heads), np.sort(cluster_heads)):
                final_assignments, final_heads = self._remove_empty_clusters(assignments, updated_heads)
                return final_assignments, final_heads, 0 # Success
            
            cluster_heads = updated_heads
        
        return None, None, -1

    def _assign_sensors_to_clusters(self, sensor_positions: np.ndarray, 
                                    cluster_head_indices: np.ndarray) -> np.ndarray:
        """
        Assigns each sensor to the nearest valid cluster head (Assignment Step).
        
        This method iterates through each unassigned sensor and assigns it to the
        closest cluster head, but only if the assignment satisfies two critical
        constraints:
        1. The sensor must be within the communication range (`comm_range`) of the head.
        2. The target cluster must not have exceeded its maximum size (`max_cluster_size`).
        
        Sensors that cannot be assigned to any valid cluster are marked as orphans (-1).
        """
        n_sensors = len(sensor_positions)
        n_clusters = len(cluster_head_indices)
        
        assignments = np.full(n_sensors, -1, dtype=int)
        cluster_capacities = np.full(n_clusters, self.max_cluster_size, dtype=int)
        
        for cluster_id, head_idx in enumerate(cluster_head_indices):
            if assignments[head_idx] == -1:
                assignments[head_idx] = cluster_id
                cluster_capacities[cluster_id] -= 1
            
        unassigned_indices = [i for i in range(n_sensors) if assignments[i] == -1]
        np.random.shuffle(unassigned_indices)
        
        for sensor_idx in unassigned_indices:
            sensor_pos = sensor_positions[sensor_idx]
            distances = [(np.linalg.norm(sensor_pos - sensor_positions[h]), cid) 
                        for cid, h in enumerate(cluster_head_indices)]
            distances.sort()
            
            for dist, cluster_id in distances:
                if (cluster_capacities[cluster_id] > 0 and dist <= self.comm_range): 
                    assignments[sensor_idx] = cluster_id
                    cluster_capacities[cluster_id] -= 1
                    break 
                
        return assignments

    def _update_cluster_heads(self, sensor_positions: np.ndarray, assignments: np.ndarray, 
                            current_heads: np.ndarray) -> np.ndarray:
        """
        Updates cluster head positions based on new member assignments (Update Step).
        
        For each cluster, this method calculates the geometric centroid (mean position)
        of all its members. It then selects the actual sensor node within that cluster
        that is closest to this ideal centroid to become the new cluster head.
        This ensures heads are always actual sensors and are centrally located.
        It includes a check to prevent new heads from being too close to each other.
        """
        new_heads = []
        
        for cluster_id in range(len(current_heads)):
            cluster_members = np.where(assignments == cluster_id)[0]
            
            if len(cluster_members) == 0:
                # If a cluster becomes empty, retain its old head temporarily
                new_heads.append(current_heads[cluster_id])
                continue
            
            # Find the most central sensor in the cluster
            cluster_positions = sensor_positions[cluster_members]
            centroid = np.mean(cluster_positions, axis=0)
            
            # Find the sensor within the cluster closest to the calculated centroid
            distances_to_centroid = [np.linalg.norm(sensor_positions[member] - centroid) 
                                for member in cluster_members]
            best_head_idx = cluster_members[np.argmin(distances_to_centroid)]
            
            # Check if this new head would be too close to other new heads
            too_close = False
            for other_head in new_heads:
                dist = np.linalg.norm(sensor_positions[best_head_idx] - sensor_positions[other_head])
                if dist < self.comm_range * 0.8: # Heuristic to maintain separation
                    too_close = True
                    break
            
            if too_close:
                # Fallback: keep the current head to avoid cluster overlap
                new_heads.append(current_heads[cluster_id])
            else:
                new_heads.append(best_head_idx)
        
        return np.array(new_heads)
        
    def _remove_empty_clusters(self, assignments: np.ndarray, 
                               cluster_heads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cleans up the final result by removing clusters that have no members.
        
        After convergence, some clusters might end up empty. This function
        removes them and re-indexes the remaining clusters and assignments to be
        contiguous (e.g., 0, 1, 2, ...).
        """
        cluster_sizes = np.bincount(assignments, minlength=len(cluster_heads))
        active_clusters_mask = cluster_sizes > 0
        
        if np.all(active_clusters_mask):
            return assignments, cluster_heads

        new_heads = cluster_heads[active_clusters_mask]
        old_to_new_map = -np.ones(len(cluster_heads), dtype=int)
        old_to_new_map[active_clusters_mask] = np.arange(np.sum(active_clusters_mask))
        new_assignments = old_to_new_map[assignments]
        
        return new_assignments, new_heads
        
    # =========================================================================
    # 4. Cluster Head Initialization Strategies
    # =========================================================================
    
    def _initialize_cluster_heads_plus_plus(self, sensor_positions: np.ndarray, n_clusters: int, 
                                            rng: np.random.Generator) -> np.ndarray:
        """
        Selects initial cluster heads using the K-means++ strategy.
        
        This method is designed to produce better initial heads than pure random
        selection. It works by:
        1. Choosing the first head uniformly at random.
        2. For each subsequent head, choosing a sensor with a probability
           proportional to its squared distance from the nearest existing head.
        This tends to select heads that are far apart, leading to faster and
        more stable convergence. A penalty is added to discourage selecting
        heads that are too close.
        """
        n_sensors = len(sensor_positions)
        if n_clusters >= n_sensors:
            return np.arange(n_sensors)

        first_head_idx = rng.integers(0, n_sensors)
        cluster_heads = [first_head_idx]
        
        for _ in range(1, n_clusters):
            distances_sq = np.array([
                min([np.linalg.norm(sensor_positions[i] - sensor_positions[h])**2 
                    for h in cluster_heads]) 
                for i in range(n_sensors)
            ])
            
            probabilities = distances_sq.copy()
            for i in range(n_sensors):
                if i in cluster_heads:
                    probabilities[i] = 0
                    continue
                min_dist = np.sqrt(distances_sq[i])
                if min_dist < self.comm_range * 0.8:  
                    probabilities[i] *= 0.1  
            
            if np.sum(probabilities) == 0:
                # Fallback: if all probabilities are zero, pick the farthest available sensor
                available_indices = [i for i in range(n_sensors) if i not in cluster_heads]
                if not available_indices: break
                farthest_distances = [min([np.linalg.norm(sensor_positions[i] - sensor_positions[h]) 
                                           for h in cluster_heads]) for i in available_indices]
                next_head = available_indices[np.argmax(farthest_distances)]
            else:
                probabilities /= np.sum(probabilities)
                next_head = rng.choice(range(n_sensors), p=probabilities)
            
            cluster_heads.append(next_head)
        
        return np.array(cluster_heads)

    def _initialize_heads_by_coverage(self, sensor_positions: np.ndarray, k: int,
                                    rng: np.random.Generator) -> np.ndarray:
        """
        Selects initial cluster heads by maximizing network coverage.
        
        This is a custom heuristic that prioritizes placing heads where they can
        cover the most uncovered sensors. It works by:
        1. Selecting the first head as the sensor that can communicate with the most neighbors.
        2. Iteratively selecting subsequent heads that cover the maximum number
           of *new* (previously uncovered) sensors, with a penalty for being
           too close to existing heads.
        """
        n_sensors = len(sensor_positions)
        is_covered = np.zeros(n_sensors, dtype=bool)
        cluster_heads = []
        
        coverages = [np.sum(np.linalg.norm(sensor_positions - sensor_positions[i], axis=1) <= self.comm_range) for i in range(n_sensors)]
        best_first_head = np.argmax(coverages)
        
        cluster_heads.append(best_first_head)
        distances = np.linalg.norm(sensor_positions - sensor_positions[best_first_head], axis=1)
        is_covered |= (distances <= self.comm_range)
        
        for _ in range(1, k):
            max_score = -1
            best_candidate_idx = -1
            
            for candidate_idx in range(n_sensors):
                if candidate_idx in cluster_heads: continue
                    
                distances = np.linalg.norm(sensor_positions - sensor_positions[candidate_idx], axis=1)
                new_coverage = np.sum((distances <= self.comm_range) & ~is_covered)
                
                # Penalize candidates too close to existing heads
                min_dist_to_head = min([np.linalg.norm(sensor_positions[candidate_idx] - sensor_positions[h]) for h in cluster_heads])
                penalty = 0
                if min_dist_to_head < self.comm_range:
                    penalty = (self.comm_range - min_dist_to_head) / self.comm_range
                
                score = new_coverage - penalty * 5 # Weighted penalty
                
                if score > max_score:
                    max_score = score
                    best_candidate_idx = candidate_idx
            
            # Fallback if no good candidate is found 
            if best_candidate_idx == -1:
                available_indices = [i for i in range(n_sensors) if i not in cluster_heads]
                if not available_indices: break
                farthest_distances = [min([np.linalg.norm(sensor_positions[i] - sensor_positions[h]) for h in cluster_heads]) for i in available_indices]
                best_candidate_idx = available_indices[np.argmax(farthest_distances)]
            
            if best_candidate_idx != -1:
                cluster_heads.append(best_candidate_idx)
                distances = np.linalg.norm(sensor_positions - sensor_positions[best_candidate_idx], axis=1)
                is_covered |= (distances <= self.comm_range)
        
        return np.array(cluster_heads)

    # =========================================================================
    # 5. Optimal 'k' Calculation Heuristics
    # =========================================================================

    def calculate_optimal_clusters_elbow(self, sensor_positions: np.ndarray, 
                                         random_state: Optional[int] = None) -> int:
        """
        Estimates the optimal number of clusters using a heuristic based on the Elbow Method.
        
        This method provides a safe starting 'k' by calculating the theoretical
        minimum number of clusters needed to house all sensors without violating
        the `max_cluster_size` constraint. A full implementation would test a
        range of 'k' values and find the "elbow" point in the SSE curve.
        """
        n_sensors = len(sensor_positions)
        min_k_by_size = int(np.ceil(n_sensors / self.max_cluster_size))
        
        print(f"Elbow method (heuristic): Min clusters required by size is {min_k_by_size}.")
        return max(self.k_min, min_k_by_size)
        
    def calculate_optimal_clusters_mobile_rover(self, sensor_positions: np.ndarray) -> int:
        """
        Estimates optimal clusters for a mobile rover scenario.
        
        Similar to the elbow heuristic, this provides a starting 'k' based on
        the `max_cluster_size` constraint. This could be extended with logic
        specific to rover pathing, energy, or data collection efficiency.
        """
        n_sensors = len(sensor_positions)
        min_k_by_size = int(np.ceil(n_sensors / self.max_cluster_size))
        
        print(f"Mobile Rover calculation: Min clusters required by size is {min_k_by_size}.")
        return max(self.k_min, min_k_by_size)
    
    # =========================================================================
    # 6. Analysis and Utility Functions
    # =========================================================================

    def compute_total_sse(self, sensor_positions: np.ndarray, assignments: np.ndarray, 
                          cluster_heads: np.ndarray) -> float:
        """
        Computes the total Sum of Squared Errors (SSE) for the clustering.
        
        SSE is a common metric for evaluating the quality of a clustering. It
        measures the total squared Euclidean distance of each sensor from its
        assigned cluster head. A lower SSE generally indicates a more compact
        and well-formed set of clusters.
        """
        total_sse = 0.0
        head_positions = sensor_positions[cluster_heads]
        
        for cid in range(len(cluster_heads)):
            cluster_members_mask = (assignments == cid)
            if np.any(cluster_members_mask):
                head_pos = head_positions[cid]
                cluster_positions = sensor_positions[cluster_members_mask]
                total_sse += np.sum((cluster_positions - head_pos)**2)
        return total_sse

    def analyze_cluster_connectivity(self, sensor_positions: np.ndarray,
                                assignments: np.ndarray, 
                                cluster_heads: np.ndarray) -> dict:
        """
        Analyzes connectivity within each cluster.
        
        This function checks, for each cluster, how many member sensors are
        actually within the communication range of their assigned cluster head.
        It returns a dictionary with statistics like cluster size, connectivity
        rate, and lists of any isolated (unconnected) sensors.
        """
        connectivity_results = {}
        
        for cluster_id, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            cluster_sensors = np.where(assignments == cluster_id)[0]
            
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
                'connectivity_rate': connected_to_head / len(cluster_sensors) * 100 if len(cluster_sensors) > 0 else 0,
                'isolated_sensors': isolated_sensors,
                'head_position': head_pos
            }
        
        return connectivity_results
        
    def calculate_euclidean_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """
        Calculates the straight-line Euclidean distance between two points.
        """
        return np.linalg.norm(coord1 - coord2)
    
    # =========================================================================
    # 7. Visualization
    # =========================================================================

    def plot_clustering_results(self, sensor_positions: np.ndarray,
                                assignments: np.ndarray, 
                                cluster_heads: np.ndarray,
                                title: str = "WSN Clustering Results"):
        """
        Visualizes the final clustering results.
        
        This method generates a 2D scatter plot showing:
        - Regular sensor nodes, colored by their cluster assignment.
        - Cluster heads, marked with a distinct symbol (*).
        - The communication range of each cluster head, drawn as a circle.
        It provides a clear visual representation of the network topology.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Handle the case of a failed clustering for graceful plotting
        if len(cluster_heads) == 0:
            ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='red', marker='x', label='Unclustered Sensors')
            ax.set_title("Clustering Failed", fontsize=16, fontweight='bold')
            ax.set_xlabel("X-coordinate (m)", fontsize=12)
            ax.set_ylabel("Y-coordinate (m)", fontsize=12)
            ax.legend()
            plt.show()
            return
            
        n_clusters = len(cluster_heads)
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

        # Plot regular member sensors
        for cid in range(n_clusters):
            cluster_members_mask = (assignments == cid)
            # Ensure head is not plotted as a regular sensor
            regular_sensors_indices = np.where(cluster_members_mask & ~np.isin(np.arange(len(assignments)), cluster_heads))[0]
            if len(regular_sensors_indices) > 0:
                ax.scatter(sensor_positions[regular_sensors_indices, 0], sensor_positions[regular_sensors_indices, 1],
                           c=[colors[cid]], s=50, alpha=0.7, label=f'Cluster {cid} Members' if cid < 0 else None) # Simplified label
        
        # Plot cluster heads and their communication ranges
        for cid, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            
            # Draw communication range circle
            circle = plt.Circle(head_pos, self.comm_range, fill=False, color=colors[cid], 
                                linestyle='--', linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
            
            # Draw cluster head marker (plotted on top)
            ax.scatter(head_pos[0], head_pos[1], c=[colors[cid]], marker='*', s=250, 
                       edgecolors='black', linewidth=1.5)

        # Create a clean and informative legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i} (Size: {np.sum(assignments==i)})',
                                      markerfacecolor=colors[i], markersize=10) for i in range(n_clusters)]
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label='Cluster Head',
                                         markerfacecolor='gray', markersize=15, markeredgecolor='black'))
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("X-coordinate (m)", fontsize=12)
        ax.set_ylabel("Y-coordinate (m)", fontsize=12)
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()