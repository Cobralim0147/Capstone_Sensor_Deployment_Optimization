"""
Fuzzy Logic Clustering Optimizer for Wireless Sensor Networks (WSN)

This implementation enhances K-means clustering with fuzzy logic optimization:
- Extends initial clustering with fuzzy validation
- Optimizes cluster head selection using multiple criteria
- Provides comprehensive clustering quality assessment
- Integrates seamlessly with pure K-means clustering

Features:
- Fuzzy logic cluster head evaluation
- Multiple optimization criteria (energy, centrality, density, distance)
- Adaptive cluster head selection
- Quality metrics and validation
- Comprehensive reporting and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import *
import warnings

# Import the pure clustering functionality
from initial_clustering import SensorNetworkClustering, analyze_network_connectivity

# Import fuzzy logic components
try:
    from fuzzy_logic import FuzzyClusterValidator, SensorNode
    FUZZY_AVAILABLE = True
except ImportError:
    print("Warning: Fuzzy validation module not available. Some features will be disabled.")
    FUZZY_AVAILABLE = False


class FuzzyClusteringOptimizer(SensorNetworkClustering):
    """
    Enhanced K-means clustering with fuzzy logic optimization.
    
    This class extends the base clustering functionality with fuzzy logic
    to optimize cluster head selection and validate clustering quality.
    """
    
    def __init__(self, comm_range: float, max_cluster_size: int, 
                 max_iterations: int, tolerance: float,
                 base_station_pos: Optional[np.ndarray] = None,
                 enable_fuzzy: bool = True):
        """
        Initialize the fuzzy clustering optimizer.
        
        Args:
            comm_range: Communication range of sensors
            max_cluster_size: Maximum number of sensors per cluster
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for cluster head changes
            base_station_pos: Position of base station [x, y]
            enable_fuzzy: Whether to enable fuzzy logic optimization
        """
        super().__init__(comm_range, max_cluster_size, max_iterations, tolerance)
        
        self.enable_fuzzy = enable_fuzzy and FUZZY_AVAILABLE
        self.fuzzy_validator = None
        
        if self.enable_fuzzy and base_station_pos is not None:
            try:
                self.fuzzy_validator = FuzzyClusterValidator(base_station_pos)
                print("Fuzzy logic optimization enabled")
            except Exception as e:
                print(f"Warning: Could not initialize fuzzy validator: {e}")
                self.enable_fuzzy = False
        elif self.enable_fuzzy:
            print("Warning: Base station position required for fuzzy optimization")
            self.enable_fuzzy = False
    
    def setup_fuzzy_system(self, base_station_pos: np.ndarray):
        """
        Set up the fuzzy logic system with base station position.
        
        Args:
            base_station_pos: Position of base station [x, y]
        """
        if not FUZZY_AVAILABLE:
            print("Warning: Fuzzy validation module not available")
            return False
            
        try:
            self.fuzzy_validator = FuzzyClusterValidator(base_station_pos)
            self.enable_fuzzy = True
            print("Fuzzy logic system initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing fuzzy system: {e}")
            self.enable_fuzzy = False
            return False
    
    def perform_fuzzy_optimized_clustering(self, sensor_positions: np.ndarray, 
                                         n_clusters: int,
                                         random_state: Optional[int] = None,
                                         max_fuzzy_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform clustering with fuzzy logic optimization.
        
        Args:
            sensor_positions: Array of sensor positions (n_sensors, 2)
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
            max_fuzzy_iterations: Maximum iterations for fuzzy optimization
            
        Returns:
            Tuple of (cluster_assignments, cluster_head_indices, clustering_info)
        """
        if not self.enable_fuzzy or self.fuzzy_validator is None:
            print("Fuzzy optimization not available, using standard K-means")
            return self.perform_clustering(sensor_positions, n_clusters, random_state)
        
        print(f"Starting fuzzy-optimized clustering with {n_clusters} clusters...")
        
        # Step 1: Perform initial K-means clustering
        assignments, cluster_heads, info = self.perform_clustering(
            sensor_positions, n_clusters, random_state
        )
        
        initial_sse = info['total_sse']
        best_assignments = assignments.copy()
        best_cluster_heads = cluster_heads.copy()
        best_sse = initial_sse
        
        print(f"Initial clustering SSE: {initial_sse:.2f}")
        
        # Step 2: Iteratively optimize using fuzzy logic
        for iteration in range(max_fuzzy_iterations):
            print(f"Fuzzy optimization iteration {iteration + 1}/{max_fuzzy_iterations}")
            
            try:
                # Create sensor nodes for fuzzy evaluation
                sensor_nodes = self._create_sensor_nodes(
                    sensor_positions, assignments, cluster_heads
                )
                
                # Optimize cluster heads using fuzzy logic
                improved_heads = self._optimize_cluster_heads_fuzzy(
                    sensor_positions, assignments, cluster_heads, sensor_nodes
                )
                
                # Re-assign sensors with new cluster heads
                new_assignments = self.assign_sensors_to_clusters(
                    sensor_positions, improved_heads
                )
                
                # Validate cluster sizes
                new_assignments, improved_heads = self.validate_cluster_sizes(
                    new_assignments, sensor_positions, improved_heads
                )
                
                # Calculate new SSE
                new_sse = self.compute_total_sse(
                    sensor_positions, new_assignments, improved_heads
                )
                
                print(f"  New SSE: {new_sse:.2f} (improvement: {best_sse - new_sse:.2f})")
                
                # Keep improvement if it's better
                if new_sse < best_sse:
                    best_assignments = new_assignments.copy()
                    best_cluster_heads = improved_heads.copy()
                    best_sse = new_sse
                    print(f"  ✓ Improvement accepted")
                else:
                    print(f"  ✗ No improvement, keeping previous solution")
                    break
                    
            except Exception as e:
                print(f"  Error in fuzzy optimization: {e}")
                break
        
        # Update final info
        final_info = {
            'iterations': info['iterations'],
            'fuzzy_iterations': iteration + 1,
            'initial_sse': initial_sse,
            'final_sse': best_sse,
            'sse_improvement': initial_sse - best_sse,
            'cluster_sizes': np.bincount(best_assignments, minlength=n_clusters),
            'converged': True,
            'fuzzy_optimized': True
        }
        
        print(f"Fuzzy optimization complete. Total SSE improvement: {initial_sse - best_sse:.2f}")
        
        return best_assignments, best_cluster_heads, final_info
    
    def _create_sensor_nodes(self, sensor_positions: np.ndarray, 
                           assignments: np.ndarray, 
                           cluster_heads: np.ndarray) -> List[SensorNode]:
        """
        Create sensor node objects with clustering information.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            
        Returns:
            List of SensorNode objects
        """
        sensor_nodes = []
        for i, pos in enumerate(sensor_positions):
            energy_level = np.random.uniform(0.3, 1.0)  # Random energy for demo
            is_head = i in cluster_heads
            cluster_id = assignments[i]
            
            node = SensorNode(
                id=i,
                x=pos[0],
                y=pos[1],
                energy_level=energy_level,
                cluster_id=cluster_id,
                is_cluster_head=is_head
            )
            sensor_nodes.append(node)
        
        return sensor_nodes
    
    def _optimize_cluster_heads_fuzzy(self, sensor_positions: np.ndarray,
                                    assignments: np.ndarray,
                                    current_heads: np.ndarray,
                                    sensor_nodes: List[SensorNode]) -> np.ndarray:
        """
        Optimize cluster heads using fuzzy logic evaluation.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Current cluster assignments
            current_heads: Current cluster head indices
            sensor_nodes: List of sensor nodes
            
        Returns:
            Optimized cluster head indices
        """
        if not self.enable_fuzzy or self.fuzzy_validator is None:
            return current_heads
        
        n_clusters = len(current_heads)
        optimized_heads = current_heads.copy()
        
        for cluster_id in range(n_clusters):
            cluster_sensors = np.where(assignments == cluster_id)[0]
            
            if len(cluster_sensors) <= 1:
                continue  # Skip single-sensor clusters
            
            # Evaluate all sensors in the cluster as potential heads
            best_sensor = current_heads[cluster_id]
            best_score = 0.0
            
            for sensor_idx in cluster_sensors:
                try:
                    # Create temporary sensor nodes for evaluation
                    temp_nodes = [node for node in sensor_nodes]
                    
                    # Set this sensor as temporary cluster head
                    for node in temp_nodes:
                        node.is_cluster_head = (node.id == sensor_idx)
                    
                    # Evaluate using fuzzy logic
                    cluster_quality = self.fuzzy_validator.evaluate_cluster_quality(
                        temp_nodes, cluster_id
                    )
                    
                    if cluster_quality > best_score:
                        best_score = cluster_quality
                        best_sensor = sensor_idx
                        
                except Exception as e:
                    print(f"    Error evaluating sensor {sensor_idx}: {e}")
                    continue
            
            optimized_heads[cluster_id] = best_sensor
        
        return optimized_heads
    
    def compare_clustering_methods(self, sensor_positions: np.ndarray,
                                 n_clusters_range: Tuple[int, int],
                                 random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare standard K-means with fuzzy-optimized clustering.
        
        Args:
            sensor_positions: Array of sensor positions
            n_clusters_range: Tuple of (min_clusters, max_clusters) to test
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with comparison results
        """
        min_k, max_k = n_clusters_range
        results = {
            'k_values': [],
            'standard_sse': [],
            'fuzzy_sse': [],
            'improvements': [],
            'fuzzy_available': self.enable_fuzzy
        }
        
        print(f"Comparing clustering methods for k={min_k} to {max_k}")
        
        for k in range(min_k, min(max_k + 1, len(sensor_positions))):
            print(f"\nTesting k={k}")
            
            # Standard K-means
            assignments_std, heads_std, info_std = self.perform_clustering(
                sensor_positions, k, random_state
            )
            standard_sse = info_std['total_sse']
            
            # Fuzzy-optimized clustering
            if self.enable_fuzzy:
                assignments_fuzzy, heads_fuzzy, info_fuzzy = self.perform_fuzzy_optimized_clustering(
                    sensor_positions, k, random_state, max_fuzzy_iterations=3
                )
                fuzzy_sse = info_fuzzy['final_sse']
                improvement = standard_sse - fuzzy_sse
            else:
                fuzzy_sse = standard_sse
                improvement = 0.0
            
            results['k_values'].append(k)
            results['standard_sse'].append(standard_sse)
            results['fuzzy_sse'].append(fuzzy_sse)
            results['improvements'].append(improvement)
            
            print(f"  Standard SSE: {standard_sse:.2f}")
            print(f"  Fuzzy SSE: {fuzzy_sse:.2f}")
            print(f"  Improvement: {improvement:.2f}")
        
        return results
    
    def evaluate_clustering_quality(self, sensor_positions: np.ndarray,
                                  assignments: np.ndarray,
                                  cluster_heads: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['sse'] = self.compute_total_sse(sensor_positions, assignments, cluster_heads)
        metrics['n_clusters'] = len(cluster_heads)
        
        # Cluster size metrics
        cluster_sizes = np.bincount(assignments, minlength=len(cluster_heads))
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['cluster_size_std'] = np.std(cluster_sizes)
        metrics['max_cluster_size'] = np.max(cluster_sizes)
        metrics['min_cluster_size'] = np.min(cluster_sizes)
        
        # Connectivity metrics
        connectivity_info = analyze_network_connectivity(sensor_positions, self.comm_range)
        metrics['avg_connectivity'] = connectivity_info['average_degree']
        metrics['connectivity_ratio'] = connectivity_info['connectivity_ratio']
        
        # Distance metrics
        distances_to_heads = []
        for i, pos in enumerate(sensor_positions):
            cluster_id = assignments[i]
            head_pos = sensor_positions[cluster_heads[cluster_id]]
            distance = self.calculate_distance_sensor_to_center(pos, head_pos)
            distances_to_heads.append(distance)
        
        metrics['avg_distance_to_head'] = np.mean(distances_to_heads)
        metrics['max_distance_to_head'] = np.max(distances_to_heads)
        
        # Fuzzy quality metrics (if available)
        if self.enable_fuzzy and self.fuzzy_validator is not None:
            try:
                sensor_nodes = self._create_sensor_nodes(sensor_positions, assignments, cluster_heads)
                
                cluster_qualities = []
                for cluster_id in range(len(cluster_heads)):
                    quality = self.fuzzy_validator.evaluate_cluster_quality(
                        sensor_nodes, cluster_id
                    )
                    cluster_qualities.append(quality)
                
                metrics['fuzzy_avg_quality'] = np.mean(cluster_qualities)
                metrics['fuzzy_min_quality'] = np.min(cluster_qualities)
                
            except Exception as e:
                print(f"Error calculating fuzzy metrics: {e}")
        
        return metrics
    
    def plot_comparison_results(self, comparison_results: Dict[str, Any]):
        """
        Plot comparison between standard and fuzzy-optimized clustering.
        
        Args:
            comparison_results: Results from compare_clustering_methods
        """
        if not comparison_results['k_values']:
            print("No comparison results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        k_values = comparison_results['k_values']
        standard_sse = comparison_results['standard_sse']
        fuzzy_sse = comparison_results['fuzzy_sse']
        improvements = comparison_results['improvements']
        
        # Plot 1: SSE comparison
        ax1.plot(k_values, standard_sse, 'o-', label='Standard K-means', color='blue')
        if comparison_results['fuzzy_available']:
            ax1.plot(k_values, fuzzy_sse, 's-', label='Fuzzy-optimized', color='red')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Sum of Squared Errors (SSE)')
        ax1.set_title('Clustering Quality Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvements
        if comparison_results['fuzzy_available']:
            ax2.bar(k_values, improvements, alpha=0.7, color='green')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('SSE Improvement')
            ax2.set_title('Fuzzy Optimization Improvements')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Fuzzy optimization\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax2.set_title('Fuzzy Optimization Status')
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, sensor_positions: np.ndarray,
                                    assignments: np.ndarray,
                                    cluster_heads: np.ndarray,
                                    clustering_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive clustering report.
        
        Args:
            sensor_positions: Array of sensor positions
            assignments: Cluster assignments
            cluster_heads: Cluster head indices
            clustering_info: Clustering information dictionary
            
        Returns:
            Formatted report string
        """
        quality_metrics = self.evaluate_clustering_quality(
            sensor_positions, assignments, cluster_heads
        )
        
        report = []
        report.append("=" * 60)
        report.append("WIRELESS SENSOR NETWORK CLUSTERING REPORT")
        report.append("=" * 60)
        
        # Basic information
        report.append(f"\nBASIC INFORMATION:")
        report.append(f"  Total sensors: {len(sensor_positions)}")
        report.append(f"  Number of clusters: {len(cluster_heads)}")
        report.append(f"  Communication range: {self.comm_range}")
        report.append(f"  Max cluster size: {self.max_cluster_size}")
        
        # Clustering process information
        report.append(f"\nCLUSTERING PROCESS:")
        report.append(f"  Standard iterations: {clustering_info.get('iterations', 'N/A')}")
        if 'fuzzy_iterations' in clustering_info:
            report.append(f"  Fuzzy iterations: {clustering_info['fuzzy_iterations']}")
            report.append(f"  Fuzzy optimization: Enabled")
        else:
            report.append(f"  Fuzzy optimization: Disabled")
        
        # Quality metrics
        report.append(f"\nQUALITY METRICS:")
        report.append(f"  Sum of Squared Errors: {quality_metrics['sse']:.2f}")
        if 'initial_sse' in clustering_info:
            improvement = clustering_info['initial_sse'] - clustering_info['final_sse']
            report.append(f"  SSE Improvement: {improvement:.2f}")
        
        report.append(f"  Average cluster size: {quality_metrics['avg_cluster_size']:.1f}")
        report.append(f"  Cluster size std dev: {quality_metrics['cluster_size_std']:.1f}")
        report.append(f"  Size range: {quality_metrics['min_cluster_size']}-{quality_metrics['max_cluster_size']}")
        
        # Connectivity metrics
        report.append(f"\nCONNECTIVITY METRICS:")
        report.append(f"  Average node degree: {quality_metrics['avg_connectivity']:.1f}")
        report.append(f"  Connectivity ratio: {quality_metrics['connectivity_ratio']:.2f}")
        
        # Distance metrics  
        report.append(f"\nDISTANCE METRICS:")
        report.append(f"  Avg distance to cluster head: {quality_metrics['avg_distance_to_head']:.2f}")
        report.append(f"  Max distance to cluster head: {quality_metrics['max_distance_to_head']:.2f}")
        
        # Fuzzy metrics (if available)
        if 'fuzzy_avg_quality' in quality_metrics:
            report.append(f"\nFUZZY QUALITY METRICS:")
            report.append(f"  Average fuzzy quality: {quality_metrics['fuzzy_avg_quality']:.3f}")
            report.append(f"  Minimum fuzzy quality: {quality_metrics['fuzzy_min_quality']:.3f}")
        
        # Cluster head information
        report.append(f"\nCLUSTER HEAD INFORMATION:")
        cluster_sizes = np.bincount(assignments, minlength=len(cluster_heads))
        for i, head_idx in enumerate(cluster_heads):
            head_pos = sensor_positions[head_idx]
            report.append(f"  Cluster {i}: Head #{head_idx} at ({head_pos[0]:.1f}, {head_pos[1]:.1f}), {cluster_sizes[i]} sensors")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def validate_fuzzy_system(self) -> None:
        """
        Validate that the fuzzy system is working correctly.
        """
        if not FUZZY_AVAILABLE:
            print("❌ Fuzzy validation module not available")
            return
            
        if self.fuzzy_validator is None:
            print("❌ Fuzzy validator not initialized")
            return
        
        try:
            # Test fuzzy system with dummy data
            test_nodes = [
                SensorNode(0, 10.0, 10.0, energy_level=0.8, cluster_id=0, is_cluster_head=True),
                SensorNode(1, 12.0, 11.0, energy_level=0.6, cluster_id=0, is_cluster_head=False),
                SensorNode(2, 11.0, 9.0, energy_level=0.7, cluster_id=0, is_cluster_head=False)
            ]
            
            quality = self.fuzzy_validator.evaluate_cluster_quality(test_nodes, 0)
            print(f"✅ Fuzzy system validation successful")
            print(f"   Test cluster quality: {quality:.3f}")
            
        except Exception as e:
            print(f"❌ Fuzzy system validation failed: {e}")


# Convenience functions for easy usage
def create_standard_clustering_system(comm_range: float = 50.0, 
                                    max_cluster_size: int = 20,
                                    max_iterations: int = 100,
                                    tolerance: float = 1e-4) -> SensorNetworkClustering:
    """
    Create a standard K-means clustering system.
    
    Args:
        comm_range: Communication range of sensors
        max_cluster_size: Maximum sensors per cluster
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
    Returns:
        SensorNetworkClustering instance
    """
    return SensorNetworkClustering(comm_range, max_cluster_size, max_iterations, tolerance)


def create_fuzzy_clustering_system(comm_range: float = 50.0,
                                 max_cluster_size: int = 20,
                                 max_iterations: int = 100,
                                 tolerance: float = 1e-4,
                                 base_station_pos: Optional[np.ndarray] = None) -> FuzzyClusteringOptimizer:
    """
    Create a fuzzy-optimized clustering system.
    
    Args:
        comm_range: Communication range of sensors
        max_cluster_size: Maximum sensors per cluster
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        base_station_pos: Base station position [x, y]
        
    Returns:
        FuzzyClusteringOptimizer instance
    """
    return FuzzyClusteringOptimizer(
        comm_range, max_cluster_size, max_iterations, tolerance, 
        base_station_pos, enable_fuzzy=True
    )
