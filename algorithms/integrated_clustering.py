"""
Integration module for combining K-means clustering with Fuzzy Logic validation.

This module integrates the existing K-means clustering from initial_clustering.py
with the fuzzy logic cluster head selection and validation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import os
import sys
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
try:
    from .k_mean import KMeansClustering
    from environment.sym_env import FieldEnvironmentGenerator
    from .fuzzy_logic import FuzzyLogicCluster
    from configurations.data_structure import SensorNode
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure initial_clustering.py and generate_field.py are in the same directory")


class IntegratedClusteringSystem:
    """
    Integrated system that combines K-means clustering with fuzzy logic validation.
    """
    
    def __init__(self, base_station_pos: Tuple[float, float]):
        """
        Initialize the integrated clustering system.
        
        Parameters:
        -----------
        base_station_pos : Tuple[float, float]
            Position of the base station
        """
        self.base_station_pos = base_station_pos
        self.fuzzy_validator = FuzzyLogicCluster(base_station_pos)
        self.kmeans_results = None
        self.fuzzy_results = None
        
    def convert_positions_to_sensor_nodes(self, sensor_positions: np.ndarray, 
                                        energy_range: Tuple[float, float] = (0.5, 1.0)) -> List[SensorNode]:
        """
        Convert sensor positions to SensorNode objects.
        
        Parameters:
        -----------
        sensor_positions : np.ndarray
            Array of sensor positions [x, y]
        energy_range : Tuple[float, float]
            Range for random energy assignment (min, max)
            
        Returns:
        --------
        List[SensorNode]
            List of sensor nodes
        """
        nodes = []
        for i, pos in enumerate(sensor_positions):
            # Assign random energy level within specified range
            energy = np.random.uniform(energy_range[0], energy_range[1])
            
            node = SensorNode(
                id=i,
                x=float(pos[0]),
                y=float(pos[1]),
                energy_level=energy
            )
            nodes.append(node)
        
        return nodes
    
    def perform_integrated_clustering(self, sensor_positions: np.ndarray, 
                                    n_clusters: int = 5, 
                                    energy_range: Tuple[float, float] = (0.5, 1.0)) -> Dict[str, Any]:
        """
        Perform integrated clustering using K-means followed by fuzzy validation.
        
        Parameters:
        -----------
        sensor_positions : np.ndarray
            Array of sensor positions [x, y]
        n_clusters : int
            Number of clusters for K-means
        energy_range : Tuple[float, float]
            Range for random energy assignment
            
        Returns:
        --------
        Dict[str, Any]
            Results containing both K-means and fuzzy analysis
        """
        # Convert positions to sensor nodes
        sensor_nodes = self.convert_positions_to_sensor_nodes(sensor_positions, energy_range)
        
        # Perform K-means clustering
        # try:
        positions = np.array([[node.x, node.y] for node in sensor_nodes])
        
        # Create clustering system and get results
        clustering_system = KMeansClustering()
        cluster_labels, cluster_heads, clustering_info = clustering_system.perform_clustering(positions, n_clusters)
        
        # Extract cluster centers from cluster heads
        cluster_centers = positions[cluster_heads]
        
        # Create kmeans_results dictionary for later use
        kmeans_results = {
            'labels': cluster_labels,
            'centers': cluster_centers,
            'cluster_heads': cluster_heads,
            'inertia': clustering_info.get('total_sse', 0)
        }
            
        # except NameError:
        #     # Fallback to sklearn if initial_clustering is not available
        #     from sklearn.cluster import KMeans
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        #     cluster_labels = kmeans.fit_predict(positions)
        #     cluster_centers = kmeans.cluster_centers_
            
        #     kmeans_results = {
        #         'labels': cluster_labels,
        #         'centers': cluster_centers,
        #         'inertia': kmeans.inertia_
        #     }
        
        # Assign cluster labels to nodes
        for i, node in enumerate(sensor_nodes):
            node.cluster_id = cluster_labels[i]
        
        # Group nodes by cluster
        clusters = {}
        for node in sensor_nodes:
            if node.cluster_id not in clusters:
                clusters[node.cluster_id] = []
            clusters[node.cluster_id].append(node)
        
        # Perform fuzzy logic cluster head selection
        cluster_metrics = self.fuzzy_validator.select_cluster_heads(clusters)
        
        # Validate clustering quality
        quality_metrics = self.fuzzy_validator.validate_clustering_quality(cluster_metrics)
        
        # Store results
        self.kmeans_results = kmeans_results
        self.fuzzy_results = {
            'cluster_metrics': cluster_metrics,
            'quality_metrics': quality_metrics
        }
        
        return {
            'sensor_nodes': sensor_nodes,
            'kmeans_results': kmeans_results,
            'fuzzy_results': self.fuzzy_results,
            'clusters': clusters,
            'quality_metrics': quality_metrics
        }
    
    def compare_clustering_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare K-means clustering with fuzzy logic validation.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results from integrated clustering
            
        Returns:
        --------
        Dict[str, Any]
            Comparison metrics
        """
        comparison = {}
        
        # K-means metrics
        comparison['kmeans_inertia'] = results['kmeans_results'].get('inertia', 0)
        comparison['num_clusters'] = len(results['clusters'])
        
        # Fuzzy logic metrics
        fuzzy_quality = results['quality_metrics']
        comparison['fuzzy_overall_quality'] = fuzzy_quality['overall_quality']
        comparison['fuzzy_avg_score'] = fuzzy_quality['avg_fuzzy_score']
        comparison['cluster_balance'] = fuzzy_quality['cluster_balance']
        
        # Cluster head quality analysis
        cluster_head_scores = []
        for cluster_id, metrics in results['fuzzy_results']['cluster_metrics'].items():
            if metrics.selected_head:
                cluster_head_scores.append(metrics.fuzzy_score)
        
        comparison['avg_cluster_head_score'] = np.mean(cluster_head_scores) if cluster_head_scores else 0.0
        comparison['min_cluster_head_score'] = np.min(cluster_head_scores) if cluster_head_scores else 0.0
        comparison['max_cluster_head_score'] = np.max(cluster_head_scores) if cluster_head_scores else 0.0
        
        # Recommendation based on fuzzy scores
        if comparison['fuzzy_overall_quality'] >= 0.7:
            comparison['recommendation'] = "Excellent clustering - K-means results validated by fuzzy logic"
        elif comparison['fuzzy_overall_quality'] >= 0.5:
            comparison['recommendation'] = "Good clustering - minor optimization possible"
        elif comparison['fuzzy_overall_quality'] >= 0.3:
            comparison['recommendation'] = "Moderate clustering - consider different parameters or methods"
        else:
            comparison['recommendation'] = "Poor clustering - re-clustering recommended"
        
        return comparison
    
    def visualize_integrated_results(self, results: Dict[str, Any], comparison: Dict[str, Any]):
        """
        Create comprehensive visualization of integrated clustering results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results from integrated clustering
        comparison : Dict[str, Any]
            Comparison metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Integrated K-means + Fuzzy Logic Clustering Analysis', fontsize=16)
        
        # Plot 1: K-means clustering results
        ax1 = axes[0, 0]
        self._plot_kmeans_results(ax1, results)
        
        # Plot 2: Fuzzy logic cluster heads
        ax2 = axes[0, 1]
        self._plot_fuzzy_results(ax2, results)
        
        # Plot 3: Cluster quality comparison
        ax3 = axes[1, 0]
        self._plot_quality_comparison(ax3, comparison)
        
        # Plot 4: Energy vs Fuzzy Score analysis
        ax4 = axes[1, 1]
        self._plot_energy_analysis(ax4, results)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_kmeans_results(self, ax, results):
        """Plot K-means clustering results."""
        clusters = results['clusters']
        centers = results['kmeans_results']['centers']
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for i, (cluster_id, nodes) in enumerate(clusters.items()):
            x_coords = [node.x for node in nodes]
            y_coords = [node.y for node in nodes]
            ax.scatter(x_coords, y_coords, c=[colors[i]], alpha=0.6, s=50, 
                      label=f'Cluster {cluster_id}')
        
        # Plot cluster centers
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, 
                  linewidths=3, label='K-means Centers')
        
        # Plot base station
        ax.scatter(self.base_station_pos[0], self.base_station_pos[1], 
                  c='blue', marker='s', s=150, label='Base Station')
        
        ax.set_title('K-means Clustering Results')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_fuzzy_results(self, ax, results):
        """Plot fuzzy logic cluster head selection results."""
        cluster_metrics = results['fuzzy_results']['cluster_metrics']
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_metrics)))
        
        for i, (cluster_id, metrics) in enumerate(cluster_metrics.items()):
            x_coords = [node.x for node in metrics.nodes]
            y_coords = [node.y for node in metrics.nodes]
            ax.scatter(x_coords, y_coords, c=[colors[i]], alpha=0.6, s=50)
            
            # Highlight cluster head
            if metrics.selected_head:
                ax.scatter(metrics.selected_head.x, metrics.selected_head.y, 
                          c='red', marker='*', s=200, edgecolors='black', linewidth=2)
                ax.annotate(f'CH{cluster_id}\n{metrics.fuzzy_score:.2f}',
                           (metrics.selected_head.x, metrics.selected_head.y),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot base station
        ax.scatter(self.base_station_pos[0], self.base_station_pos[1], 
                  c='blue', marker='s', s=150, label='Base Station')
        
        quality = results['quality_metrics']['overall_quality']
        ax.set_title(f'Fuzzy Logic Cluster Heads (Quality: {quality:.3f})')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_comparison(self, ax, comparison):
        """Plot quality metrics comparison."""
        metrics = ['Fuzzy Quality', 'Cluster Balance', 'Avg CH Score']
        values = [
            comparison['fuzzy_overall_quality'],
            comparison['cluster_balance'],
            comparison['avg_cluster_head_score']
        ]
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Clustering Quality Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_energy_analysis(self, ax, results):
        """Plot energy vs fuzzy score analysis."""
        cluster_metrics = results['fuzzy_results']['cluster_metrics']
        
        # Collect data for all nodes
        energies = []
        fuzzy_scores = []
        is_cluster_head = []
        
        for metrics in cluster_metrics.values():
            for node in metrics.nodes:
                energies.append(node.energy_level)
                # Calculate fuzzy score for this node
                score = self.fuzzy_validator.evaluate_node_for_cluster_head(
                    node, metrics.nodes, max(len(m.nodes) for m in cluster_metrics.values())
                )
                fuzzy_scores.append(score)
                is_cluster_head.append(node.is_cluster_head)
        
        # Plot all nodes
        energies = np.array(energies)
        fuzzy_scores = np.array(fuzzy_scores)
        is_cluster_head = np.array(is_cluster_head)
        
        # Regular nodes
        regular_mask = ~is_cluster_head
        ax.scatter(energies[regular_mask], fuzzy_scores[regular_mask], 
                  c='lightblue', alpha=0.6, s=50, label='Regular Nodes')
        
        # Cluster heads
        ch_mask = is_cluster_head
        ax.scatter(energies[ch_mask], fuzzy_scores[ch_mask], 
                  c='red', marker='*', s=150, edgecolors='black', 
                  linewidth=1, label='Cluster Heads')
        
        ax.set_xlabel('Energy Level')
        ax.set_ylabel('Fuzzy Score')
        ax.set_title('Energy vs Fuzzy Score Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_report(self, results: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text report of the clustering analysis.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results from integrated clustering
        comparison : Dict[str, Any]
            Comparison metrics
            
        Returns:
        --------
        str
            Formatted report
        """
        report = []
        report.append("INTEGRATED CLUSTERING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overview
        report.append("OVERVIEW:")
        report.append(f"Number of sensor nodes: {len(results['sensor_nodes'])}")
        report.append(f"Number of clusters: {comparison['num_clusters']}")
        report.append(f"Base station position: {self.base_station_pos}")
        report.append("")
        
        # K-means results
        report.append("K-MEANS CLUSTERING RESULTS:")
        report.append(f"Inertia (within-cluster sum of squares): {comparison['kmeans_inertia']:.2f}")
        report.append("")
        
        # Fuzzy logic results
        report.append("FUZZY LOGIC VALIDATION RESULTS:")
        report.append(f"Overall quality score: {comparison['fuzzy_overall_quality']:.3f}")
        report.append(f"Average fuzzy score: {comparison['fuzzy_avg_score']:.3f}")
        report.append(f"Cluster balance: {comparison['cluster_balance']:.3f}")
        report.append("")
        
        # Cluster head analysis
        report.append("CLUSTER HEAD ANALYSIS:")
        report.append(f"Average cluster head score: {comparison['avg_cluster_head_score']:.3f}")
        report.append(f"Minimum cluster head score: {comparison['min_cluster_head_score']:.3f}")
        report.append(f"Maximum cluster head score: {comparison['max_cluster_head_score']:.3f}")
        report.append("")
        
        # Individual cluster details
        report.append("CLUSTER DETAILS:")
        cluster_metrics = results['fuzzy_results']['cluster_metrics']
        for cluster_id, metrics in cluster_metrics.items():
            report.append(f"Cluster {cluster_id}:")
            report.append(f"  Number of nodes: {len(metrics.nodes)}")
            report.append(f"  Center: ({metrics.center[0]:.1f}, {metrics.center[1]:.1f})")
            if metrics.selected_head:
                report.append(f"  Cluster head: Node {metrics.selected_head.id}")
                report.append(f"  CH position: ({metrics.selected_head.x:.1f}, {metrics.selected_head.y:.1f})")
                report.append(f"  CH energy level: {metrics.selected_head.energy_level:.3f}")
                report.append(f"  CH fuzzy score: {metrics.fuzzy_score:.3f}")
            else:
                report.append("  No cluster head selected")
            report.append("")
        
        # Recommendation
        report.append("RECOMMENDATION:")
        report.append(comparison['recommendation'])
        report.append("")
        
        return "\n".join(report)

