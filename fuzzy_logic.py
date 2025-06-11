"""
Fuzzy Logic Cluster Head Selection and Validation System

This module implements a fuzzy logic system to validate K-means clustering results
and select optimal cluster heads based on five key factors: energy level, node centrality,
cluster density, distance to base station, and probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SensorNode:
    """Represents a sensor node with its properties."""
    id: int
    x: float
    y: float
    energy_level: float  # Remaining energy (0-1)
    cluster_id: int = -1
    is_cluster_head: bool = False


@dataclass
class ClusterMetrics:
    """Container for cluster evaluation metrics."""
    cluster_id: int
    center: Tuple[float, float]
    nodes: List[SensorNode]
    density: int
    selected_head: SensorNode
    fuzzy_score: float


class FuzzyClusterValidator:
    """
    Fuzzy logic system for cluster head selection and clustering validation.
    """
    
    def __init__(self, base_station_pos: Tuple[float, float] = (150, 50)):
        """
        Initialize the fuzzy logic system.
        
        Parameters:
        -----------
        base_station_pos : Tuple[float, float]
            Position of the base station (x, y)
        """
        self.base_station_pos = base_station_pos
        self.fuzzy_system = None
        self._setup_fuzzy_system()
    
    def _setup_fuzzy_system(self):
        """Setup the Mamdani fuzzy inference system."""
        # Define input variables with their ranges
        energy_level = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'energy_level')
        node_centrality = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'node_centrality')
        cluster_density = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cluster_density')
        distance_bs = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'distance_bs')
        
        # Define output variable
        probability = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'probability')
        
        # Define membership functions for Energy Level (EL)
        energy_level['low'] = fuzz.trapmf(energy_level.universe, [0, 0, 0.2, 0.4])
        energy_level['medium'] = fuzz.trimf(energy_level.universe, [0.2, 0.5, 0.8])
        energy_level['high'] = fuzz.trapmf(energy_level.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Define membership functions for Node Centrality (NC)
        node_centrality['close'] = fuzz.trapmf(node_centrality.universe, [0, 0, 0.2, 0.4])
        node_centrality['adequate'] = fuzz.trimf(node_centrality.universe, [0.2, 0.5, 0.8])
        node_centrality['far'] = fuzz.trapmf(node_centrality.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Define membership functions for Cluster Density (CD)
        cluster_density['low'] = fuzz.trapmf(cluster_density.universe, [0, 0, 0.2, 0.4])
        cluster_density['medium'] = fuzz.trimf(cluster_density.universe, [0.2, 0.5, 0.8])
        cluster_density['high'] = fuzz.trapmf(cluster_density.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Define membership functions for Distance to Base Station (DoBS)
        distance_bs['close'] = fuzz.trapmf(distance_bs.universe, [0, 0, 0.2, 0.4])
        distance_bs['adequate'] = fuzz.trimf(distance_bs.universe, [0.2, 0.5, 0.8])
        distance_bs['far'] = fuzz.trapmf(distance_bs.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Define membership functions for Probability (P) - output
        probability['weak'] = fuzz.trapmf(probability.universe, [0, 0, 0.1, 0.25])
        probability['lower_medium'] = fuzz.trimf(probability.universe, [0.1, 0.3, 0.5])
        probability['medium'] = fuzz.trapmf(probability.universe, [0.25, 0.4, 0.6, 0.75])
        probability['higher_medium'] = fuzz.trimf(probability.universe, [0.5, 0.7, 0.9])
        probability['strong'] = fuzz.trapmf(probability.universe, [0.75, 0.9, 1.0, 1.0])
        
        # Store antecedents and consequent
        self.energy_level = energy_level
        self.node_centrality = node_centrality
        self.cluster_density = cluster_density
        self.distance_bs = distance_bs
        self.probability = probability
        
        # Define fuzzy rules
        self._define_fuzzy_rules()
    
    def _define_fuzzy_rules(self):
        """Define the fuzzy rules for cluster head selection."""
        # Rule base for cluster head selection
        rules = []
        
        # High priority rules (high energy + good conditions)
        rules.append(ctrl.Rule(
            self.energy_level['high'] & self.node_centrality['close'] & 
            self.cluster_density['high'] & self.distance_bs['close'],
            self.probability['strong']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['high'] & self.node_centrality['close'] & 
            self.cluster_density['medium'] & self.distance_bs['close'],
            self.probability['higher_medium']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['high'] & self.node_centrality['adequate'] & 
            self.cluster_density['high'] & self.distance_bs['close'],
            self.probability['higher_medium']
        ))
        
        # Medium priority rules
        rules.append(ctrl.Rule(
            self.energy_level['medium'] & self.node_centrality['close'] & 
            self.cluster_density['high'] & self.distance_bs['close'],
            self.probability['medium']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['high'] & self.node_centrality['close'] & 
            self.cluster_density['low'] & self.distance_bs['close'],
            self.probability['medium']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['medium'] & self.node_centrality['adequate'] & 
            self.cluster_density['medium'] & self.distance_bs['adequate'],
            self.probability['lower_medium']
        ))
        
        # Low priority rules
        rules.append(ctrl.Rule(
            self.energy_level['low'] & self.node_centrality['far'] & 
            self.cluster_density['low'] & self.distance_bs['far'],
            self.probability['weak']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['low'] & self.node_centrality['close'] & 
            self.cluster_density['high'] & self.distance_bs['close'],
            self.probability['lower_medium']
        ))
        
        # Additional comprehensive rules
        rules.append(ctrl.Rule(
            self.energy_level['medium'] & self.node_centrality['far'] & 
            self.cluster_density['low'] & self.distance_bs['far'],
            self.probability['weak']
        ))
        
        rules.append(ctrl.Rule(
            self.energy_level['high'] & self.node_centrality['far'] & 
            self.cluster_density['medium'] & self.distance_bs['adequate'],
            self.probability['lower_medium']
        ))
        
        # Create control system
        self.fuzzy_system = ctrl.ControlSystem(rules)
        self.fuzzy_simulation = ctrl.ControlSystemSimulation(self.fuzzy_system)
    
    def calculate_node_centrality(self, node: SensorNode, cluster_nodes: List[SensorNode]) -> float:
        """
        Calculate node centrality within a cluster.
        
        Parameters:
        -----------
        node : SensorNode
            The node to calculate centrality for
        cluster_nodes : List[SensorNode]
            All nodes in the cluster
            
        Returns:
        --------
        float
            Normalized centrality value (0-1, where 0 is most central)
        """
        if len(cluster_nodes) <= 1:
            return 0.0
        
        total_distance = 0.0
        for other_node in cluster_nodes:
            if other_node.id != node.id:
                distance = euclidean([node.x, node.y], [other_node.x, other_node.y])
                total_distance += distance
        
        avg_distance = total_distance / (len(cluster_nodes) - 1)
        
        # Calculate max possible distance for normalization
        max_distance = 0.0
        for i, node1 in enumerate(cluster_nodes):
            for j, node2 in enumerate(cluster_nodes):
                if i != j:
                    dist = euclidean([node1.x, node1.y], [node2.x, node2.y])
                    max_distance = max(max_distance, dist)
        
        # Normalize (invert so that closer to center = higher value)
        if max_distance > 0:
            centrality = 1.0 - (avg_distance / max_distance)
            return max(0.0, min(1.0, centrality))
        return 1.0
    
    def calculate_distance_to_base_station(self, node: SensorNode) -> float:
        """
        Calculate normalized distance from node to base station.
        
        Parameters:
        -----------
        node : SensorNode
            The sensor node
            
        Returns:
        --------
        float
            Normalized distance (0-1, where 0 is closest)
        """
        distance = euclidean([node.x, node.y], self.base_station_pos)
        
        # Normalize based on maximum possible distance in field
        # Assuming field dimensions are roughly 200x100 based on generate_field.py
        max_distance = euclidean([0, 0], [200, 100])
        
        normalized = distance / max_distance
        return min(1.0, normalized)
    
    def calculate_cluster_density(self, cluster_nodes: List[SensorNode], max_cluster_size: int) -> float:
        """
        Calculate normalized cluster density.
        
        Parameters:
        -----------
        cluster_nodes : List[SensorNode]
            Nodes in the cluster
        max_cluster_size : int
            Maximum cluster size across all clusters
            
        Returns:
        --------
        float
            Normalized density (0-1)
        """
        if max_cluster_size == 0:
            return 0.0
        
        density = len(cluster_nodes) / max_cluster_size
        return min(1.0, density)
    
    def evaluate_node_for_cluster_head(self, node: SensorNode, cluster_nodes: List[SensorNode], 
                                     max_cluster_size: int) -> float:
        """
        Evaluate a node's suitability as cluster head using fuzzy logic.
        
        Parameters:
        -----------
        node : SensorNode
            Node to evaluate
        cluster_nodes : List[SensorNode]
            All nodes in the cluster
        max_cluster_size : int
            Maximum cluster size for normalization
            
        Returns:
        --------
        float
            Fuzzy probability score (0-1)
        """
        try:
            # Calculate input variables
            energy = node.energy_level
            centrality = self.calculate_node_centrality(node, cluster_nodes)
            density = self.calculate_cluster_density(cluster_nodes, max_cluster_size)
            distance = 1.0 - self.calculate_distance_to_base_station(node)  # Invert for closer = better
            
            # Set inputs to fuzzy system
            self.fuzzy_simulation.input['energy_level'] = energy
            self.fuzzy_simulation.input['node_centrality'] = centrality
            self.fuzzy_simulation.input['cluster_density'] = density
            self.fuzzy_simulation.input['distance_bs'] = distance
            
            # Compute fuzzy output
            self.fuzzy_simulation.compute()
            
            return self.fuzzy_simulation.output['probability']
            
        except Exception as e:
            print(f"Error evaluating node {node.id}: {e}")
            return 0.0
    
    def select_cluster_heads(self, clusters: Dict[int, List[SensorNode]]) -> Dict[int, ClusterMetrics]:
        """
        Select cluster heads for each cluster using fuzzy logic.
        
        Parameters:
        -----------
        clusters : Dict[int, List[SensorNode]]
            Dictionary mapping cluster_id to list of nodes
            
        Returns:
        --------
        Dict[int, ClusterMetrics]
            Dictionary mapping cluster_id to cluster metrics
        """
        cluster_metrics = {}
        max_cluster_size = max(len(nodes) for nodes in clusters.values()) if clusters else 1
        
        for cluster_id, nodes in clusters.items():
            if not nodes:
                continue
            
            # Calculate cluster center
            center_x = np.mean([node.x for node in nodes])
            center_y = np.mean([node.y for node in nodes])
            center = (center_x, center_y)
            
            # Evaluate each node for cluster head suitability
            best_node = None
            best_score = -1.0
            
            for node in nodes:
                score = self.evaluate_node_for_cluster_head(node, nodes, max_cluster_size)
                if score > best_score:
                    best_score = score
                    best_node = node
            
            # Mark as cluster head
            if best_node:
                best_node.is_cluster_head = True
            
            # Create cluster metrics
            metrics = ClusterMetrics(
                cluster_id=cluster_id,
                center=center,
                nodes=nodes,
                density=len(nodes),
                selected_head=best_node,
                fuzzy_score=best_score
            )
            
            cluster_metrics[cluster_id] = metrics
        
        return cluster_metrics
    
    def validate_clustering_quality(self, cluster_metrics: Dict[int, ClusterMetrics]) -> Dict[str, float]:
        """
        Validate the overall clustering quality using fuzzy scores.
        
        Parameters:
        -----------
        cluster_metrics : Dict[int, ClusterMetrics]
            Cluster metrics from fuzzy evaluation
            
        Returns:
        --------
        Dict[str, float]
            Quality metrics for the clustering
        """
        if not cluster_metrics:
            return {'overall_quality': 0.0, 'avg_fuzzy_score': 0.0, 'cluster_balance': 0.0}
        
        # Calculate average fuzzy score
        fuzzy_scores = [metrics.fuzzy_score for metrics in cluster_metrics.values()]
        avg_fuzzy_score = np.mean(fuzzy_scores)
        
        # Calculate cluster balance (how evenly distributed are the clusters)
        cluster_sizes = [metrics.density for metrics in cluster_metrics.values()]
        cluster_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes)) if np.mean(cluster_sizes) > 0 else 0.0
        cluster_balance = max(0.0, min(1.0, cluster_balance))
        
        # Calculate overall quality as weighted combination
        overall_quality = 0.7 * avg_fuzzy_score + 0.3 * cluster_balance
        
        return {
            'overall_quality': overall_quality,
            'avg_fuzzy_score': avg_fuzzy_score,
            'cluster_balance': cluster_balance,
            'num_clusters': len(cluster_metrics)
        }
    
    def plot_membership_functions(self):
        """Plot the membership functions for visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fuzzy Membership Functions', fontsize=16)
        
        # Energy Level
        axes[0, 0].plot(self.energy_level.universe, 
                       fuzz.interp_membership(self.energy_level.universe, self.energy_level['low'].mf, self.energy_level.universe),
                       'r', label='Low')
        axes[0, 0].plot(self.energy_level.universe,
                       fuzz.interp_membership(self.energy_level.universe, self.energy_level['medium'].mf, self.energy_level.universe),
                       'g', label='Medium')
        axes[0, 0].plot(self.energy_level.universe,
                       fuzz.interp_membership(self.energy_level.universe, self.energy_level['high'].mf, self.energy_level.universe),
                       'b', label='High')
        axes[0, 0].set_title('Energy Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Node Centrality
        axes[0, 1].plot(self.node_centrality.universe,
                       fuzz.interp_membership(self.node_centrality.universe, self.node_centrality['close'].mf, self.node_centrality.universe),
                       'r', label='Close')
        axes[0, 1].plot(self.node_centrality.universe,
                       fuzz.interp_membership(self.node_centrality.universe, self.node_centrality['adequate'].mf, self.node_centrality.universe),
                       'g', label='Adequate')
        axes[0, 1].plot(self.node_centrality.universe,
                       fuzz.interp_membership(self.node_centrality.universe, self.node_centrality['far'].mf, self.node_centrality.universe),
                       'b', label='Far')
        axes[0, 1].set_title('Node Centrality')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Cluster Density
        axes[0, 2].plot(self.cluster_density.universe,
                       fuzz.interp_membership(self.cluster_density.universe, self.cluster_density['low'].mf, self.cluster_density.universe),
                       'r', label='Low')
        axes[0, 2].plot(self.cluster_density.universe,
                       fuzz.interp_membership(self.cluster_density.universe, self.cluster_density['medium'].mf, self.cluster_density.universe),
                       'g', label='Medium')
        axes[0, 2].plot(self.cluster_density.universe,
                       fuzz.interp_membership(self.cluster_density.universe, self.cluster_density['high'].mf, self.cluster_density.universe),
                       'b', label='High')
        axes[0, 2].set_title('Cluster Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Distance to Base Station
        axes[1, 0].plot(self.distance_bs.universe,
                       fuzz.interp_membership(self.distance_bs.universe, self.distance_bs['close'].mf, self.distance_bs.universe),
                       'r', label='Close')
        axes[1, 0].plot(self.distance_bs.universe,
                       fuzz.interp_membership(self.distance_bs.universe, self.distance_bs['adequate'].mf, self.distance_bs.universe),
                       'g', label='Adequate')
        axes[1, 0].plot(self.distance_bs.universe,
                       fuzz.interp_membership(self.distance_bs.universe, self.distance_bs['far'].mf, self.distance_bs.universe),
                       'b', label='Far')
        axes[1, 0].set_title('Distance to Base Station')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Probability (Output)
        axes[1, 1].plot(self.probability.universe,
                       fuzz.interp_membership(self.probability.universe, self.probability['weak'].mf, self.probability.universe),
                       'r', label='Weak')
        axes[1, 1].plot(self.probability.universe,
                       fuzz.interp_membership(self.probability.universe, self.probability['lower_medium'].mf, self.probability.universe),
                       'orange', label='Lower Medium')
        axes[1, 1].plot(self.probability.universe,
                       fuzz.interp_membership(self.probability.universe, self.probability['medium'].mf, self.probability.universe),
                       'g', label='Medium')
        axes[1, 1].plot(self.probability.universe,
                       fuzz.interp_membership(self.probability.universe, self.probability['higher_medium'].mf, self.probability.universe),
                       'cyan', label='Higher Medium')
        axes[1, 1].plot(self.probability.universe,
                       fuzz.interp_membership(self.probability.universe, self.probability['strong'].mf, self.probability.universe),
                       'b', label='Strong')
        axes[1, 1].set_title('Probability (Output)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Hide the last subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_clustering_results(self, cluster_metrics: Dict[int, ClusterMetrics], 
                                   quality_metrics: Dict[str, float]):
        """
        Visualize the clustering results with cluster heads and quality metrics.
        
        Parameters:
        -----------
        cluster_metrics : Dict[int, ClusterMetrics]
            Cluster metrics from fuzzy evaluation
        quality_metrics : Dict[str, float]
            Overall quality metrics
        """
        plt.figure(figsize=(12, 8))
        
        # Color map for clusters
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_metrics)))
        
        for i, (cluster_id, metrics) in enumerate(cluster_metrics.items()):
            # Plot cluster nodes
            x_coords = [node.x for node in metrics.nodes]
            y_coords = [node.y for node in metrics.nodes]
            
            plt.scatter(x_coords, y_coords, c=[colors[i]], alpha=0.6, 
                       s=50, label=f'Cluster {cluster_id}')
            
            # Highlight cluster head
            if metrics.selected_head:
                plt.scatter(metrics.selected_head.x, metrics.selected_head.y, 
                           c='red', marker='*', s=200, edgecolors='black', linewidth=2)
                plt.annotate(f'CH{cluster_id}\nScore: {metrics.fuzzy_score:.2f}',
                           (metrics.selected_head.x, metrics.selected_head.y),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot cluster center
            plt.scatter(metrics.center[0], metrics.center[1], 
                       c='black', marker='x', s=100, alpha=0.8)
        
        # Plot base station
        plt.scatter(self.base_station_pos[0], self.base_station_pos[1], 
                   c='blue', marker='s', s=150, label='Base Station')
        
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title(f'Fuzzy Logic Cluster Head Selection Results\n'
                 f'Overall Quality: {quality_metrics["overall_quality"]:.3f}, '
                 f'Avg Fuzzy Score: {quality_metrics["avg_fuzzy_score"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
