"""
Simple K-means Clustering Consistency Tester

This module tests K-means clustering consistency by:
1. Getting sensor positions from SPEA2 (deterministic)
2. Running K-means clustering 30 times with same random state
3. Analyzing consistency and saving CSV reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
from main import SensorOptimizer
from algorithms.k_mean import KMeansClustering
from configurations.config_file import OptimizationConfig


class KMeansTester:
    """Simple K-means clustering consistency tester."""
    
    def __init__(self, config: OptimizationConfig, test_runs: int = 30):
        self.config = config
        self.test_runs = test_runs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_kmeans_test(self) -> Dict[str, Any]:
        """Main function to orchestrate the entire testing process."""
        print(f"=== K-MEANS CLUSTERING CONSISTENCY TEST ({self.test_runs} runs) ===")
        print(f"Started at: {datetime.now()}")
        
        # Step 1: Get sensor positions from SPEA2
        sensor_positions = self._get_sensor_positions()
        if sensor_positions is None:
            return {'error': 'Failed to get sensor positions'}
        
        print(f"Got {len(sensor_positions)} sensor positions")
        
        # Step 2: Run K-means multiple times
        results = self._run_kmeans_tests(sensor_positions)
        
        # Step 3: Analyze consistency
        analysis = self._analyze_consistency(results)
        
        # Step 4: Save results
        self._save_results(results, analysis, sensor_positions)
        
        return {
            'results': results,
            'analysis': analysis,
            'sensor_positions': sensor_positions.tolist()
        }

    # =========================================================================
    # SENSOR POSITION ACQUISITION
    # =========================================================================
    
    def _get_sensor_positions(self) -> Optional[np.ndarray]:
        """Get sensor positions from SPEA2 optimization."""
        try:
            optimizer = self._setup_optimizer()
            result = self._run_spea2_optimization(optimizer)
            
            if result.X is not None and len(result.X) > 0:
                best_chromosome = result.X[0]
                sensor_positions = optimizer.problem.get_sensor_positions(best_chromosome)
                return sensor_positions
            else:
                return None
                
        except Exception as e:
            print(f"Error getting sensor positions: {e}")
            return None
    
    def _setup_optimizer(self) -> SensorOptimizer:
        """Setup and configure the SPEA2 optimizer."""
        optimizer = SensorOptimizer(self.config)
        field_data = optimizer.generate_field_environment()
        optimizer.setup_generated_field()
        return optimizer
    
    def _run_spea2_optimization(self, optimizer: SensorOptimizer):
        """Run SPEA2 optimization with fixed seed."""
        np.random.seed(42)  # Fixed seed for consistency
        return optimizer.run_optimization()

    # =========================================================================
    # K-MEANS TESTING
    # =========================================================================
    
    def _run_kmeans_tests(self, sensor_positions: np.ndarray) -> List[Dict]:
        """Coordinate multiple K-means test runs."""
        results = []
        
        for run_id in range(self.test_runs):
            print(f"Run {run_id + 1}/{self.test_runs}", end=" -> ")
            
            result = self._run_single_kmeans_test(run_id, sensor_positions)
            if result is not None:
                results.append(result)
        
        print(f"\nCompleted: {len(results)}/{self.test_runs} successful runs")
        return results
    
    def _run_single_kmeans_test(self, run_id: int, sensor_positions: np.ndarray) -> Optional[Dict]:
        """Run a single K-means clustering test."""
        try:
            start_time = time.time()
            
            # Run K-means clustering
            assignments, cluster_heads, clustering_info = self._execute_kmeans_clustering(sensor_positions)
            execution_time = time.time() - start_time
            
            # Check for failure
            if clustering_info.get('clustering_failed', False):
                print(f"❌ Failed: {clustering_info.get('reason', 'Unknown error')}")
                return None
            
            # Calculate connectivity rate
            connectivity_rate = self._calculate_connectivity_rate(
                sensor_positions, assignments, cluster_heads
            )
            
            # Build result dictionary
            result = self._build_result_dict(
                run_id, assignments, cluster_heads, clustering_info, 
                sensor_positions, connectivity_rate, execution_time
            )
            
            print(f"✅ Clusters: {result['n_clusters']}, Connectivity: {result['connectivity_rate']:.1f}%")
            return result
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def _execute_kmeans_clustering(self, sensor_positions: np.ndarray):
        """Execute K-means clustering algorithm."""
        kmeans = KMeansClustering()
        return kmeans.perform_clustering(
            sensor_positions=sensor_positions,
            n_clusters=None,
            random_state=42  # Fixed random state for consistency testing
        )
    
    def _calculate_connectivity_rate(self, sensor_positions: np.ndarray, 
                                   assignments: np.ndarray, cluster_heads: np.ndarray) -> float:
        """Calculate connectivity rate for clustering result."""
        if len(cluster_heads) > 0 and len(assignments) > 0:
            kmeans = KMeansClustering()
            connectivity_analysis = kmeans.analyze_cluster_connectivity(
                sensor_positions, assignments, cluster_heads
            )
            total_sensors = len(sensor_positions)
            total_connected = sum([cluster['connected_to_head'] for cluster in connectivity_analysis.values()])
            return (total_connected / total_sensors) * 100 if total_sensors > 0 else 0.0
        else:
            return 0.0
    
    def _build_result_dict(self, run_id: int, assignments: np.ndarray, cluster_heads: np.ndarray,
                          clustering_info: Dict, sensor_positions: np.ndarray, 
                          connectivity_rate: float, execution_time: float) -> Dict:
        """Build comprehensive result dictionary."""
        return {
            'run_id': run_id + 1,
            'n_clusters': int(clustering_info.get('n_clusters_final', len(cluster_heads))),
            'connectivity_rate': float(connectivity_rate),
            'execution_time': execution_time,
            'iterations': int(clustering_info.get('iterations', 0)),
            'converged': bool(clustering_info.get('converged', True)),
            'assignments': assignments.tolist() if len(assignments) > 0 else [],
            'cluster_heads': cluster_heads.tolist() if len(cluster_heads) > 0 else [],
            'cluster_head_positions': sensor_positions[cluster_heads].tolist() if len(cluster_heads) > 0 else [],
            'best_sse': float(clustering_info.get('best_sse', 0.0))
        }

    # =========================================================================
    # CONSISTENCY ANALYSIS
    # =========================================================================
    
    def _analyze_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Main consistency analysis coordinator."""
        if not results:
            return {'error': 'No results to analyze'}

        # Calculate basic statistics
        basic_stats = self._calculate_basic_statistics(results)
        
        # Perform consistency checks
        consistency_checks = self._perform_consistency_checks(results)
        
        # Combine results
        analysis = {**basic_stats, 'consistency': consistency_checks}
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _calculate_basic_statistics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate basic statistics for all metrics."""
        metrics = ['n_clusters', 'connectivity_rate', 'execution_time']
        basic_stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            basic_stats[metric] = self._calc_stats(values)
        
        return basic_stats
    
    def _perform_consistency_checks(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform detailed consistency checks across runs."""
        if not results or len(results[0].get('assignments', [])) == 0:
            return self._empty_consistency_result()
        
        base_result = results[0]
        base_assignments = np.array(base_result['assignments'])
        base_cluster_heads = set(base_result['cluster_heads'])
        base_centroids = np.array(base_result['cluster_head_positions'])
        
        # Check consistency across all runs
        assignment_consistency = self._check_assignment_consistency(results[1:], base_assignments)
        head_consistency = self._check_head_consistency(results[1:], base_cluster_heads)
        centroid_shifts = self._calculate_centroid_shifts(results[1:], base_centroids)
        
        return {
            'same_assignments_across_runs': assignment_consistency,
            'same_cluster_heads_across_runs': head_consistency,
            'average_centroid_position_shift': float(np.mean(centroid_shifts)) if centroid_shifts else 0.0,
            'max_centroid_shift': float(np.max(centroid_shifts)) if centroid_shifts else 0.0,
            'centroid_shift_std': float(np.std(centroid_shifts)) if centroid_shifts else 0.0
        }
    
    def _check_assignment_consistency(self, results: List[Dict], base_assignments: np.ndarray) -> bool:
        """Check if sensor assignments are consistent across runs."""
        for result in results:
            current_assignments = np.array(result['assignments'])
            if not np.array_equal(base_assignments, current_assignments):
                return False
        return True
    
    def _check_head_consistency(self, results: List[Dict], base_cluster_heads: set) -> bool:
        """Check if cluster heads are consistent across runs."""
        for result in results:
            current_heads = set(result['cluster_heads'])
            if base_cluster_heads != current_heads:
                return False
        return True
    
    def _calculate_centroid_shifts(self, results: List[Dict], base_centroids: np.ndarray) -> List[float]:
        """Calculate centroid position shifts across runs."""
        shifts = []
        for result in results:
            current_centroids = np.array(result['cluster_head_positions'])
            
            if base_centroids.shape == current_centroids.shape:
                # Sort centroids to handle cluster ordering differences
                base_sorted = np.sort(base_centroids, axis=0)
                current_sorted = np.sort(current_centroids, axis=0)
                shift = np.linalg.norm(base_sorted - current_sorted)
                shifts.append(shift)
            else:
                shifts.append(float('inf'))
        
        return shifts
    
    def _empty_consistency_result(self) -> Dict[str, Any]:
        """Return empty consistency result for failed cases."""
        return {
            'same_assignments_across_runs': False,
            'same_cluster_heads_across_runs': False,
            'average_centroid_position_shift': 0.0,
            'note': 'No valid clustering results to compare'
        }
    
    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values."""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'cv': 0.0}
        
        values_array = np.array(values)
        mean_val = float(np.mean(values_array))
        
        return {
            'mean': mean_val,
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'cv': float(np.std(values_array) / mean_val) if mean_val > 0 else 0.0
        }
    
    def _print_analysis_summary(self, analysis: Dict) -> None:
        """Print comprehensive analysis summary."""
        print(f"\n=== CONSISTENCY ANALYSIS ===")
        
        # Basic statistics
        print(f"Clusters:     {analysis['n_clusters']['mean']:.0f} ± {analysis['n_clusters']['std']:.1f}")
        print(f"Connectivity: {analysis['connectivity_rate']['mean']:.1f}% ± {analysis['connectivity_rate']['std']:.1f}%")
        print(f"Exec Time:    {analysis['execution_time']['mean']:.3f}s ± {analysis['execution_time']['std']:.3f}s")
        
        # Consistency results
        consistency = analysis['consistency']
        print(f"\nConsistency Results:")
        print(f"  Same Assignments:     {'✅ Yes' if consistency['same_assignments_across_runs'] else '❌ No'}")
        print(f"  Same Cluster Heads:   {'✅ Yes' if consistency['same_cluster_heads_across_runs'] else '❌ No'}")
        print(f"  Avg Centroid Shift:   {consistency['average_centroid_position_shift']:.4f}")
        print(f"  Max Centroid Shift:   {consistency.get('max_centroid_shift', 0.0):.4f}")  # Always print this
        
        # Additional detailed statistics
        if 'centroid_shift_std' in consistency:
            print(f"  Centroid Shift Std:   {consistency['centroid_shift_std']:.4f}")
        
        # Overall assessment
        is_perfectly_consistent = (
            consistency['same_assignments_across_runs'] and 
            consistency['same_cluster_heads_across_runs'] and
            consistency['average_centroid_position_shift'] == 0.0
        )
        
        print(f"\n  Overall Consistency:  {'🎯 Perfect' if is_perfectly_consistent else '⚠️  Variable'}")
        
        # Additional insights
        cv_clusters = analysis['n_clusters'].get('cv', 0.0)
        cv_connectivity = analysis['connectivity_rate'].get('cv', 0.0)
        cv_time = analysis['execution_time'].get('cv', 0.0)
        
        print(f"\n=== VARIABILITY METRICS ===")
        print(f"Cluster Count CV:     {cv_clusters:.4f} ({'Low' if cv_clusters < 0.1 else 'High'} variability)")
        print(f"Connectivity CV:      {cv_connectivity:.4f} ({'Low' if cv_connectivity < 0.1 else 'High'} variability)")
        print(f"Execution Time CV:    {cv_time:.4f} ({'Low' if cv_time < 0.1 else 'High'} variability)")

    # =========================================================================
    # DATA PERSISTENCE
    # =========================================================================
    
    def _save_results(self, results: List[Dict], analysis: Dict, sensor_positions: np.ndarray) -> None:
        """Coordinate saving all result files."""
        os.makedirs('test_results', exist_ok=True)
        
        file_paths = {
            'summary': self._save_summary_results(results),
            'sensors': self._save_sensor_positions(sensor_positions),
            'assignments': self._save_assignments_data(results, sensor_positions),
            'heads': self._save_cluster_heads_data(results)
        }
        
        self._print_save_summary(file_paths)
    
    def _save_summary_results(self, results: List[Dict]) -> str:
        """Save summary metrics to CSV."""
        summary_df = pd.DataFrame([{
            'run_id': r['run_id'],
            'n_clusters': r['n_clusters'],
            'connectivity_rate': r['connectivity_rate'],
            'execution_time': r['execution_time'],
            'iterations': r['iterations'],
            'converged': r['converged'],
            'best_sse': r['best_sse']
        } for r in results])
        
        file_path = f'test_results/kmeans_summary_{self.timestamp}.csv'
        summary_df.to_csv(file_path, index=False)
        return file_path
    
    def _save_sensor_positions(self, sensor_positions: np.ndarray) -> str:
        """Save sensor positions to CSV."""
        positions_df = pd.DataFrame(sensor_positions, columns=['x', 'y'])
        positions_df['sensor_id'] = positions_df.index
        
        file_path = f'test_results/kmeans_sensors_{self.timestamp}.csv'
        positions_df.to_csv(file_path, index=False)
        return file_path
    
    def _save_assignments_data(self, results: List[Dict], sensor_positions: np.ndarray) -> str:
        """Save detailed sensor assignments to CSV."""
        assignments_data = []
        
        for result in results:
            for sensor_id, cluster_id in enumerate(result['assignments']):
                assignments_data.append({
                    'run_id': result['run_id'],
                    'sensor_id': sensor_id,
                    'cluster_id': cluster_id,
                    'sensor_x': sensor_positions[sensor_id][0],
                    'sensor_y': sensor_positions[sensor_id][1],
                    'is_cluster_head': sensor_id in result['cluster_heads']
                })
        
        assignments_df = pd.DataFrame(assignments_data)
        file_path = f'test_results/kmeans_assignments_{self.timestamp}.csv'
        assignments_df.to_csv(file_path, index=False)
        return file_path
    
    def _save_cluster_heads_data(self, results: List[Dict]) -> str:
        """Save cluster head information to CSV."""
        heads_data = []
        
        for result in results:
            for cluster_id, head_sensor_id in enumerate(result['cluster_heads']):
                head_pos = result['cluster_head_positions'][cluster_id]
                heads_data.append({
                    'run_id': result['run_id'],
                    'cluster_id': cluster_id,
                    'head_sensor_id': head_sensor_id,
                    'head_x': head_pos[0],
                    'head_y': head_pos[1]
                })
        
        heads_df = pd.DataFrame(heads_data)
        file_path = f'test_results/kmeans_cluster_heads_{self.timestamp}.csv'
        heads_df.to_csv(file_path, index=False)
        return file_path
    
    def _print_save_summary(self, file_paths: Dict[str, str]) -> None:
        """Print summary of saved files."""
        print(f"\n💾 Results saved:")
        print(f"  Summary:           {file_paths['summary']}")
        print(f"  Sensor Positions:  {file_paths['sensors']}")
        print(f"  Assignments:       {file_paths['assignments']}")
        print(f"  Cluster Heads:     {file_paths['heads']}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_results(self, results: List[Dict], sensor_positions: np.ndarray) -> None:
        """Create and save visualization of clustering results."""
        fig, axes = self._setup_visualization_grid()
        
        # Plot first 9 runs
        for i, result in enumerate(results[:9]):
            ax = axes[i // 3, i % 3]
            self._plot_single_clustering_result(ax, result, sensor_positions)
        
        # Finalize and save visualization
        self._finalize_visualization(fig)
    
    def _setup_visualization_grid(self):
        """Setup matplotlib figure and axes for visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        return fig, axes
    
    def _plot_single_clustering_result(self, ax, result: Dict, sensor_positions: np.ndarray) -> None:
        """Plot a single clustering result on given axis."""
        assignments = np.array(result['assignments'])
        cluster_heads = np.array(result['cluster_heads'])
        
        # Define colors for clusters
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot sensors by cluster
        for cluster_id in np.unique(assignments):
            cluster_sensors = sensor_positions[assignments == cluster_id]
            color = colors[cluster_id % len(colors)]
            ax.scatter(cluster_sensors[:, 0], cluster_sensors[:, 1], 
                      c=color, alpha=0.6, s=30, label=f'Cluster {cluster_id}')
        
        # Plot cluster heads
        if len(cluster_heads) > 0:
            head_positions = sensor_positions[cluster_heads]
            ax.scatter(head_positions[:, 0], head_positions[:, 1], 
                      c='red', marker='X', s=100, edgecolors='black', label='Heads')
        
        # Format axis
        ax.set_title(f'Run {result["run_id"]}: {result["n_clusters"]} clusters')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    def _finalize_visualization(self, fig) -> None:
        """Finalize and save visualization."""
        plt.tight_layout()
        viz_file = f'test_results/kmeans_visualization_{self.timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Visualization saved: {viz_file}")


def main():
    """Main function to run K-means consistency test."""
    config = OptimizationConfig()
    test_runs = config.test_num_runs  
    
    tester = KMeansTester(config, test_runs)
    results = tester.run_kmeans_test()
    
    if 'error' not in results:
        # Create visualization
        tester.visualize_results(results['results'], np.array(results['sensor_positions']))
    
    print(f"\n{'='*50}")
    print("K-MEANS CONSISTENCY TEST COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()