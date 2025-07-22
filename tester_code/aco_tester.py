"""
Simple Ant Colony Optimization (ACO) Consistency Tester

This module tests ACO path optimization consistency by:
1. Getting cluster head positions from SPEA2+K-means (deterministic)
2. Running ACO path optimization multiple times with different random seeds
3. Analyzing path consistency and saving CSV reports
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
from algorithms.ant_colony import AntColonyOptimizer
from configurations.config_file import OptimizationConfig


class ACOTester:
    """Simple ACO algorithm consistency tester."""
    
    def __init__(self, config: OptimizationConfig, test_runs: int = 20):
        self.config = config
        self.test_runs = test_runs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_aco_test(self) -> Dict[str, Any]:
        """Main function to orchestrate the entire testing process."""
        print(f"=== ACO PATH OPTIMIZATION CONSISTENCY TEST ({self.test_runs} runs) ===")
        print(f"Started at: {datetime.now()}")
        
        # Step 1: Get cluster head positions
        cluster_data = self._get_cluster_head_data()
        if cluster_data is None:
            return {'error': 'Failed to get cluster head data'}
        
        print(f"Got {len(cluster_data['cluster_heads'])} cluster head positions")
        
        # Step 2: Run ACO multiple times
        results = self._run_aco_tests(cluster_data)
        
        # Step 3: Analyze consistency
        analysis = self._analyze_consistency(results)
        
        # Step 4: Save results
        self._save_results(results, analysis, cluster_data)
        
        return {
            'results': results,
            'analysis': analysis,
            'cluster_data': cluster_data
        }

    # =========================================================================
    # CLUSTER HEAD DATA ACQUISITION
    # =========================================================================
    
    def _get_cluster_head_data(self) -> Optional[Dict]:
        """Get cluster head positions from SPEA2 + K-means pipeline."""
        try:
            # Step 1: Get sensor positions from SPEA2
            optimizer = self._setup_optimizer()
            result = self._run_spea2_optimization(optimizer)
            
            if result.X is None or len(result.X) == 0:
                return None
            
            # Step 2: Get deployed sensors
            best_chromosome = result.X[0]
            sensor_positions = optimizer.problem.get_sensor_positions(best_chromosome)
            
            # Step 3: Run K-means clustering
            kmeans = KMeansClustering()
            assignments, cluster_heads_indices, clustering_info = kmeans.perform_clustering(
                sensor_positions=sensor_positions,
                n_clusters=None,
                random_state=42
            )
            
            if clustering_info.get('clustering_failed', False):
                return None
            
            cluster_head_positions = sensor_positions[cluster_heads_indices]
            field_data = optimizer.field_data
            
            return {
                'cluster_heads': cluster_head_positions,
                'sensor_positions': sensor_positions,
                'field_data': field_data
            }
            
        except Exception as e:
            print(f"Error getting cluster head data: {e}")
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
    # ACO TESTING
    # =========================================================================
    
    def _run_aco_tests(self, cluster_data: Dict) -> List[Dict]:
        """Coordinate multiple ACO test runs."""
        results = []
        
        for run_id in range(self.test_runs):
            print(f"Run {run_id + 1}/{self.test_runs}", end=" -> ")
            
            result = self._run_single_aco_test(run_id, cluster_data)
            if result is not None:
                results.append(result)
        
        print(f"\nCompleted: {len(results)}/{self.test_runs} successful runs")
        return results
    
    def _run_single_aco_test(self, run_id: int, cluster_data: Dict) -> Optional[Dict]:
        """Run a single ACO path optimization test."""
        try:
            start_time = time.time()
            
            # Execute ACO optimization with different random seed
            np.random.seed(42 + run_id)
            aco = AntColonyOptimizer(
                cluster_heads=cluster_data['cluster_heads'],
                field_data=cluster_data['field_data']
            )
            
            # Run path optimization
            optimal_path, path_distance, convergence_history = aco.optimize()
            execution_time = time.time() - start_time
            
            # Get full path coordinates for visualization
            path_coordinates = aco.get_full_path_for_plotting(optimal_path)
            
            # Build result dictionary
            result = {
                'run_id': run_id + 1,
                'optimal_path': optimal_path,
                'path_coordinates': path_coordinates.tolist() if len(path_coordinates) > 0 else [],
                'total_distance': float(path_distance),
                'path_length': int(len(path_coordinates)),
                'execution_time': execution_time,
                'n_cluster_heads': len(cluster_data['cluster_heads']),
                'converged': True
            }
            
            print(f"✅ Distance: {result['total_distance']:.2f}m, Path Length: {result['path_length']}")
            return result
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

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
        metrics = ['total_distance', 'path_length', 'execution_time']
        basic_stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            basic_stats[metric] = self._calc_stats(values)
        
        return basic_stats
    
    def _perform_consistency_checks(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform detailed consistency checks across runs."""
        if not results:
            return self._empty_consistency_result()
        
        # Calculate coefficient of variation for key metrics
        distance_cv = self._calculate_cv([r['total_distance'] for r in results])
        path_length_cv = self._calculate_cv([r['path_length'] for r in results])
        execution_time_cv = self._calculate_cv([r['execution_time'] for r in results])
        
        # Check path consistency
        path_consistency = self._check_path_consistency(results)
        
        return {
            'distance_cv': distance_cv,
            'path_length_cv': path_length_cv,
            'execution_time_cv': execution_time_cv,
            'identical_paths': path_consistency,
            'is_consistent': (
                distance_cv < 0.15 and  # 15% variation threshold
                path_length_cv < 0.20
            )
        }
    
    def _check_path_consistency(self, results: List[Dict]) -> bool:
        """Check if paths are identical across runs."""
        if len(results) < 2:
            return True
        
        base_path = results[0]['optimal_path']
        for result in results[1:]:
            if result['optimal_path'] != base_path:
                return False
        return True
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if not values or np.mean(values) == 0:
            return 0.0
        return float(np.std(values) / np.mean(values))
    
    def _empty_consistency_result(self) -> Dict[str, Any]:
        """Return empty consistency result for failed cases."""
        return {
            'distance_cv': 0.0,
            'path_length_cv': 0.0,
            'execution_time_cv': 0.0,
            'identical_paths': False,
            'is_consistent': False,
            'note': 'No valid results to compare'
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
        print(f"Total Distance:   {analysis['total_distance']['mean']:.2f}m ± {analysis['total_distance']['std']:.2f}m")
        print(f"Path Length:      {analysis['path_length']['mean']:.1f} ± {analysis['path_length']['std']:.1f} waypoints")
        print(f"Exec Time:        {analysis['execution_time']['mean']:.3f}s ± {analysis['execution_time']['std']:.3f}s")
        
        # Consistency results
        consistency = analysis['consistency']
        print(f"\nConsistency Results:")
        print(f"  Identical Paths:      {'✅ Yes' if consistency['identical_paths'] else '❌ No'}")
        print(f"  Distance CV:          {consistency['distance_cv']:.4f} ({'Low' if consistency['distance_cv'] < 0.15 else 'High'} variability)")
        print(f"  Path Length CV:       {consistency['path_length_cv']:.4f} ({'Low' if consistency['path_length_cv'] < 0.20 else 'High'} variability)")
        print(f"  Execution Time CV:    {consistency['execution_time_cv']:.4f} ({'Low' if consistency['execution_time_cv'] < 0.25 else 'High'} variability)")
        print(f"  Overall Consistency:  {'🎯 Consistent' if consistency['is_consistent'] else '⚠️ Variable'}")
        
        # Best performance
        best_distance = analysis['total_distance']['min']
        worst_distance = analysis['total_distance']['max']
        improvement = ((worst_distance - best_distance) / worst_distance * 100) if worst_distance > 0 else 0.0
        
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Best Distance:        {best_distance:.2f}m")
        print(f"Worst Distance:       {worst_distance:.2f}m")
        print(f"Improvement Potential: {improvement:.1f}%")

    # =========================================================================
    # DATA PERSISTENCE
    # =========================================================================
    
    def _save_results(self, results: List[Dict], analysis: Dict, cluster_data: Dict) -> None:
        """Coordinate saving all result files."""
        os.makedirs('test_results', exist_ok=True)
        
        file_paths = {
            'summary': self._save_summary_results(results),
            'paths': self._save_path_data(results),
            'waypoints': self._save_waypoint_data(results),
            'cluster_heads': self._save_cluster_head_data(cluster_data)
        }
        
        self._print_save_summary(file_paths)
    
    def _save_summary_results(self, results: List[Dict]) -> str:
        """Save summary metrics to CSV."""
        summary_df = pd.DataFrame([{
            'run_id': r['run_id'],
            'total_distance': r['total_distance'],
            'path_length': r['path_length'],
            'execution_time': r['execution_time'],
            'converged': r['converged'],
            'n_cluster_heads': r['n_cluster_heads']
        } for r in results])
        
        file_path = f'test_results/aco_summary_{self.timestamp}.csv'
        summary_df.to_csv(file_path, index=False)
        return file_path
    
    def _save_path_data(self, results: List[Dict]) -> str:
        """Save path sequence data to CSV."""
        path_data = []
        
        for result in results:
            for step, node in enumerate(result['optimal_path']):
                path_data.append({
                    'run_id': result['run_id'],
                    'step': step,
                    'node_id': node,
                    'total_distance': result['total_distance']
                })
        
        if path_data:
            path_df = pd.DataFrame(path_data)
            file_path = f'test_results/aco_paths_{self.timestamp}.csv'
            path_df.to_csv(file_path, index=False)
            return file_path
        return 'No path data'
    
    def _save_waypoint_data(self, results: List[Dict]) -> str:
        """Save waypoint coordinate data to CSV."""
        waypoint_data = []
        
        for result in results:
            for step, coord in enumerate(result['path_coordinates']):
                waypoint_data.append({
                    'run_id': result['run_id'],
                    'step': step,
                    'x': coord[0],
                    'y': coord[1],
                    'total_distance': result['total_distance']
                })
        
        if waypoint_data:
            waypoint_df = pd.DataFrame(waypoint_data)
            file_path = f'test_results/aco_waypoints_{self.timestamp}.csv'
            waypoint_df.to_csv(file_path, index=False)
            return file_path
        return 'No waypoint data'
    
    def _save_cluster_head_data(self, cluster_data: Dict) -> str:
        """Save cluster head positions to CSV."""
        cluster_heads_df = pd.DataFrame(cluster_data['cluster_heads'], columns=['x', 'y'])
        cluster_heads_df['cluster_head_id'] = cluster_heads_df.index
        
        file_path = f'test_results/aco_cluster_heads_{self.timestamp}.csv'
        cluster_heads_df.to_csv(file_path, index=False)
        return file_path
    
    def _print_save_summary(self, file_paths: Dict[str, str]) -> None:
        """Print summary of saved files."""
        print(f"\n💾 Results saved:")
        print(f"  Summary:       {file_paths['summary']}")
        print(f"  Paths:         {file_paths['paths']}")
        print(f"  Waypoints:     {file_paths['waypoints']}")
        print(f"  Cluster Heads: {file_paths['cluster_heads']}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_results(self, results: List[Dict], cluster_data: Dict) -> None:
        """Create and save visualization of ACO results."""
        fig, axes = self._setup_visualization_grid()
        
        # Plot first 9 runs
        for i, result in enumerate(results[:9]):
            ax = axes[i // 3, i % 3]
            self._plot_single_path_result(ax, result, cluster_data)
        
        # Finalize and save visualization
        self._finalize_visualization(fig)
    
    def _setup_visualization_grid(self):
        """Setup matplotlib figure and axes for visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        return fig, axes
    
    def _plot_single_path_result(self, ax, result: Dict, cluster_data: Dict) -> None:
        """Plot a single path result on given axis."""
        cluster_heads = cluster_data['cluster_heads']
        path_coords = np.array(result['path_coordinates'])
        
        # Plot cluster heads
        ax.scatter(cluster_heads[:, 0], cluster_heads[:, 1], 
                  c='red', marker='s', s=100, alpha=0.8, label='Cluster Heads')
        
        # Plot base station
        if len(path_coords) > 0:
            ax.scatter(path_coords[0, 0], path_coords[0, 1], 
                      c='green', marker='o', s=100, label='Base Station')
        
        # Plot path
        if len(path_coords) > 1:
            ax.plot(path_coords[:, 0], path_coords[:, 1], 'b-', 
                   linewidth=2, alpha=0.7, label='Path')
        
        # Format axis
        ax.set_title(f'Run {result["run_id"]}: {result["total_distance"]:.1f}m')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.set_aspect('equal')
    
    def _finalize_visualization(self, fig) -> None:
        """Finalize and save visualization."""
        plt.tight_layout()
        viz_file = f'test_results/aco_visualization_{self.timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Visualization saved: {viz_file}")


def main():
    """Main function to run ACO consistency test."""
    config = OptimizationConfig()
    test_runs = getattr(config, 'test_num_runs', 2)  # Default to 20 if not in config

    tester = ACOTester(config, test_runs)
    results = tester.run_aco_test()
    
    if 'error' not in results:
        # Create visualization
        tester.visualize_results(results['results'], results['cluster_data'])
    
    print(f"\n{'='*50}")
    print("ACO CONSISTENCY TEST COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()