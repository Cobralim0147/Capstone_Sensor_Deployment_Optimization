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
        """Run K-means clustering consistency test."""
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
    
    def _get_sensor_positions(self) -> Optional[np.ndarray]:
        """Get sensor positions from SPEA2."""
        try:
            optimizer = SensorOptimizer(self.config)
            field_data = optimizer.generate_field_environment()
            optimizer.setup_generated_field()
            
            np.random.seed(42)  # Fixed seed for consistency
            result = optimizer.run_optimization()
            
            if result.X is not None and len(result.X) > 0:
                best_chromosome = result.X[0]
                sensor_positions = optimizer.problem.get_sensor_positions(best_chromosome)
                return sensor_positions
            else:
                return None
                
        except Exception as e:
            print(f"Error getting sensor positions: {e}")
            return None
    
    def _run_kmeans_tests(self, sensor_positions: np.ndarray) -> List[Dict]:
        """Run K-means clustering multiple times."""
        results = []
        
        for run_id in range(self.test_runs):
            print(f"Run {run_id + 1}/{self.test_runs}", end=" -> ")
            
            try:
                start_time = time.time()
                
                kmeans = KMeansClustering()
                assignments, cluster_heads, clustering_info = kmeans.perform_clustering(
                    sensor_positions=sensor_positions,
                    n_clusters=None,
                    random_state=None  # Same random state for consistency
                )
                
                execution_time = time.time() - start_time
                
                result = {
                    'run_id': run_id + 1,
                    'n_clusters': int(clustering_info['n_clusters_final']),
                    'total_sse': float(clustering_info['total_sse']),
                    'connectivity_rate': float(clustering_info['connectivity_report']['connectivity_rate']),
                    'execution_time': execution_time,
                    'iterations': int(clustering_info['iterations']),
                    'converged': bool(clustering_info['converged']),
                    'assignments': assignments.tolist(),
                    'cluster_heads': cluster_heads.tolist(),
                    'cluster_head_positions': sensor_positions[cluster_heads].tolist()
                }
                
                results.append(result)
                print(f"✅ Clusters: {result['n_clusters']}, SSE: {result['total_sse']:.2f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        print(f"\nCompleted: {len(results)}/{self.test_runs} successful runs")
        return results
    
    def _analyze_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze clustering consistency."""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Extract metrics
        n_clusters = [r['n_clusters'] for r in results]
        total_sse = [r['total_sse'] for r in results]
        connectivity_rate = [r['connectivity_rate'] for r in results]
        execution_time = [r['execution_time'] for r in results]
        
        # Calculate statistics
        analysis = {
            'n_clusters': self._calc_stats(n_clusters),
            'total_sse': self._calc_stats(total_sse),
            'connectivity_rate': self._calc_stats(connectivity_rate),
            'execution_time': self._calc_stats(execution_time),
        }
        
        # Consistency evaluation
        cluster_count_consistent = np.std(n_clusters) == 0
        sse_cv = np.std(total_sse) / np.mean(total_sse) if np.mean(total_sse) > 0 else 0
        
        analysis['consistency'] = {
            'cluster_count_consistent': cluster_count_consistent,
            'sse_coefficient_variation': sse_cv,
            'is_consistent': cluster_count_consistent and sse_cv < 0.05
        }
        
        # Print summary
        self._print_summary(analysis)
        return analysis
    
    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
        }
    
    def _print_summary(self, analysis: Dict) -> None:
        """Print analysis summary."""
        print(f"\n=== CONSISTENCY ANALYSIS ===")
        print(f"Clusters:     {analysis['n_clusters']['mean']:.0f} ± {analysis['n_clusters']['std']:.0f}")        
        print(f"Exec Time:    {analysis['execution_time']['mean']:.2f}s ± {analysis['execution_time']['std']:.2f}s")
        
        consistency = analysis['consistency']
        print(f"\nConsistency:")
        print(f"  Cluster Count: {'✅ Consistent' if consistency['cluster_count_consistent'] else '❌ Variable'}")
        print(f"  SSE Variation: {consistency['sse_coefficient_variation']:.4f} {'✅ Low' if consistency['sse_coefficient_variation'] < 0.05 else '⚠️ High'}")
        print(f"  Overall:       {'✅ CONSISTENT' if consistency['is_consistent'] else '⚠️ There is variation'}")
    
    def _save_results(self, results: List[Dict], analysis: Dict, sensor_positions: np.ndarray) -> None:
        """Save results to CSV files."""
        os.makedirs('test_results', exist_ok=True)
        
        # 1. Summary metrics CSV
        summary_df = pd.DataFrame([{
            'run_id': r['run_id'],
            'n_clusters': r['n_clusters'],
            'total_sse': r['total_sse'],
            'connectivity_rate': r['connectivity_rate'],
            'execution_time': r['execution_time'],
            'iterations': r['iterations'],
            'converged': r['converged']
        } for r in results])
        
        summary_file = f'test_results/kmeans_summary_{self.timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # 2. Sensor positions CSV
        positions_df = pd.DataFrame(sensor_positions, columns=['x', 'y'])
        positions_df['sensor_id'] = positions_df.index
        positions_file = f'test_results/kmeans_sensors_{self.timestamp}.csv'
        positions_df.to_csv(positions_file, index=False)
        
        # 3. Detailed assignments CSV
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
        assignments_file = f'test_results/kmeans_assignments_{self.timestamp}.csv'
        assignments_df.to_csv(assignments_file, index=False)
        
        # 4. Cluster heads CSV
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
        heads_file = f'test_results/kmeans_cluster_heads_{self.timestamp}.csv'
        heads_df.to_csv(heads_file, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"  Summary:           {summary_file}")
        print(f"  Sensor Positions:  {positions_file}")
        print(f"  Assignments:       {assignments_file}")
        print(f"  Cluster Heads:     {heads_file}")
    
    def visualize_results(self, results: List[Dict], sensor_positions: np.ndarray) -> None:
        """Create simple visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Plot only first 4 runs to match 2x2 grid
        for i, result in enumerate(results[:9]):  # Changed from [:10] to [:4]
            ax = axes[i // 3, i % 3]
            
            assignments = np.array(result['assignments'])
            cluster_heads = np.array(result['cluster_heads'])
            
            # Plot sensors by cluster
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
            for cluster_id in np.unique(assignments):
                cluster_sensors = sensor_positions[assignments == cluster_id]
                color = colors[cluster_id % len(colors)]
                ax.scatter(cluster_sensors[:, 0], cluster_sensors[:, 1], 
                        c=color, alpha=0.6, s=30)
            
            # Plot cluster heads
            head_positions = sensor_positions[cluster_heads]
            ax.scatter(head_positions[:, 0], head_positions[:, 1], 
                    c='red', marker='X', s=100, edgecolors='black')
            
            ax.set_title(f'Run {result["run_id"]}: {result["n_clusters"]} clusters')
            ax.grid(True, alpha=0.3)
    
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