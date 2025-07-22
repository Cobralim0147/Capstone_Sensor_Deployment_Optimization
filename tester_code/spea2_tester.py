"""
SPEA2 Algorithm Performance Testing Framework

This module provides functionality for running multiple iterations of SPEA2 optimization
and analyzing their consistency and performance with the same structure as kmean_tester.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import time
import json
from datetime import datetime
import os
import sys
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
from main import SensorOptimizer
from configurations.config_file import OptimizationConfig


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class SPEA2Tester:
    """SPEA2 algorithm consistency and performance tester."""
    
    def __init__(self, config: OptimizationConfig, test_runs: int = 5):
        self.config = config
        self.test_runs = test_runs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_spea2_test(self) -> Dict[str, Any]:
        """Main function to orchestrate the entire testing process."""
        print(f"=== SPEA2 ALGORITHM CONSISTENCY TEST ({self.test_runs} runs) ===")
        print(f"Started at: {datetime.now()}")
        
        # Step 1: Run SPEA2 multiple times
        results = self._run_spea2_tests()
        
        if not results:
            return {'error': 'All SPEA2 runs failed'}
        
        # Step 2: Analyze consistency
        analysis = self._analyze_consistency(results)
        
        # Step 3: Save results
        self._save_results(results, analysis)
        
        return {
            'algorithm': 'SPEA2',
            'results': results,
            'analysis': analysis,
            'total_runs': self.test_runs,
            'successful_runs': len(results)
        }

    # =========================================================================
    # SPEA2 TESTING
    # =========================================================================
    
    def _run_spea2_tests(self) -> List[Dict]:
        """Coordinate multiple SPEA2 test runs."""
        results = []
        
        for run_id in range(self.test_runs):
            print(f"Run {run_id + 1}/{self.test_runs}", end=" -> ")
            
            result = self._run_single_spea2_test(run_id)
            if result is not None:
                results.append(result)
        
        print(f"\nCompleted: {len(results)}/{self.test_runs} successful runs")
        return results
    
    def _run_single_spea2_test(self, run_id: int) -> Optional[Dict]:
        """Run a single SPEA2 optimization test."""
        try:
            start_time = time.time()
            
            # Execute SPEA2 optimization
            optimizer, result = self._execute_spea2_optimization(run_id)
            execution_time = time.time() - start_time
            
            # Check for failure
            if result.X is None or len(result.X) == 0:
                print(f"❌ Failed: No solutions found")
                return None
            
            # Analyze solutions and get best one
            best_solution = self._get_best_solution(optimizer, result)
            if best_solution is None:
                print(f"❌ Failed: No valid solution analysis")
                return None
            
            # Build result dictionary
            result_dict = self._build_result_dict(
                run_id, best_solution, result, execution_time
            )
            
            print(f"✅ Coverage: {result_dict['coverage_rate']:.1f}%, Sensors: {result_dict['n_sensors']}")
            return result_dict
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def _execute_spea2_optimization(self, run_id: int) -> Tuple[SensorOptimizer, Any]:
        """Execute SPEA2 optimization algorithm."""
        # Create fresh optimizer for each run
        optimizer = SensorOptimizer(self.config)
        
        # Generate field and setup (with different random seed for variety)
        np.random.seed(42 + run_id)  # Different seed for each run
        field_data = optimizer.generate_field_environment()
        optimizer.setup_generated_field()
        
        # Run SPEA2 optimization
        result = optimizer.run_optimization()
        
        return optimizer, result
    
    def _get_best_solution(self, optimizer: SensorOptimizer, result: Any) -> Optional[Dict]:
        """Get the best solution from SPEA2 result."""
        try:
            # Analyze solutions using existing method
            solutions = optimizer.analyze_pareto_front(result)
            
            if not solutions:
                return None
            
            # Get best coverage solution (first one returned)
            solution_name, solution_idx, chromosome = solutions[0]
            
            # Get sensor positions and metrics
            sensor_positions = optimizer.problem.get_sensor_positions(chromosome)
            metrics = optimizer.problem.evaluate_solution(chromosome)
            
            return {
                'solution_name': solution_name,
                'solution_idx': solution_idx,
                'chromosome': chromosome,
                'sensor_positions': sensor_positions,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error in _get_best_solution: {e}")
            return None
    
    def _build_result_dict(self, run_id: int, best_solution: Dict, result: Any, 
                          execution_time: float) -> Dict:
        """Build comprehensive result dictionary."""
        metrics = best_solution['metrics']
        sensor_positions = best_solution['sensor_positions']
        
        return {
            'run_id': run_id + 1,
            'solution_name': best_solution['solution_name'],
            'n_sensors': int(metrics.n_sensors),
            'coverage_rate': float(metrics.coverage_rate),
            'over_coverage_rate': float(metrics.over_coverage_rate),
            'connectivity_rate': float(metrics.connectivity_rate),
            'count_rate': float(metrics.count_rate),
            'placement_rate': float(metrics.placement_rate),
            'execution_time': execution_time,
            'archive_size': int(len(result.X)),
            'sensor_positions': sensor_positions.tolist() if len(sensor_positions) > 0 else [],
            'chromosome': best_solution['chromosome'].tolist(),
            'converged': True,  # SPEA2 always converges in given generations
            'generations': self.config.deployment_generations
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
        
        # Analyze sensor placement consistency
        sensor_placement_analysis = self._analyze_sensor_placement_consistency(results)
        
        # Combine results
        analysis = {
            **basic_stats, 
            'consistency': consistency_checks,
            'sensor_placement': sensor_placement_analysis
        }
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _calculate_basic_statistics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate basic statistics for all metrics."""
        metrics = ['n_sensors', 'coverage_rate', 'connectivity_rate', 'execution_time', 'archive_size']
        basic_stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            basic_stats[metric] = self._calc_stats(values)
        
        return basic_stats
    
    def _perform_consistency_checks(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform detailed consistency checks across runs."""
        if not results or len(results[0].get('sensor_positions', [])) == 0:
            return self._empty_consistency_result()
        
        # Calculate coefficient of variation for key metrics
        coverage_cv = self._calculate_cv([r['coverage_rate'] for r in results])
        sensor_count_cv = self._calculate_cv([r['n_sensors'] for r in results])
        connectivity_cv = self._calculate_cv([r['connectivity_rate'] for r in results])
        
        # Check solution consistency
        chromosome_consistency = self._check_chromosome_consistency(results)
        position_consistency = self._check_position_consistency(results)
        
        return {
            'coverage_cv': coverage_cv,
            'sensor_count_cv': sensor_count_cv,
            'connectivity_cv': connectivity_cv,
            'chromosome_consistency': chromosome_consistency,
            'position_consistency': position_consistency,
            'is_consistent': (
                coverage_cv < 0.15 and 
                sensor_count_cv < 0.20 and 
                connectivity_cv < 0.15
            )
        }
    
    def _check_chromosome_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Check if chromosomes (binary solutions) are consistent."""
        if len(results) < 2:
            return {'identical_chromosomes': True, 'hamming_distances': []}
        
        chromosomes = [np.array(r['chromosome']) for r in results]
        base_chromosome = chromosomes[0]
        
        hamming_distances = []
        identical = True
        
        for chromosome in chromosomes[1:]:
            if len(chromosome) == len(base_chromosome):
                hamming_dist = np.sum(chromosome != base_chromosome)
                hamming_distances.append(int(hamming_dist))
                if hamming_dist > 0:
                    identical = False
            else:
                hamming_distances.append(len(base_chromosome))  # Maximum possible distance
                identical = False
        
        return {
            'identical_chromosomes': identical,
            'hamming_distances': hamming_distances,
            'avg_hamming_distance': float(np.mean(hamming_distances)) if hamming_distances else 0.0,
            'max_hamming_distance': int(np.max(hamming_distances)) if hamming_distances else 0
        }
    
    def _check_position_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Check if sensor positions are consistent across runs."""
        position_shifts = self._calculate_position_shifts(results)
        
        return {
            'avg_position_shift': float(np.mean(position_shifts)) if position_shifts else 0.0,
            'max_position_shift': float(np.max(position_shifts)) if position_shifts else 0.0,
            'position_shift_std': float(np.std(position_shifts)) if position_shifts else 0.0,
            'identical_positions': all(shift == 0.0 for shift in position_shifts)
        }
    
    def _calculate_position_shifts(self, results: List[Dict]) -> List[float]:
        """Calculate position shifts between runs."""
        if len(results) < 2:
            return [0.0]
        
        shifts = []
        base_positions = np.array(results[0]['sensor_positions'])
        
        for result in results[1:]:
            current_positions = np.array(result['sensor_positions'])
            
            if base_positions.shape == current_positions.shape and len(base_positions) > 0:
                # Sort positions to handle ordering differences
                base_sorted = np.sort(base_positions.view('f8,f8'), order=['f0', 'f1'], axis=0).view(float).reshape(-1, 2)
                current_sorted = np.sort(current_positions.view('f8,f8'), order=['f0', 'f1'], axis=0).view(float).reshape(-1, 2)
                shift = np.linalg.norm(base_sorted - current_sorted)
                shifts.append(shift)
            else:
                shifts.append(float('inf'))
        
        return shifts
    
    def _analyze_sensor_placement_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze sensor placement consistency using clustering."""
        print(f"\n🎯 ANALYZING SENSOR PLACEMENT CONSISTENCY...")
        
        # Collect all sensor positions
        all_positions = []
        run_positions = []
        
        for result in results:
            positions = np.array(result['sensor_positions'])
            all_positions.extend(positions)
            run_positions.append(positions)
        
        all_positions = np.array(all_positions)
        
        if len(all_positions) == 0:
            return {'error': 'No sensor positions to analyze'}
        
        # Clustering parameters
        position_tolerance = 2.0
        min_usage_threshold = 0.2
        
        # Use DBSCAN to find common positions
        clustering = DBSCAN(eps=position_tolerance, min_samples=max(1, int(len(results) * min_usage_threshold)))
        cluster_labels = clustering.fit_predict(all_positions)
        
        # Analyze clusters
        cluster_analysis = self._analyze_position_clusters(
            cluster_labels, all_positions, run_positions, results, position_tolerance
        )
        
        # Create deployment recommendations
        final_deployment = self._create_deployment_recommendations(cluster_analysis, results)
        
        placement_analysis = {
            'total_unique_clusters': len([c for c in cluster_analysis if c['cluster_id'] >= 0]),
            'common_positions': [c for c in cluster_analysis if c['usage_frequency'] >= min_usage_threshold],
            'all_clusters': cluster_analysis,
            'final_deployment': final_deployment,
            'analysis_parameters': {
                'position_tolerance': position_tolerance,
                'min_usage_threshold': min_usage_threshold,
                'total_runs_analyzed': len(results)
            }
        }
        
        self._print_placement_analysis_summary(placement_analysis)
        return placement_analysis
    
    def _analyze_position_clusters(self, cluster_labels: np.ndarray, all_positions: np.ndarray,
                                  run_positions: List[np.ndarray], results: List[Dict],
                                  position_tolerance: float) -> List[Dict]:
        """Analyze position clusters to find common sensor locations."""
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise points
        
        cluster_analysis = []
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_positions = all_positions[cluster_mask]
            cluster_center = np.mean(cluster_positions, axis=0)
            
            # Count runs with this position
            runs_with_position = 0
            position_variants = []
            
            for run_idx, positions in enumerate(run_positions):
                if len(positions) > 0:
                    distances = cdist([cluster_center], positions)[0]
                    closest_distance = np.min(distances)
                    
                    if closest_distance <= position_tolerance:
                        runs_with_position += 1
                        closest_idx = np.argmin(distances)
                        position_variants.append({
                            'run_id': results[run_idx]['run_id'],
                            'position': positions[closest_idx].tolist(),
                            'distance_from_center': float(closest_distance)
                        })
            
            usage_frequency = runs_with_position / len(results)
            
            cluster_info = {
                'cluster_id': int(label),
                'representative_position': cluster_center.tolist(),
                'usage_count': int(runs_with_position),
                'usage_frequency': float(usage_frequency),
                'total_occurrences': int(len(cluster_positions)),
                'position_variants': position_variants,
                'consistency_score': float(1.0 - (np.std(cdist([cluster_center], cluster_positions)[0]) / position_tolerance))
            }
            
            cluster_analysis.append(cluster_info)
        
        return sorted(cluster_analysis, key=lambda x: x['usage_frequency'], reverse=True)
    
    def _create_deployment_recommendations(self, cluster_analysis: List[Dict], 
                                         results: List[Dict]) -> Dict[str, Any]:
        """Create deployment recommendations based on analysis."""
        if not cluster_analysis:
            return {'error': 'No clusters found for recommendation'}
        
        # Find most representative run
        avg_coverage = np.mean([r['coverage_rate'] for r in results])
        avg_sensors = np.mean([r['n_sensors'] for r in results])
        
        best_run = min(results, key=lambda r: abs(r['coverage_rate'] - avg_coverage) + abs(r['n_sensors'] - avg_sensors) * 10)
        
        # Primary recommendation from common positions
        common_positions = [c for c in cluster_analysis if c['usage_frequency'] >= 0.2]
        primary_positions = [pos['representative_position'] for pos in common_positions]
        
        return {
            'primary_deployment': {
                'positions': primary_positions,
                'total_sensors': len(primary_positions),
                'description': 'Most frequently used sensor positions across all runs',
                'confidence_scores': [{
                    'position': pos['representative_position'],
                    'confidence_score': pos['usage_frequency']
                } for pos in common_positions]
            },
            'alternative_deployment': {
                'positions': best_run['sensor_positions'],
                'total_sensors': best_run['n_sensors'],
                'description': f'Representative deployment from Run {best_run["run_id"]} (closest to average)',
                'performance_metrics': {
                    'coverage_rate': best_run['coverage_rate'],
                    'connectivity_rate': best_run['connectivity_rate'],
                    'n_sensors': best_run['n_sensors']
                }
            },
            'deployment_statistics': {
                'avg_sensors_per_run': float(np.mean([r['n_sensors'] for r in results])),
                'sensor_count_range': [
                    int(np.min([r['n_sensors'] for r in results])),
                    int(np.max([r['n_sensors'] for r in results]))
                ],
                'position_consistency_score': float(np.mean([pos['usage_frequency'] for pos in common_positions])) if common_positions else 0.0
            }
        }
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if not values or np.mean(values) == 0:
            return 0.0
        return float(np.std(values) / np.mean(values))
    
    def _empty_consistency_result(self) -> Dict[str, Any]:
        """Return empty consistency result for failed cases."""
        return {
            'coverage_cv': 0.0,
            'sensor_count_cv': 0.0,
            'connectivity_cv': 0.0,
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
        print(f"\n=== SPEA2 CONSISTENCY ANALYSIS ===")
        
        # Basic statistics
        print(f"Coverage:     {analysis['coverage_rate']['mean']:.1f}% ± {analysis['coverage_rate']['std']:.1f}%")
        print(f"Sensors:      {analysis['n_sensors']['mean']:.1f} ± {analysis['n_sensors']['std']:.1f}")
        print(f"Connectivity: {analysis['connectivity_rate']['mean']:.1f}% ± {analysis['connectivity_rate']['std']:.1f}%")
        print(f"Exec Time:    {analysis['execution_time']['mean']:.3f}s ± {analysis['execution_time']['std']:.3f}s")
        print(f"Archive Size: {analysis['archive_size']['mean']:.1f} ± {analysis['archive_size']['std']:.1f}")
        
        # Consistency results
        consistency = analysis['consistency']
        print(f"\nConsistency Results:")
        print(f"  Coverage CV:          {consistency['coverage_cv']:.4f} ({'✅ Consistent' if consistency['coverage_cv'] < 0.15 else '⚠️ Variable'})")
        print(f"  Sensor Count CV:      {consistency['sensor_count_cv']:.4f} ({'✅ Consistent' if consistency['sensor_count_cv'] < 0.20 else '⚠️ Variable'})")
        print(f"  Connectivity CV:      {consistency['connectivity_cv']:.4f} ({'✅ Consistent' if consistency['connectivity_cv'] < 0.15 else '⚠️ Variable'})")
        print(f"  Overall Consistency:  {'🎯 Consistent' if consistency['is_consistent'] else '⚠️ Variable'}")
        
        # Chromosome consistency
        if 'chromosome_consistency' in consistency:
            chrom_cons = consistency['chromosome_consistency']
            print(f"  Identical Solutions:  {'✅ Yes' if chrom_cons['identical_chromosomes'] else '❌ No'}")
            print(f"  Avg Hamming Distance: {chrom_cons.get('avg_hamming_distance', 0):.1f}")
        
        # Position consistency
        if 'position_consistency' in consistency:
            pos_cons = consistency['position_consistency']
            print(f"  Identical Positions:  {'✅ Yes' if pos_cons['identical_positions'] else '❌ No'}")
            print(f"  Avg Position Shift:   {pos_cons['avg_position_shift']:.4f}")
            print(f"  Max Position Shift:   {pos_cons.get('max_position_shift', 0):.4f}")
    
    def _print_placement_analysis_summary(self, placement_analysis: Dict) -> None:
        """Print sensor placement analysis summary."""
        print(f"\n📍 SENSOR PLACEMENT CONSISTENCY ANALYSIS:")
        
        common_positions = placement_analysis.get('common_positions', [])
        final_deployment = placement_analysis.get('final_deployment', {})
        
        print(f"Total Position Clusters: {placement_analysis.get('total_unique_clusters', 0)}")
        print(f"Common Positions (≥20% usage): {len(common_positions)}")
        
        if common_positions:
            print(f"\n🎯 MOST FREQUENTLY USED POSITIONS:")
            for i, pos in enumerate(common_positions[:5]):  # Show top 5
                print(f"  {i+1}. Position ({pos['representative_position'][0]:.1f}, {pos['representative_position'][1]:.1f})")
                print(f"     Usage: {pos['usage_count']}/{placement_analysis['analysis_parameters']['total_runs_analyzed']} runs ({pos['usage_frequency']:.1%})")
        
        if 'primary_deployment' in final_deployment:
            primary = final_deployment['primary_deployment']
            print(f"\n✅ RECOMMENDED DEPLOYMENT:")
            print(f"   Total Sensors: {primary['total_sensors']}")
            print(f"   Based on: {primary['description']}")

    # =========================================================================
    # DATA PERSISTENCE
    # =========================================================================
    
    def _save_results(self, results: List[Dict], analysis: Dict) -> None:
        """Coordinate saving all result files."""
        os.makedirs('test_results', exist_ok=True)
        
        file_paths = {
            'summary': self._save_summary_results(results),
            'detailed': self._save_detailed_results(results, analysis),
            'deployments': self._save_deployment_data(results)
        }
        
        self._print_save_summary(file_paths)
    
    def _save_summary_results(self, results: List[Dict]) -> str:
        """Save summary metrics to CSV."""
        summary_df = pd.DataFrame([{
            'run_id': r['run_id'],
            'solution_name': r['solution_name'],
            'n_sensors': r['n_sensors'],
            'coverage_rate': r['coverage_rate'],
            'connectivity_rate': r['connectivity_rate'],
            'execution_time': r['execution_time'],
            'archive_size': r['archive_size'],
            'converged': r['converged']
        } for r in results])
        
        file_path = f'test_results/spea2_summary_{self.timestamp}.csv'
        summary_df.to_csv(file_path, index=False)
        return file_path
    
    def _save_detailed_results(self, results: List[Dict], analysis: Dict) -> str:
        """Save detailed results to JSON."""
        output_data = {
            'algorithm': 'SPEA2',
            'timestamp': self.timestamp,
            'config': {
                'test_runs': self.test_runs,
                'population_size': self.config.deployment_pop_size,
                'archive_size': self.config.deployment_archive_size,
                'max_generations': self.config.deployment_generations,
                'field_length': self.config.environment_field_length,
                'field_width': self.config.environment_field_width,
                'max_sensors': self.config.deployment_max_sensors,
                'sensor_range': self.config.deployment_sensor_range,
                'comm_range': self.config.deployment_comm_range
            },
            'results': results,
            'analysis': analysis
        }
        
        file_path = f'test_results/spea2_detailed_{self.timestamp}.json'
        with open(file_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        return file_path
    
    def _save_deployment_data(self, results: List[Dict]) -> str:
        """Save deployment data to CSV."""
        deployment_data = []
        
        for result in results:
            for i, position in enumerate(result['sensor_positions']):
                deployment_data.append({
                    'run_id': result['run_id'],
                    'sensor_id': i,
                    'x': position[0],
                    'y': position[1],
                    'solution_name': result['solution_name']
                })
        
        if deployment_data:
            deployment_df = pd.DataFrame(deployment_data)
            file_path = f'test_results/spea2_deployments_{self.timestamp}.csv'
            deployment_df.to_csv(file_path, index=False)
            return file_path
        return 'No deployment data'
    
    def _print_save_summary(self, file_paths: Dict[str, str]) -> None:
        """Print summary of saved files."""
        print(f"\n💾 Results saved:")
        for key, path in file_paths.items():
            print(f"  {key.title()}: {path}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_results(self, test_results: Dict) -> None:
        """Create and save visualization of SPEA2 results."""
        if not test_results['results']:
            print("No results to visualize")
            return
        
        results = test_results['results']
        fig, axes = self._setup_visualization_grid()
        
        # Plot various metrics
        self._plot_coverage_distribution(axes[0, 0], results)
        self._plot_sensor_count_distribution(axes[0, 1], results)
        self._plot_connectivity_distribution(axes[0, 2], results)
        self._plot_execution_time_distribution(axes[1, 0], results)
        self._plot_coverage_vs_sensors(axes[1, 1], results)
        self._plot_performance_over_runs(axes[1, 2], results)
        self._plot_sensor_placements(axes[2, 0], results, test_results.get('analysis', {}))
        self._plot_position_frequency(axes[2, 1], test_results.get('analysis', {}))
        self._plot_consistency_metrics(axes[2, 2], test_results.get('analysis', {}))
        
        self._finalize_visualization(fig)
    
    def _setup_visualization_grid(self):
        """Setup matplotlib figure and axes for visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))  # Increased size
        fig.suptitle(f'SPEA2 Algorithm Performance Analysis ({self.test_runs} runs)', 
                    fontsize=16, fontweight='bold', y=0.98)  # Position title properly
        return fig, axes
    
    def _plot_coverage_distribution(self, ax, results: List[Dict]) -> None:
        """Plot coverage rate distribution."""
        coverage_rates = [r['coverage_rate'] for r in results]
        ax.hist(coverage_rates, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(coverage_rates), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(coverage_rates):.1f}%')
        ax.set_title('Coverage Rate Distribution')
        ax.set_xlabel('Coverage Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_sensor_count_distribution(self, ax, results: List[Dict]) -> None:
        """Plot sensor count distribution."""
        sensor_counts = [r['n_sensors'] for r in results]
        ax.hist(sensor_counts, bins=range(min(sensor_counts), max(sensor_counts) + 2), 
               alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(sensor_counts), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(sensor_counts):.1f}')
        ax.set_title('Sensor Count Distribution')
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_connectivity_distribution(self, ax, results: List[Dict]) -> None:
        """Plot connectivity rate distribution."""
        connectivity_rates = [r['connectivity_rate'] for r in results]
        ax.hist(connectivity_rates, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(connectivity_rates), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(connectivity_rates):.1f}%')
        ax.set_title('Connectivity Rate Distribution')
        ax.set_xlabel('Connectivity Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_execution_time_distribution(self, ax, results: List[Dict]) -> None:
        """Plot execution time distribution."""
        exec_times = [r['execution_time'] for r in results]
        ax.hist(exec_times, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(np.mean(exec_times), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(exec_times):.1f}s')
        ax.set_title('Execution Time Distribution')
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_coverage_vs_sensors(self, ax, results: List[Dict]) -> None:
        """Plot coverage vs sensor count scatter."""
        sensor_counts = [r['n_sensors'] for r in results]
        coverage_rates = [r['coverage_rate'] for r in results]
        
        ax.scatter(sensor_counts, coverage_rates, alpha=0.6, color='red', s=50)
        ax.set_title('Coverage vs Sensor Count')
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('Coverage Rate (%)')
        ax.grid(True, alpha=0.3)
        
        # Add trend line if there's variation
        if len(set(sensor_counts)) > 1:
            z = np.polyfit(sensor_counts, coverage_rates, 1)
            p = np.poly1d(z)
            ax.plot(sensor_counts, p(sensor_counts), "r--", alpha=0.8, linewidth=2)
    
    def _plot_performance_over_runs(self, ax, results: List[Dict]) -> None:
        """Plot performance consistency over runs."""
        run_ids = [r['run_id'] for r in results]
        coverage_rates = [r['coverage_rate'] for r in results]
        
        ax.plot(run_ids, coverage_rates, 'o-', alpha=0.7, color='green', label='Coverage Rate')
        ax.set_title('Performance Consistency Over Runs')
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Coverage Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_sensor_placements(self, ax, results: List[Dict], analysis: Dict) -> None:
        """Plot sensor placements with common positions highlighted."""
        # Plot all positions lightly
        for result in results:
            positions = np.array(result['sensor_positions'])
            if len(positions) > 0:
                ax.scatter(positions[:, 0], positions[:, 1], alpha=0.3, s=20, color='lightgray')
        
        # Highlight common positions
        sensor_placement = analysis.get('sensor_placement', {})
        common_positions = sensor_placement.get('common_positions', [])
        
        if common_positions:
            for pos_info in common_positions:
                pos = pos_info['representative_position']
                frequency = pos_info['usage_frequency']
                size = 50 + frequency * 200
                ax.scatter(pos[0], pos[1], s=size, alpha=0.8, color='red', 
                          edgecolor='black', linewidth=2)
        
        ax.set_title('Sensor Placement Map\n(Red: Common, Gray: All)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_position_frequency(self, ax, analysis: Dict) -> None:
        """Plot position usage frequency."""
        sensor_placement = analysis.get('sensor_placement', {})
        common_positions = sensor_placement.get('common_positions', [])
        
        if not common_positions:
            ax.text(0.5, 0.5, 'No common positions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Usage Frequency')
            return
        
        top_positions = common_positions[:8]
        frequencies = [pos['usage_frequency'] * 100 for pos in top_positions]
        labels = [f"Pos {i+1}" for i in range(len(top_positions))]
        
        bars = ax.bar(range(len(frequencies)), frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Position Rank')
        ax.set_ylabel('Usage Frequency (%)')
        ax.set_title('Top Common Position Usage')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3)
        
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{freq:.0f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_consistency_metrics(self, ax, analysis: Dict) -> None:
        """Plot consistency metrics."""
        consistency = analysis.get('consistency', {})
        
        if not consistency:
            ax.text(0.5, 0.5, 'No consistency data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consistency Metrics')
            return
        
        metrics = {
            'Coverage': (1 - consistency.get('coverage_cv', 1)) * 100,
            'Sensors': (1 - consistency.get('sensor_count_cv', 1)) * 100,
            'Connectivity': (1 - consistency.get('connectivity_cv', 1)) * 100
        }
        
        # Ensure values are between 0 and 100
        for key in metrics:
            metrics[key] = max(0, min(100, metrics[key]))
        
        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Consistency Score (%)')
        ax.set_title('Algorithm Consistency')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _finalize_visualization(self, fig) -> None:
        """Finalize and save visualization with proper spacing."""
        # Adjust subplot spacing to prevent overlap
        plt.subplots_adjust(
            left=0.08,      # Left margin
            bottom=0.08,    # Bottom margin
            right=0.95,     # Right margin
            top=0.92,       # Top margin (leave space for main title)
            wspace=0.35,    # Width spacing between subplots
            hspace=0.45     # Height spacing between subplots
        )
        
        # Save visualization
        viz_file = f'test_results/spea2_visualization_{self.timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Visualization saved: {viz_file}")

def main():
    """Main function to run SPEA2 consistency test."""
    config = OptimizationConfig()
    test_runs = getattr(config, 'test_num_runs', 5)  # Default to 5 if not in config
    
    tester = SPEA2Tester(config, test_runs)
    results = tester.run_spea2_test()
    
    if 'error' not in results:
        # Create visualization
        tester.visualize_results(results)
    
    print(f"\n{'='*50}")
    print("SPEA2 CONSISTENCY TEST COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()