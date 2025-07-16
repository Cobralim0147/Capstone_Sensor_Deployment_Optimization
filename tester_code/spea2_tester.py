"""
Algorithm Performance Testing Framework

This module provides functionality for running multiple iterations of optimization algorithms
and analyzing their consistency and performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import time
import json
from datetime import datetime
import os
import sys
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Add parent directory to Python path to import main2
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your existing modules
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

        
class AlgorithmTester:
    """Framework for testing algorithm consistency and performance."""
    
    def __init__(self, config: OptimizationConfig, test_runs: int = 5):
        self.config = config
        self.test_runs = test_runs
        self.results = []
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_spea2_tests(self) -> Dict[str, Any]:
        """
        Run SPEA2 algorithm multiple times and collect results.
        
        Returns:
            Dict containing all test results and analysis
        """
        print(f"=== SPEA2 ALGORITHM TESTING ({self.test_runs} runs) ===")
        print(f"Test started at: {datetime.now()}")
        
        results = []
        failed_runs = 0
        
        for run_id in range(self.test_runs):
            print(f"\n--- Run {run_id + 1}/{self.test_runs} ---")
            
            try:
                # Create fresh optimizer for each run
                optimizer = SensorOptimizer(self.config)
                
                # Setup and run optimization
                start_time = time.time()
                
                # Generate field and setup
                field_data = optimizer.generate_field_environment()
                optimizer.setup_generated_field()
                
                # Run SPEA2 optimization
                result = optimizer.run_optimization()
                
                if result.X is not None and len(result.X) > 0:
                    # Analyze solutions
                    solutions = optimizer.analyze_pareto_front(result)
                    
                    if solutions:
                        # Get best coverage solution for analysis
                        name, idx, chromosome = solutions[0]
                        sensors = optimizer.problem.get_sensor_positions(chromosome)
                        metrics = optimizer.problem.evaluate_solution(chromosome)

                        print(f"\nAnalyzing {name} solution with {len(sensors)} sensors")
                        
                        # Record run results
                        run_result = {
                            'run_id': int(run_id + 1),
                            'success': True,
                            'execution_time': float(time.time() - start_time),
                            'n_sensors': int(metrics.n_sensors),
                            'coverage_rate': float(metrics.coverage_rate),
                            'over_coverage_rate': float(metrics.over_coverage_rate), 
                            'connectivity_rate': float(metrics.connectivity_rate),   
                            'count_rate': float(metrics.count_rate),                
                            'placement_rate': float(metrics.placement_rate),        
                            'sensor_positions': [list(map(float, pos)) for pos in sensors.tolist()],  
                            'archive_size': int(len(result.X)),
                            'solution_name': str(name),
                            'chromosome': [float(x) for x in chromosome.tolist()]  
                        }
                        
                        results.append(run_result)
                        
                        print(f"✅ Run {run_id + 1} completed successfully")
                        print(f"   Sensors: {metrics.n_sensors}, Coverage: {metrics.coverage_rate:.1f}%")
                        print(f"   Connectivity: {metrics.connectivity_rate:.1f}%, Time: {run_result['execution_time']:.1f}s")
                        
                    else:
                        failed_runs += 1
                        print(f"❌ Run {run_id + 1} failed: No valid solutions found")
                        
                else:
                    failed_runs += 1
                    print(f"❌ Run {run_id + 1} failed: No solutions in result")
                    
            except Exception as e:
                failed_runs += 1
                print(f"❌ Run {run_id + 1} failed with error: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
        
        print(f"\n=== TEST COMPLETION SUMMARY ===")
        print(f"Successful runs: {len(results)}/{self.test_runs}")
        print(f"Failed runs: {failed_runs}")
        
        # Analyze results
        analysis = self._analyze_spea2_results(results)
        
        # Save results
        self._save_results("SPEA2", results, analysis)
        
        return {
            'algorithm': 'SPEA2',
            'total_runs': self.test_runs,
            'successful_runs': len(results),
            'failed_runs': failed_runs,
            'results': results,
            'analysis': analysis,
            'timestamp': self.test_timestamp
        }
    
    def _analyze_spea2_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze SPEA2 test results for consistency and performance."""
        if not results:
            return {'error': 'No successful runs to analyze'}
        
        # Extract metrics for analysis
        metrics = {
            'n_sensors': [r['n_sensors'] for r in results],
            'coverage_rate': [r['coverage_rate'] for r in results],
            'over_coverage_rate': [r['over_coverage_rate'] for r in results],
            'connectivity_rate': [r['connectivity_rate'] for r in results],
            'execution_time': [r['execution_time'] for r in results],
            'archive_size': [r['archive_size'] for r in results]
        }
        
        # Calculate statistics
        analysis = {}
        for metric_name, values in metrics.items():
            analysis[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0  # Coefficient of variation
            }
        
        # Consistency analysis
        analysis['consistency'] = {
            'coverage_cv': analysis['coverage_rate']['cv'],
            'sensor_count_cv': analysis['n_sensors']['cv'],
            'connectivity_cv': analysis['connectivity_rate']['cv'],
            'is_consistent': (
                analysis['coverage_rate']['cv'] < 0.15 and  # Less than 15% variation
                analysis['n_sensors']['cv'] < 0.20 and      # Less than 20% variation
                analysis['connectivity_rate']['cv'] < 0.15   # Less than 15% variation
            )
        }
        
        # Performance benchmarks
        analysis['performance'] = {
            'avg_coverage': analysis['coverage_rate']['mean'],
            'avg_efficiency': analysis['coverage_rate']['mean'] / analysis['n_sensors']['mean'] if analysis['n_sensors']['mean'] > 0 else 0.0,
            'avg_execution_time': analysis['execution_time']['mean'],
            'reliability': float(len(results) / self.test_runs)  # Success rate
        }
        
        # Sensor placement consistency analysis
        analysis['sensor_placement'] = self._analyze_sensor_placement_consistency(results)
        
        # Print analysis summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _analyze_sensor_placement_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sensor placement consistency across all runs.
        
        Returns:
            Dict containing sensor placement analysis including most common positions
        """
        print(f"\n🎯 ANALYZING SENSOR PLACEMENT CONSISTENCY...")
        
        # Collect all sensor positions from all runs
        all_positions = []
        run_positions = []
        
        for result in results:
            positions = np.array(result['sensor_positions'])
            all_positions.extend(positions)
            run_positions.append(positions)
        
        all_positions = np.array(all_positions)
        
        if len(all_positions) == 0:
            return {'error': 'No sensor positions to analyze'}
        
        # Parameters for clustering (you can adjust these)
        position_tolerance = 2.0  # Distance tolerance for considering positions "similar"
        min_usage_threshold = 0.2  # Minimum fraction of runs a position must appear in to be considered "common"
        
        # Use DBSCAN to cluster similar positions
        clustering = DBSCAN(eps=position_tolerance, min_samples=max(1, int(len(results) * min_usage_threshold)))
        cluster_labels = clustering.fit_predict(all_positions)
        
        # Analyze clusters
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise points
        
        cluster_analysis = []
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_positions = all_positions[cluster_mask]
            
            # Calculate cluster center (most representative position)
            cluster_center = np.mean(cluster_positions, axis=0)
            
            # Count how many runs this cluster appears in
            runs_with_position = 0
            position_variants = []
            
            for run_idx, positions in enumerate(run_positions):
                # Check if any position in this run is close to the cluster center
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
        
        # Sort by usage frequency (most common first)
        cluster_analysis.sort(key=lambda x: x['usage_frequency'], reverse=True)
        
        # Identify most common sensor positions
        common_positions = [cluster for cluster in cluster_analysis if cluster['usage_frequency'] >= min_usage_threshold]
        
        # Create final sensor deployment recommendation
        final_deployment = self._create_final_deployment_recommendation(common_positions, results)
        
        placement_analysis = {
            'total_unique_clusters': len(cluster_analysis),
            'common_positions': common_positions,
            'all_clusters': cluster_analysis,
            'final_deployment': final_deployment,
            'analysis_parameters': {
                'position_tolerance': position_tolerance,
                'min_usage_threshold': min_usage_threshold,
                'total_runs_analyzed': len(results)
            }
        }
        
        # Print placement analysis summary
        self._print_placement_analysis_summary(placement_analysis)
        
        return placement_analysis
    
    def _create_final_deployment_recommendation(self, common_positions: List[Dict], results: List[Dict]) -> Dict[str, Any]:
        """
        Create a final sensor deployment recommendation based on the most consistent positions.
        """
        if not common_positions:
            return {'error': 'No common positions found for recommendation'}
        
        # Find the run with performance closest to average
        avg_coverage = np.mean([r['coverage_rate'] for r in results])
        avg_sensors = np.mean([r['n_sensors'] for r in results])
        
        best_representative_run = None
        best_score = float('inf')
        
        for result in results:
            # Score based on how close to average performance
            coverage_diff = abs(result['coverage_rate'] - avg_coverage)
            sensor_diff = abs(result['n_sensors'] - avg_sensors)
            combined_score = coverage_diff + sensor_diff * 10  # Weight sensor count more
            
            if combined_score < best_score:
                best_score = combined_score
                best_representative_run = result
        
        # Primary recommendation: most common positions
        primary_positions = [pos['representative_position'] for pos in common_positions]
        
        # Alternative recommendation: from the most representative run
        alternative_positions = best_representative_run['sensor_positions'] if best_representative_run else []
        
        # Create confidence scores for each position
        position_confidence = []
        for pos in common_positions:
            confidence = {
                'position': pos['representative_position'],
                'confidence_score': pos['usage_frequency'],
                'usage_description': f"Used in {pos['usage_count']}/{len(results)} runs ({pos['usage_frequency']:.1%})",
                'variants': pos['position_variants'][:5]  # Limit to top 5 variants
            }
            position_confidence.append(confidence)
        
        return {
            'primary_deployment': {
                'positions': primary_positions,
                'total_sensors': len(primary_positions),
                'description': 'Most frequently used sensor positions across all runs',
                'confidence_scores': position_confidence
            },
            'alternative_deployment': {
                'positions': alternative_positions,
                'total_sensors': len(alternative_positions),
                'description': f'Representative deployment from Run {best_representative_run["run_id"]} (closest to average performance)',
                'performance_metrics': {
                    'coverage_rate': best_representative_run['coverage_rate'],
                    'connectivity_rate': best_representative_run['connectivity_rate'],
                    'n_sensors': best_representative_run['n_sensors']
                } if best_representative_run else None
            },
            'deployment_statistics': {
                'avg_sensors_per_run': float(np.mean([len(r['sensor_positions']) for r in results])),
                'sensor_count_range': [
                    int(np.min([len(r['sensor_positions']) for r in results])),
                    int(np.max([len(r['sensor_positions']) for r in results]))
                ],
                'position_consistency_score': float(np.mean([pos['usage_frequency'] for pos in common_positions])) if common_positions else 0.0
            }
        }
    
    def _print_placement_analysis_summary(self, placement_analysis: Dict) -> None:
        """Print sensor placement analysis summary."""
        print(f"\n📍 SENSOR PLACEMENT CONSISTENCY ANALYSIS:")
        
        common_positions = placement_analysis.get('common_positions', [])
        final_deployment = placement_analysis.get('final_deployment', {})
        
        print(f"Total Position Clusters Found: {placement_analysis.get('total_unique_clusters', 0)}")
        print(f"Common Positions (≥{placement_analysis['analysis_parameters']['min_usage_threshold']:.0%} usage): {len(common_positions)}")
        
        if common_positions:
            print(f"\n🎯 MOST FREQUENTLY USED POSITIONS:")
            for i, pos in enumerate(common_positions[:10]):  # Show top 10
                print(f"  {i+1}. Position ({pos['representative_position'][0]:.1f}, {pos['representative_position'][1]:.1f})")
                print(f"     Usage: {pos['usage_count']}/{placement_analysis['analysis_parameters']['total_runs_analyzed']} runs ({pos['usage_frequency']:.1%})")
                print(f"     Consistency: {pos['consistency_score']:.2f}")
        
        if 'primary_deployment' in final_deployment:
            primary = final_deployment['primary_deployment']
            print(f"\n✅ RECOMMENDED PRIMARY DEPLOYMENT:")
            print(f"   Total Sensors: {primary['total_sensors']}")
            print(f"   Based on: {primary['description']}")
            
            if 'alternative_deployment' in final_deployment:
                alt = final_deployment['alternative_deployment']
                print(f"\n🔄 ALTERNATIVE DEPLOYMENT:")
                print(f"   Total Sensors: {alt['total_sensors']}")
                print(f"   Based on: {alt['description']}")
                if alt.get('performance_metrics'):
                    metrics = alt['performance_metrics']
                    print(f"   Performance: {metrics['coverage_rate']:.1f}% coverage, {metrics['connectivity_rate']:.1f}% connectivity")
        
        if 'deployment_statistics' in final_deployment:
            stats = final_deployment['deployment_statistics']
            print(f"\n📊 DEPLOYMENT STATISTICS:")
            print(f"   Average Sensors/Run: {stats['avg_sensors_per_run']:.1f}")
            print(f"   Sensor Count Range: {stats['sensor_count_range'][0]} - {stats['sensor_count_range'][1]}")
            print(f"   Position Consistency: {stats['position_consistency_score']:.2f}")
    
    def _print_analysis_summary(self, analysis: Dict) -> None:
        """Print a human-readable analysis summary."""
        print(f"\n=== SPEA2 ALGORITHM ANALYSIS ===")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"Average Coverage:     {analysis['performance']['avg_coverage']:.1f}%")
        print(f"Average Efficiency:   {analysis['performance']['avg_efficiency']:.2f} coverage/sensor")
        print(f"Average Sensors:      {analysis['n_sensors']['mean']:.1f} ± {analysis['n_sensors']['std']:.1f}")
        print(f"Average Connectivity: {analysis['connectivity_rate']['mean']:.1f}% ± {analysis['connectivity_rate']['std']:.1f}%")
        print(f"Average Exec Time:    {analysis['execution_time']['mean']:.1f}s ± {analysis['execution_time']['std']:.1f}s")
        print(f"Algorithm Reliability: {analysis['performance']['reliability']:.1%}")
        
        print(f"\n🎯 CONSISTENCY ANALYSIS:")
        print(f"Coverage Variation:    {analysis['consistency']['coverage_cv']:.3f} ({'✅ Consistent' if analysis['consistency']['coverage_cv'] < 0.15 else '⚠️ Variable'})")
        print(f"Sensor Count Variation: {analysis['consistency']['sensor_count_cv']:.3f} ({'✅ Consistent' if analysis['consistency']['sensor_count_cv'] < 0.20 else '⚠️ Variable'})")
        print(f"Connectivity Variation: {analysis['consistency']['connectivity_cv']:.3f} ({'✅ Consistent' if analysis['consistency']['connectivity_cv'] < 0.15 else '⚠️ Variable'})")
        print(f"Overall Consistency:   {'✅ CONSISTENT' if analysis['consistency']['is_consistent'] else '⚠️ VARIABLE'}")
        
        print(f"\n📈 RANGE ANALYSIS:")
        print(f"Coverage Range:       {analysis['coverage_rate']['min']:.1f}% - {analysis['coverage_rate']['max']:.1f}%")
        print(f"Sensor Count Range:   {analysis['n_sensors']['min']:.0f} - {analysis['n_sensors']['max']:.0f} sensors")
        print(f"Connectivity Range:   {analysis['connectivity_rate']['min']:.1f}% - {analysis['connectivity_rate']['max']:.1f}%")
    
    def visualize_results(self, test_results: Dict) -> None:
        """Create visualizations of the test results including sensor placement analysis."""
        if not test_results['results']:
            print("No results to visualize")
            return
        
        results = test_results['results']
        
        # Create figure with subplots (increased to accommodate sensor placement visualization)
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'SPEA2 Algorithm Performance Analysis ({len(results)} runs)', fontsize=16, fontweight='bold')
        
        # 1. Coverage Rate Distribution
        coverage_rates = [r['coverage_rate'] for r in results]
        axes[0, 0].hist(coverage_rates, bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(np.mean(coverage_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(coverage_rates):.1f}%')
        axes[0, 0].set_title('Coverage Rate Distribution')
        axes[0, 0].set_xlabel('Coverage Rate (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sensor Count Distribution
        sensor_counts = [r['n_sensors'] for r in results]
        axes[0, 1].hist(sensor_counts, bins=range(min(sensor_counts), max(sensor_counts) + 2), alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(np.mean(sensor_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sensor_counts):.1f}')
        axes[0, 1].set_title('Sensor Count Distribution')
        axes[0, 1].set_xlabel('Number of Sensors')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Connectivity Rate Distribution
        connectivity_rates = [r['connectivity_rate'] for r in results]
        axes[0, 2].hist(connectivity_rates, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].axvline(np.mean(connectivity_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(connectivity_rates):.1f}%')
        axes[0, 2].set_title('Connectivity Rate Distribution')
        axes[0, 2].set_xlabel('Connectivity Rate (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Execution Time Distribution
        exec_times = [r['execution_time'] for r in results]
        axes[1, 0].hist(exec_times, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(exec_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(exec_times):.1f}s')
        axes[1, 0].set_title('Execution Time Distribution')
        axes[1, 0].set_xlabel('Execution Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Coverage vs Sensor Count Scatter
        axes[1, 1].scatter(sensor_counts, coverage_rates, alpha=0.6, color='red', s=50)
        axes[1, 1].set_title('Coverage vs Sensor Count')
        axes[1, 1].set_xlabel('Number of Sensors')
        axes[1, 1].set_ylabel('Coverage Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(set(sensor_counts)) > 1:  # Only add trend line if there's variation
            z = np.polyfit(sensor_counts, coverage_rates, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(sensor_counts, p(sensor_counts), "r--", alpha=0.8, linewidth=2)
        
        # 6. Run Performance Over Time
        run_ids = [r['run_id'] for r in results]
        axes[1, 2].plot(run_ids, coverage_rates, 'o-', alpha=0.7, color='green', label='Coverage Rate')
        axes[1, 2].set_title('Performance Consistency Over Runs')
        axes[1, 2].set_xlabel('Run Number')
        axes[1, 2].set_ylabel('Coverage Rate (%)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        # 7. NEW: Sensor Placement Visualization
        self._plot_sensor_placements(axes[2, 0], results, test_results.get('analysis', {}))
        
        # 8. NEW: Position Usage Frequency
        self._plot_position_usage_frequency(axes[2, 1], test_results.get('analysis', {}))
        
        # 9. NEW: Deployment Consistency Map
        self._plot_deployment_consistency(axes[2, 2], test_results.get('analysis', {}))
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('test_results', exist_ok=True)
        plot_filename = f'test_results/spea2_analysis_{self.test_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {plot_filename}")
        plt.show()

    def _plot_sensor_placements(self, ax, results: List[Dict], analysis: Dict) -> None:
        """Plot all sensor placements from all runs with common positions highlighted."""
        # Plot field boundaries (assuming rectangular field)
        field_length = self.config.environment_field_length
        field_width = self.config.environment_field_width
        
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_aspect('equal')
        
        # Plot all sensor positions (light gray)
        for result in results:
            positions = np.array(result['sensor_positions'])
            if len(positions) > 0:
                ax.scatter(positions[:, 0], positions[:, 1], alpha=0.3, s=20, color='lightgray')
        
        # Plot common positions (highlighted)
        sensor_placement = analysis.get('sensor_placement', {})
        common_positions = sensor_placement.get('common_positions', [])
        
        if common_positions:
            for pos_info in common_positions:
                pos = pos_info['representative_position']
                frequency = pos_info['usage_frequency']
                # Size based on usage frequency
                size = 50 + frequency * 200
                ax.scatter(pos[0], pos[1], s=size, alpha=0.8, color='red', edgecolor='black', linewidth=2)
        
        ax.set_title('Sensor Placement Map\n(Red: Common positions, Gray: All positions)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
    
    def _plot_position_usage_frequency(self, ax, analysis: Dict) -> None:
        """Plot usage frequency of common sensor positions."""
        sensor_placement = analysis.get('sensor_placement', {})
        common_positions = sensor_placement.get('common_positions', [])
        
        if not common_positions:
            ax.text(0.5, 0.5, 'No common positions found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Usage Frequency')
            return
        
        # Get top 10 most common positions
        top_positions = common_positions[:10]
        frequencies = [pos['usage_frequency'] * 100 for pos in top_positions]
        labels = [f"Pos {i+1}\n({pos['representative_position'][0]:.1f}, {pos['representative_position'][1]:.1f})" 
                 for i, pos in enumerate(top_positions)]
        
        bars = ax.bar(range(len(frequencies)), frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Position Rank')
        ax.set_ylabel('Usage Frequency (%)')
        ax.set_title('Top Common Position Usage')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add frequency labels on bars
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{freq:.0f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_deployment_consistency(self, ax, analysis: Dict) -> None:
        """Plot deployment consistency metrics."""
        sensor_placement = analysis.get('sensor_placement', {})
        final_deployment = sensor_placement.get('final_deployment', {})
        
        if 'deployment_statistics' not in final_deployment:
            ax.text(0.5, 0.5, 'No deployment statistics available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Deployment Consistency')
            return
        
        stats = final_deployment['deployment_statistics']
        
        # Create consistency metrics visualization
        metrics = {
            'Position\nConsistency': stats.get('position_consistency_score', 0) * 100,
            'Coverage\nConsistency': (1 - analysis.get('consistency', {}).get('coverage_cv', 1)) * 100,
            'Sensor Count\nConsistency': (1 - analysis.get('consistency', {}).get('sensor_count_cv', 1)) * 100,
            'Connectivity\nConsistency': (1 - analysis.get('consistency', {}).get('connectivity_cv', 1)) * 100
        }
        
        # Ensure all values are between 0 and 100
        for key in metrics:
            metrics[key] = max(0, min(100, metrics[key]))
        
        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Consistency Score (%)')
        ax.set_title('Algorithm Consistency Metrics')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add threshold lines
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
        ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Fair (60%)')
        ax.legend(loc='upper right', fontsize=8)
    
    def save_final_deployment(self, test_results: Dict) -> None:
        """Save the final deployment recommendation to separate files for easy use."""
        analysis = test_results.get('analysis', {})
        sensor_placement = analysis.get('sensor_placement', {})
        final_deployment = sensor_placement.get('final_deployment', {})
        
        if not final_deployment:
            print("❌ No final deployment recommendation available")
            return
        
        os.makedirs('test_results/deployments', exist_ok=True)
        
        # Save primary deployment
        if 'primary_deployment' in final_deployment:
            primary = final_deployment['primary_deployment']
            primary_file = f'test_results/deployments/primary_deployment_{self.test_timestamp}.json'
            
            primary_data = {
                'deployment_type': 'primary',
                'algorithm': 'SPEA2',
                'timestamp': self.test_timestamp,
                'total_sensors': primary['total_sensors'],
                'sensor_positions': primary['positions'],
                'description': primary['description'],
                'confidence_scores': primary.get('confidence_scores', []),
                'field_dimensions': {
                    'length': self.config.environment_field_length,
                    'width': self.config.environment_field_width
                },
                'algorithm_parameters': {
                    'max_sensors': self.config.deployment_max_sensors,
                    'sensor_range': self.config.deployment_sensor_range,
                    'comm_range': self.config.deployment_comm_range
                }
            }
            
            with open(primary_file, 'w') as f:
                json.dump(primary_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"💾 Primary deployment saved: {primary_file}")
        
        # Save alternative deployment
        if 'alternative_deployment' in final_deployment:
            alt = final_deployment['alternative_deployment']
            alt_file = f'test_results/deployments/alternative_deployment_{self.test_timestamp}.json'
            
            alt_data = {
                'deployment_type': 'alternative',
                'algorithm': 'SPEA2',
                'timestamp': self.test_timestamp,
                'total_sensors': alt['total_sensors'],
                'sensor_positions': alt['positions'],
                'description': alt['description'],
                'performance_metrics': alt.get('performance_metrics', {}),
                'field_dimensions': {
                    'length': self.config.environment_field_length,
                    'width': self.config.environment_field_width
                },
                'algorithm_parameters': {
                    'max_sensors': self.config.deployment_max_sensors,
                    'sensor_range': self.config.deployment_sensor_range,
                    'comm_range': self.config.deployment_comm_range
                }
            }
            
            with open(alt_file, 'w') as f:
                json.dump(alt_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"💾 Alternative deployment saved: {alt_file}")
        
        # Save comparison summary
        comparison_file = f'test_results/deployments/deployment_comparison_{self.test_timestamp}.json'
        comparison_data = {
            'comparison_summary': final_deployment,
            'test_metadata': {
                'algorithm': 'SPEA2',
                'total_runs': test_results['total_runs'],
                'successful_runs': test_results['successful_runs'],
                'timestamp': self.test_timestamp
            }
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"💾 Deployment comparison saved: {comparison_file}")
        
        # Create a simple CSV for easy import into other tools
        if 'primary_deployment' in final_deployment:
            csv_file = f'test_results/deployments/primary_sensors_{self.test_timestamp}.csv'
            positions = final_deployment['primary_deployment']['positions']
            
            df_positions = pd.DataFrame(positions, columns=['X', 'Y'])
            df_positions['Sensor_ID'] = range(1, len(positions) + 1)
            df_positions = df_positions[['Sensor_ID', 'X', 'Y']]  # Reorder columns
            
            df_positions.to_csv(csv_file, index=False)
            print(f"💾 Primary sensors CSV saved: {csv_file}")
    
    def compare_with_previous_runs(self, current_results: Dict, previous_results_dir: str = 'test_results') -> Dict[str, Any]:
        """
        Compare current test results with previous runs for algorithm improvement analysis.
        
        Args:
            current_results: Current test results
            previous_results_dir: Directory containing previous test results
            
        Returns:
            Comparison analysis dictionary
        """
        if not os.path.exists(previous_results_dir):
            return {'error': 'No previous results directory found'}
        
        # Find previous SPEA2 test files
        previous_files = []
        for filename in os.listdir(previous_results_dir):
            if filename.startswith('spea2_test_') and filename.endswith('.json'):
                previous_files.append(os.path.join(previous_results_dir, filename))
        
        if not previous_files:
            return {'error': 'No previous SPEA2 test files found'}
        
        # Load and compare with the most recent previous run
        previous_files.sort()  # Sort by filename (timestamp)
        latest_previous = previous_files[-1]
        
        try:
            with open(latest_previous, 'r') as f:
                previous_data = json.load(f)
            
            print(f"\n🔄 COMPARING WITH PREVIOUS RUN:")
            print(f"Current run:  {current_results['timestamp']}")
            print(f"Previous run: {previous_data['timestamp']}")
            
            # Compare key metrics
            current_analysis = current_results['analysis']
            previous_analysis = previous_data['analysis']
            
            comparison = {
                'current_timestamp': current_results['timestamp'],
                'previous_timestamp': previous_data['timestamp'],
                'performance_comparison': {},
                'consistency_comparison': {},
                'deployment_comparison': {}
            }
            
            # Performance comparison
            perf_metrics = ['avg_coverage', 'avg_efficiency', 'avg_execution_time', 'reliability']
            for metric in perf_metrics:
                current_val = current_analysis['performance'].get(metric, 0)
                previous_val = previous_analysis['performance'].get(metric, 0)
                improvement = ((current_val - previous_val) / previous_val * 100) if previous_val > 0 else 0
                
                comparison['performance_comparison'][metric] = {
                    'current': current_val,
                    'previous': previous_val,
                    'improvement_percent': improvement,
                    'is_better': improvement > 0 if metric != 'avg_execution_time' else improvement < 0
                }
            
            # Consistency comparison
            consistency_metrics = ['coverage_cv', 'sensor_count_cv', 'connectivity_cv']
            for metric in consistency_metrics:
                current_val = current_analysis['consistency'].get(metric, 1)
                previous_val = previous_analysis['consistency'].get(metric, 1)
                improvement = ((previous_val - current_val) / previous_val * 100) if previous_val > 0 else 0
                
                comparison['consistency_comparison'][metric] = {
                    'current': current_val,
                    'previous': previous_val,
                    'improvement_percent': improvement,
                    'is_better': improvement > 0  # Lower CV is better
                }
            
            # Deployment comparison
            current_deployment = current_analysis.get('sensor_placement', {}).get('final_deployment', {})
            previous_deployment = previous_analysis.get('sensor_placement', {}).get('final_deployment', {})
            
            if current_deployment and previous_deployment:
                current_stats = current_deployment.get('deployment_statistics', {})
                previous_stats = previous_deployment.get('deployment_statistics', {})
                
                comparison['deployment_comparison'] = {
                    'position_consistency': {
                        'current': current_stats.get('position_consistency_score', 0),
                        'previous': previous_stats.get('position_consistency_score', 0)
                    },
                    'avg_sensors': {
                        'current': current_stats.get('avg_sensors_per_run', 0),
                        'previous': previous_stats.get('avg_sensors_per_run', 0)
                    }
                }
            
            self._print_comparison_summary(comparison)
            
            return comparison
            
        except Exception as e:
            return {'error': f'Failed to load previous results: {e}'}
    
    def _print_comparison_summary(self, comparison: Dict) -> None:
        """Print comparison summary with previous runs."""
        print(f"\n📊 PERFORMANCE COMPARISON:")
        
        perf_comp = comparison['performance_comparison']
        for metric, data in perf_comp.items():
            symbol = "📈" if data['is_better'] else "📉"
            print(f"{symbol} {metric.replace('_', ' ').title()}: {data['current']:.3f} vs {data['previous']:.3f} "
                  f"({data['improvement_percent']:+.1f}%)")
        
        print(f"\n🎯 CONSISTENCY COMPARISON:")
        cons_comp = comparison['consistency_comparison']
        for metric, data in cons_comp.items():
            symbol = "✅" if data['is_better'] else "⚠️"
            print(f"{symbol} {metric.replace('_', ' ').title()}: {data['current']:.3f} vs {data['previous']:.3f} "
                  f"({data['improvement_percent']:+.1f}% improvement)")
        
        if comparison['deployment_comparison']:
            print(f"\n📍 DEPLOYMENT COMPARISON:")
            deploy_comp = comparison['deployment_comparison']
            for metric, data in deploy_comp.items():
                print(f"   {metric.replace('_', ' ').title()}: {data['current']:.3f} vs {data['previous']:.3f}")
    
    def _save_results(self, algorithm_name: str, results: List[Dict], analysis: Dict) -> None:
        """Save test results to files."""
        os.makedirs('test_results', exist_ok=True)
        
        # Save detailed results as JSON
        output_data = {
            'algorithm': algorithm_name,
            'timestamp': self.test_timestamp,
            'config': {
                'test_runs': self.test_runs,
                'field_length': self.config.environment_field_length,
                'field_width': self.config.environment_field_width,
                'max_sensors': self.config.deployment_max_sensors,
                'sensor_range': self.config.deployment_sensor_range,
                'comm_range': self.config.deployment_comm_range
            },
            'results': results,
            'analysis': analysis
        }
        
        # Save as JSON with custom encoder to handle NumPy types
        json_filename = f'test_results/{algorithm_name.lower()}_test_{self.test_timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        # Save summary as CSV
        if results:
            # Create a flattened version for CSV (without nested structures)
            csv_data = []
            for r in results:
                csv_row = {k: v for k, v in r.items() 
                          if not isinstance(v, (list, dict))}  # Exclude complex nested data
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            csv_filename = f'test_results/{algorithm_name.lower()}_summary_{self.test_timestamp}.csv'
            df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed: {json_filename}")
        if results:
            print(f"   Summary:  {csv_filename}")


def main():
    """Main function to run algorithm tests."""
    try:
        config = OptimizationConfig()
        
        # Set number of test runs
        test_runs = config.test_num_runs
        
        # Create tester
        tester = AlgorithmTester(config, test_runs)
        
        # Run SPEA2 tests
        print("🚀 Starting SPEA2 algorithm testing...")
        spea2_results = tester.run_spea2_tests()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        tester.visualize_results(spea2_results)
        
        # Save final deployment recommendations
        print("\n💾 Saving deployment recommendations...")
        tester.save_final_deployment(spea2_results)
        
        # Compare with previous runs (if available)
        print("\n🔄 Checking for previous runs to compare...")
        comparison = tester.compare_with_previous_runs(spea2_results)
        if 'error' not in comparison:
            # Save comparison results
            comparison_file = f'test_results/comparison_{tester.test_timestamp}.json'
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2, cls=NumpyEncoder)
            print(f"💾 Comparison saved: {comparison_file}")
        else:
            print(f"ℹ️ {comparison['error']}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"🎉 ALGORITHM TESTING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"✅ Test runs completed: {spea2_results['successful_runs']}/{spea2_results['total_runs']}")
        
        # Print key results
        analysis = spea2_results['analysis']
        print(f"📊 Average coverage: {analysis['performance']['avg_coverage']:.1f}%")
        print(f"🔧 Average sensors: {analysis['n_sensors']['mean']:.1f}")
        print(f"🎯 Algorithm consistency: {'✅ CONSISTENT' if analysis['consistency']['is_consistent'] else '⚠️ VARIABLE'}")
        
        # Print deployment info
        sensor_placement = analysis.get('sensor_placement', {})
        final_deployment = sensor_placement.get('final_deployment', {})
        if 'primary_deployment' in final_deployment:
            primary = final_deployment['primary_deployment']
            print(f"📍 Recommended sensors: {primary['total_sensors']}")
            stats = final_deployment.get('deployment_statistics', {})
            print(f"🎪 Position consistency: {stats.get('position_consistency_score', 0):.2f}")
        
        print(f"\n📁 Check 'test_results' folder for:")
        print(f"   • Detailed analysis and visualizations")
        print(f"   • Deployment recommendations (JSON & CSV)")
        print(f"   • Performance comparison with previous runs")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure 'main2.py' and 'configurations/config_file.py' are available")
        print("Also ensure 'scikit-learn' is installed: pip install scikit-learn")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()