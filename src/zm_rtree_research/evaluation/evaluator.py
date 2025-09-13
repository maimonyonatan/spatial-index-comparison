"""
Performance evaluation and metrics for spatial index comparison.
"""

import logging
import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from zm_rtree_research.query.engine import QueryEngine, QueryType, IndexType

logger = logging.getLogger(__name__)


@dataclass
class QueryBenchmark:
    """Configuration for a query benchmark."""
    query_type: QueryType
    num_queries: int
    selectivity: Optional[float] = None  # For range queries
    k: Optional[int] = None  # For kNN queries
    tolerance: Optional[float] = None  # For point queries
    description: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for an index."""
    index_name: str
    index_type: str
    build_time: float
    memory_usage_mb: float
    avg_query_time: float
    min_query_time: float
    max_query_time: float
    std_query_time: float
    throughput_queries_per_sec: float
    accuracy_metrics: Dict[str, float]
    error_rate: float = 0.0


class PerformanceEvaluator:
    """
    Comprehensive performance evaluator for spatial indexes.
    
    Provides systematic benchmarking, accuracy evaluation, and comparative
    analysis of R-Tree vs Learned ZM Index implementations.
    """
    
    def __init__(self, query_engine: QueryEngine):
        """
        Initialize performance evaluator.
        
        Args:
            query_engine: QueryEngine instance with loaded indexes
        """
        self.query_engine = query_engine
        self.evaluation_results: Dict[str, Any] = {}
        
    def generate_random_queries(
        self,
        query_type: QueryType,
        num_queries: int,
        bounds: Optional[Dict[str, float]] = None,
        selectivity: float = 0.001,
        k: int = 1,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generate random queries for benchmarking.
        
        Args:
            query_type: Type of queries to generate
            num_queries: Number of queries to generate
            bounds: Coordinate bounds (if None, inferred from data)
            selectivity: Selectivity for range queries (fraction of space)
            k: Number of neighbors for kNN queries
            random_state: Random seed
            
        Returns:
            List of query parameter dictionaries
        """
        np.random.seed(random_state)
        
        # Infer bounds from coordinates if not provided
        if bounds is None and self.query_engine.coordinates is not None:
            coords = self.query_engine.coordinates
            bounds = {
                'min_lat': float(np.min(coords[:, 0])),
                'max_lat': float(np.max(coords[:, 0])),
                'min_lon': float(np.min(coords[:, 1])),
                'max_lon': float(np.max(coords[:, 1]))
            }
        
        if bounds is None:
            raise ValueError("No bounds available and no coordinates in query engine")
        
        queries = []
        
        if query_type == QueryType.POINT:
            for _ in range(num_queries):
                lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
                lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
                queries.append({
                    'lat': lat,
                    'lon': lon,
                    'tolerance': 1e-6
                })
        
        elif query_type == QueryType.RANGE:
            lat_range = bounds['max_lat'] - bounds['min_lat']
            lon_range = bounds['max_lon'] - bounds['min_lon']
            
            # Calculate box dimensions for given selectivity
            box_area = selectivity * lat_range * lon_range
            box_side = np.sqrt(box_area)
            
            for _ in range(num_queries):
                # Random center point
                center_lat = np.random.uniform(
                    bounds['min_lat'] + box_side/2,
                    bounds['max_lat'] - box_side/2
                )
                center_lon = np.random.uniform(
                    bounds['min_lon'] + box_side/2,
                    bounds['max_lon'] - box_side/2
                )
                
                queries.append({
                    'min_lat': center_lat - box_side/2,
                    'max_lat': center_lat + box_side/2,
                    'min_lon': center_lon - box_side/2,
                    'max_lon': center_lon + box_side/2
                })
        
        elif query_type == QueryType.KNN:
            for _ in range(num_queries):
                lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
                lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
                queries.append({
                    'lat': lat,
                    'lon': lon,
                    'k': k
                })
        
        return queries
    
    def run_benchmark(
        self,
        benchmark: QueryBenchmark,
        index_names: Optional[List[str]] = None,
        bounds: Optional[Dict[str, float]] = None
    ) -> Dict[str, PerformanceMetrics]:
        """
        Run a performance benchmark.
        
        Args:
            benchmark: Benchmark configuration
            index_names: List of index names to benchmark (None for all)
            bounds: Coordinate bounds for query generation
            
        Returns:
            Dictionary mapping index names to performance metrics
        """
        logger.info(f"Running benchmark: {benchmark.description}")
        
        # Generate queries
        query_params = {}
        if benchmark.query_type == QueryType.RANGE and benchmark.selectivity:
            query_params['selectivity'] = benchmark.selectivity
        if benchmark.query_type == QueryType.KNN and benchmark.k:
            query_params['k'] = benchmark.k
        if benchmark.query_type == QueryType.POINT and benchmark.tolerance:
            query_params['tolerance'] = benchmark.tolerance
        
        queries = self.generate_random_queries(
            benchmark.query_type,
            benchmark.num_queries,
            bounds=bounds,
            **query_params
        )
        
        # Execute benchmark
        results = self.query_engine.batch_queries(
            benchmark.query_type,
            queries,
            index_name=None
        )
        
        # Calculate performance metrics
        metrics = {}
        target_indexes = index_names if index_names else list(results.keys())
        
        for index_name in target_indexes:
            if index_name not in results or 'error' in results[index_name]:
                logger.warning(f"Skipping {index_name} due to errors")
                continue
            
            result = results[index_name]
            index_stats = self.query_engine.get_index_statistics(index_name)[index_name]
            
            # Calculate timing statistics
            query_times = []
            for batch_result in result['results']:
                # For batch results, we need to estimate individual query times
                if result['total_time_seconds']:
                    estimated_time = result['total_time_seconds'] / len(result['results'])
                    query_times.append(estimated_time)
            
            if not query_times:
                query_times = [result.get('avg_time_per_query', 0.0)]
            
            # Calculate accuracy metrics (placeholder - would need ground truth)
            accuracy_metrics = self._calculate_accuracy_metrics(result, benchmark)
            
            metrics[index_name] = PerformanceMetrics(
                index_name=index_name,
                index_type=result['index_type'],
                build_time=index_stats.get('build_time_seconds', 0.0),
                memory_usage_mb=index_stats.get('memory_usage_mb', 0.0),
                avg_query_time=np.mean(query_times),
                min_query_time=np.min(query_times),
                max_query_time=np.max(query_times),
                std_query_time=np.std(query_times),
                throughput_queries_per_sec=1.0 / np.mean(query_times) if np.mean(query_times) > 0 else 0.0,
                accuracy_metrics=accuracy_metrics,
                error_rate=0.0  # Would calculate from failed queries
            )
        
        return metrics
    
    def _calculate_accuracy_metrics(
        self,
        result: Dict[str, Any],
        benchmark: QueryBenchmark
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics for learned indexes.
        
        Args:
            result: Query execution result
            benchmark: Benchmark configuration
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Placeholder implementation - in practice, would compare with ground truth
        # from R-Tree or exact computation
        
        accuracy_metrics = {
            'precision': 1.0,  # Would calculate true precision
            'recall': 1.0,     # Would calculate true recall
            'f1_score': 1.0,   # Would calculate F1 score
        }
        
        if benchmark.query_type == QueryType.RANGE:
            # For range queries, could calculate error bounds
            accuracy_metrics.update({
                'avg_error_bound': 0.0,
                'max_error_bound': 0.0
            })
        elif benchmark.query_type == QueryType.KNN:
            # For kNN queries, could calculate distance accuracy
            accuracy_metrics.update({
                'distance_accuracy': 1.0,
                'ranking_accuracy': 1.0
            })
        
        return accuracy_metrics
    
    def comprehensive_evaluation(
        self,
        output_dir: Optional[Path] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across multiple benchmarks.
        
        Args:
            output_dir: Directory to save results (None for no saving)
            include_plots: Whether to generate performance plots
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        
        # Define benchmark suite
        benchmarks = [
            QueryBenchmark(
                QueryType.POINT,
                num_queries=1000,
                tolerance=1e-6,
                description="Point Query Benchmark (1000 queries)"
            ),
            QueryBenchmark(
                QueryType.RANGE,
                num_queries=500,
                selectivity=0.0001,
                description="Range Query Benchmark - Low Selectivity (0.01%)"
            ),
            QueryBenchmark(
                QueryType.RANGE,
                num_queries=500,
                selectivity=0.001,
                description="Range Query Benchmark - Medium Selectivity (0.1%)"
            ),
            QueryBenchmark(
                QueryType.RANGE,
                num_queries=500,
                selectivity=0.01,
                description="Range Query Benchmark - High Selectivity (1%)"
            ),
            QueryBenchmark(
                QueryType.KNN,
                num_queries=500,
                k=1,
                description="1-NN Query Benchmark"
            ),
            QueryBenchmark(
                QueryType.KNN,
                num_queries=500,
                k=10,
                description="10-NN Query Benchmark"
            ),
        ]
        
        # Run all benchmarks
        all_results = {}
        for benchmark in benchmarks:
            try:
                metrics = self.run_benchmark(benchmark)
                all_results[benchmark.description] = metrics
            except Exception as e:
                logger.error(f"Error running benchmark {benchmark.description}: {e}")
                all_results[benchmark.description] = {'error': str(e)}
        
        # Aggregate results
        evaluation_summary = self._create_evaluation_summary(all_results)
        
        # Save results if output directory specified
        if output_dir:
            self._save_evaluation_results(all_results, evaluation_summary, output_dir)
        
        # Store in instance for later access
        self.evaluation_results = {
            'benchmarks': all_results,
            'summary': evaluation_summary,
            'timestamp': time.time()
        }
        
        logger.info("Comprehensive evaluation completed")
        return self.evaluation_results
    
    def _create_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of evaluation results."""
        summary = {
            'total_benchmarks': len(results),
            'successful_benchmarks': len([r for r in results.values() if 'error' not in r]),
            'index_comparison': {},
            'performance_rankings': {}
        }
        
        # Collect metrics by index
        index_metrics = {}
        for benchmark_name, benchmark_results in results.items():
            if 'error' in benchmark_results:
                continue
            
            for index_name, metrics in benchmark_results.items():
                if index_name not in index_metrics:
                    index_metrics[index_name] = {
                        'build_time': metrics.build_time,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'query_times': [],
                        'throughputs': [],
                        'index_type': metrics.index_type
                    }
                
                index_metrics[index_name]['query_times'].append(metrics.avg_query_time)
                index_metrics[index_name]['throughputs'].append(metrics.throughput_queries_per_sec)
        
        # Calculate aggregate statistics
        for index_name, metrics in index_metrics.items():
            summary['index_comparison'][index_name] = {
                'index_type': metrics['index_type'],
                'build_time': metrics['build_time'],
                'memory_usage_mb': metrics['memory_usage_mb'],
                'avg_query_time': np.mean(metrics['query_times']),
                'avg_throughput': np.mean(metrics['throughputs']),
                'query_time_std': np.std(metrics['query_times'])
            }
        
        # Create performance rankings
        if summary['index_comparison']:
            # Rank by average query time (lower is better)
            query_time_ranking = sorted(
                summary['index_comparison'].items(),
                key=lambda x: x[1]['avg_query_time']
            )
            summary['performance_rankings']['query_time'] = [
                {'index': name, 'avg_query_time': data['avg_query_time']}
                for name, data in query_time_ranking
            ]
            
            # Rank by memory usage (lower is better)
            memory_ranking = sorted(
                summary['index_comparison'].items(),
                key=lambda x: x[1]['memory_usage_mb']
            )
            summary['performance_rankings']['memory_usage'] = [
                {'index': name, 'memory_usage_mb': data['memory_usage_mb']}
                for name, data in memory_ranking
            ]
        
        return summary
    
    def _save_evaluation_results(
        self,
        results: Dict[str, Any],
        summary: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert PerformanceMetrics to dict for JSON serialization
            json_results = {}
            for benchmark_name, benchmark_results in results.items():
                if 'error' in benchmark_results:
                    json_results[benchmark_name] = benchmark_results
                else:
                    json_results[benchmark_name] = {
                        index_name: {
                            'index_name': metrics.index_name,
                            'index_type': metrics.index_type,
                            'build_time': metrics.build_time,
                            'memory_usage_mb': metrics.memory_usage_mb,
                            'avg_query_time': metrics.avg_query_time,
                            'min_query_time': metrics.min_query_time,
                            'max_query_time': metrics.max_query_time,
                            'std_query_time': metrics.std_query_time,
                            'throughput_queries_per_sec': metrics.throughput_queries_per_sec,
                            'accuracy_metrics': metrics.accuracy_metrics,
                            'error_rate': metrics.error_rate
                        }
                        for index_name, metrics in benchmark_results.items()
                    }
            
            json.dump(json_results, f, indent=2)
        
        # Save summary as JSON
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary as CSV for easy analysis
        if summary['index_comparison']:
            df_data = []
            for index_name, data in summary['index_comparison'].items():
                df_data.append({
                    'index_name': index_name,
                    **data
                })
            
            df = pd.DataFrame(df_data)
            csv_file = output_dir / "performance_summary.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def get_comparison_report(self) -> str:
        """
        Generate a text-based comparison report.
        
        Returns:
            Formatted comparison report as string
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run comprehensive_evaluation() first."
        
        summary = self.evaluation_results['summary']
        report = []
        
        report.append("=" * 80)
        report.append("SPATIAL INDEX PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Total benchmarks run: {summary['total_benchmarks']}")
        report.append(f"Successful benchmarks: {summary['successful_benchmarks']}")
        report.append("")
        
        if summary['index_comparison']:
            report.append("INDEX PERFORMANCE SUMMARY")
            report.append("-" * 40)
            
            for index_name, data in summary['index_comparison'].items():
                report.append(f"\n{index_name} ({data['index_type']}):")
                report.append(f"  Build time: {data['build_time']:.4f} seconds")
                report.append(f"  Memory usage: {data['memory_usage_mb']:.2f} MB")
                report.append(f"  Avg query time: {data['avg_query_time']:.6f} seconds")
                report.append(f"  Avg throughput: {data['avg_throughput']:.2f} queries/sec")
        
        if summary['performance_rankings']:
            report.append("\nPERFORMANCE RANKINGS")
            report.append("-" * 40)
            
            report.append("\nQuery Time (lower is better):")
            for i, entry in enumerate(summary['performance_rankings']['query_time'], 1):
                report.append(f"  {i}. {entry['index']}: {entry['avg_query_time']:.6f} seconds")
            
            report.append("\nMemory Usage (lower is better):")
            for i, entry in enumerate(summary['performance_rankings']['memory_usage'], 1):
                report.append(f"  {i}. {entry['index']}: {entry['memory_usage_mb']:.2f} MB")
        
        return "\n".join(report)