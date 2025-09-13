#!/usr/bin/env python3
"""
Basic example demonstrating the ZM R-Tree Research system.

This example shows how to:
1. Load a spatial dataset
2. Build different types of indexes
3. Execute queries
4. Compare performance
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.query.engine import QueryEngine, IndexType
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator


def create_sample_dataset(num_points: int = 10000, output_file: str = "sample_data.csv"):
    """Create a sample spatial dataset for testing."""
    print(f"Creating sample dataset with {num_points} points...")
    
    # Generate random spatial data (simulating accident locations)
    np.random.seed(42)
    
    # Focus on a region (e.g., around Philadelphia)
    lat_center, lon_center = 39.9526, -75.1652
    lat_range, lon_range = 2.0, 2.0
    
    data = {
        'ID': range(num_points),
        'Start_Lat': np.random.normal(lat_center, lat_range/6, num_points),
        'Start_Lng': np.random.normal(lon_center, lon_range/6, num_points),
        'Severity': np.random.randint(1, 5, num_points),
        'City': np.random.choice(['Philadelphia', 'Camden', 'Trenton'], num_points),
        'State': np.random.choice(['PA', 'NJ', 'DE'], num_points)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure coordinates are within reasonable bounds
    df['Start_Lat'] = np.clip(df['Start_Lat'], lat_center - lat_range, lat_center + lat_range)
    df['Start_Lng'] = np.clip(df['Start_Lng'], lon_center - lon_range, lon_center + lon_range)
    
    # Save to CSV
    output_path = Path(__file__).parent / output_file
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset saved to: {output_path}")
    return output_path


def basic_example():
    """Run a basic example of the research system."""
    print("🗺️ ZM R-Tree Research System - Basic Example")
    print("=" * 60)
    
    # Step 1: Create or load dataset
    data_file = create_sample_dataset(5000)
    
    # Step 2: Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    loader = DataLoader()
    
    # Load the dataset
    df = loader.load_csv(data_file, sample_size=2000)  # Use subset for speed
    
    # Get coordinates and prepare Morton codes
    coordinates = df[['Start_Lat', 'Start_Lng']].values
    normalized_coords = loader.normalize_coordinates(df)
    morton_codes = loader.compute_morton_codes(normalized_coords)
    
    print(f"Dataset loaded: {len(coordinates)} points")
    print(f"Coordinate bounds: Lat [{coordinates[:, 0].min():.4f}, {coordinates[:, 0].max():.4f}], "
          f"Lon [{coordinates[:, 1].min():.4f}, {coordinates[:, 1].max():.4f}]")
    
    # Step 3: Build indexes
    print("\n🏗️ Building spatial indexes...")
    query_engine = QueryEngine()
    
    # Build R-Tree index
    print("  Building R-Tree...")
    query_engine.add_index("rtree", IndexType.RTREE, coordinates)
    
    # Build ZM Linear index
    print("  Building ZM Linear...")
    query_engine.add_index("zm_linear", IndexType.ZM_LINEAR, coordinates, morton_codes)
    
    # Build ZM MLP index (with fewer epochs for speed)
    print("  Building ZM MLP...")
    query_engine.add_index("zm_mlp", IndexType.ZM_MLP, coordinates, morton_codes, epochs=30)
    
    # Step 4: Display index statistics
    print("\n📊 Index Statistics:")
    stats = query_engine.get_index_statistics()
    for name, stat in stats.items():
        print(f"  {name}:")
        print(f"    Build time: {stat.get('build_time_seconds', 0):.4f} seconds")
        print(f"    Memory usage: {stat.get('memory_usage_mb', 0):.2f} MB")
    
    # Step 5: Execute sample queries
    print("\n🔍 Executing sample queries...")
    
    # Point query
    query_lat, query_lon = coordinates[100]  # Use a point from the dataset
    print(f"\n  Point Query at ({query_lat:.4f}, {query_lon:.4f}):")
    point_results = query_engine.point_query(query_lat, query_lon, tolerance=0.001)
    
    for name, result in point_results.items():
        if 'error' not in result:
            print(f"    {name}: {result['count']} results in {result['query_time_seconds']:.6f}s")
    
    # Range query
    lat_center = np.mean(coordinates[:, 0])
    lon_center = np.mean(coordinates[:, 1])
    lat_range = (coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.1
    lon_range = (coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.1
    
    print(f"\n  Range Query (10% of data extent):")
    range_results = query_engine.range_query(
        lat_center - lat_range/2, lat_center + lat_range/2,
        lon_center - lon_range/2, lon_center + lon_range/2
    )
    
    for name, result in range_results.items():
        if 'error' not in result:
            print(f"    {name}: {result['count']} results in {result['query_time_seconds']:.6f}s")
    
    # k-NN query
    print(f"\n  5-NN Query at ({query_lat:.4f}, {query_lon:.4f}):")
    knn_results = query_engine.knn_query(query_lat, query_lon, k=5)
    
    for name, result in knn_results.items():
        if 'error' not in result:
            print(f"    {name}: {result['count']} results in {result['query_time_seconds']:.6f}s")
    
    # Step 6: Performance comparison
    print("\n⚡ Running performance evaluation...")
    evaluator = PerformanceEvaluator(query_engine)
    
    # Run a quick evaluation
    try:
        eval_results = evaluator.comprehensive_evaluation()
        
        print("\n📊 Performance Summary:")
        summary = eval_results['summary']
        
        if summary['index_comparison']:
            print("  Query Time Ranking (fastest to slowest):")
            for i, entry in enumerate(summary['performance_rankings']['query_time'], 1):
                print(f"    {i}. {entry['index']}: {entry['avg_query_time']*1000:.2f}ms")
            
            print("\n  Memory Usage Ranking (lowest to highest):")
            for i, entry in enumerate(summary['performance_rankings']['memory_usage'], 1):
                print(f"    {i}. {entry['index']}: {entry['memory_usage_mb']:.2f}MB")
    
    except Exception as e:
        print(f"  Evaluation error: {e}")
    
    print("\n✅ Basic example completed!")
    print("\nNext steps:")
    print("  1. Try the CLI: python main.py --help")
    print("  2. Launch GUI: python main.py --gui")
    print("  3. Use your own dataset with the CLI commands")


if __name__ == "__main__":
    basic_example()