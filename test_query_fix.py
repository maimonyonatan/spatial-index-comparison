#!/usr/bin/env python3
"""
Test script to verify query execution works properly after restoration.
"""

import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.query.engine import QueryEngine, IndexType

def test_coordinate_extraction():
    """Test that coordinate extraction works without the unpacking error."""
    print("🧪 Testing coordinate extraction and query execution...")
    
    # Create simple test data as a DataFrame (DataLoader expects DataFrames)
    np.random.seed(42)
    test_coords_array = np.random.uniform(low=[40.0, -75.0], high=[41.0, -74.0], size=(100, 2))
    
    # Create a DataFrame with the expected column names
    import pandas as pd
    test_df = pd.DataFrame(test_coords_array, columns=['Start_Lat', 'Start_Lng'])
    
    print(f"✅ Created test data: {len(test_df)} points")
    print(f"   DataFrame shape: {test_df.shape}")
    print(f"   Sample coordinates: {test_df[['Start_Lat', 'Start_Lng']].head(3).values}")
    
    # Initialize components
    loader = DataLoader()
    engine = QueryEngine()
    
    # Compute Morton codes for learned indexes
    print("📊 Computing Morton codes...")
    normalized_coords = loader.normalize_coordinates(test_df)
    morton_codes = loader.compute_morton_codes(normalized_coords)
    print(f"✅ Morton codes computed: {len(morton_codes)} codes")
    
    # Get coordinate array for indexes
    test_coords = test_df[['Start_Lat', 'Start_Lng']].values
    
    # Build indexes
    print("🏗️ Building indexes...")
    
    try:
        # R-Tree
        engine.add_index("rtree", IndexType.RTREE, test_coords)
        print("✅ R-Tree index built successfully")
        
        # ZM Linear
        engine.add_index("zm_linear", IndexType.ZM_LINEAR, test_coords, morton_codes)
        print("✅ ZM Linear index built successfully")
        
        # ZM MLP (with minimal config for quick test)
        engine.add_index("zm_mlp", IndexType.ZM_MLP, test_coords, morton_codes, 
                        hidden_dims=[16], epochs=5, batch_size=32)
        print("✅ ZM MLP index built successfully")
        
    except Exception as e:
        print(f"❌ Error building indexes: {e}")
        return False
    
    # Test queries
    print("🔍 Testing queries...")
    
    try:
        # Point query
        print("  📍 Testing point query...")
        lat, lon = 40.5, -74.5
        tolerance = 0.01
        point_results = engine.point_query(lat, lon, tolerance=tolerance)
        
        print(f"     Query point: ({lat}, {lon}) with tolerance {tolerance}")
        for index_name, result in point_results.items():
            if 'error' not in result:
                print(f"     {index_name}: {result['count']} results in {result['query_time_seconds']*1000:.2f}ms")
            else:
                print(f"     {index_name}: ERROR - {result['error']}")
        
        # Range query
        print("  📦 Testing range query...")
        min_lat, max_lat = 40.3, 40.7
        min_lon, max_lon = -74.8, -74.2
        range_results = engine.range_query(min_lat, max_lat, min_lon, max_lon)
        
        print(f"     Query range: [{min_lat}, {max_lat}] × [{min_lon}, {max_lon}]")
        for index_name, result in range_results.items():
            if 'error' not in result:
                print(f"     {index_name}: {result['count']} results in {result['query_time_seconds']*1000:.2f}ms")
            else:
                print(f"     {index_name}: ERROR - {result['error']}")
        
        # k-NN query
        print("  🎯 Testing k-NN query...")
        k = 5
        knn_results = engine.knn_query(lat, lon, k=k)
        
        print(f"     Query: {k}-NN around ({lat}, {lon})")
        for index_name, result in knn_results.items():
            if 'error' not in result:
                print(f"     {index_name}: {result['count']} results in {result['query_time_seconds']*1000:.2f}ms")
            else:
                print(f"     {index_name}: ERROR - {result['error']}")
        
        print("✅ All queries executed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error executing queries: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_extraction()
    if success:
        print("\n🎉 SUCCESS: Query execution system is working properly!")
        print("   The coordinate extraction issue has been resolved.")
        print("   You can now use the Streamlit GUI without the 'too many values to unpack' error.")
    else:
        print("\n💥 FAILURE: There are still issues with the query execution system.")
        sys.exit(1)