#!/usr/bin/env python3
"""
Debug script to identify coordinate extraction issues.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_coordinate_formats():
    """Debug different coordinate formats to understand the issue."""
    print("🔍 Debugging coordinate formats...")
    
    # Test 1: Simple numpy array (like original)
    print("\n1. Testing simple numpy array:")
    coords_simple = np.array([[40.1, -74.1], [40.2, -74.2], [40.3, -74.3]])
    print(f"   Shape: {coords_simple.shape}")
    print(f"   Sample: {coords_simple[0]}")
    
    try:
        for lat, lon in coords_simple:
            print(f"   Direct unpack: {lat}, {lon}")
            break
        print("   ✅ Direct unpacking works")
    except Exception as e:
        print(f"   ❌ Direct unpacking failed: {e}")
    
    # Test 2: DataFrame values (like our loader)
    print("\n2. Testing DataFrame.values:")
    df = pd.DataFrame(coords_simple, columns=['Start_Lat', 'Start_Lng'])
    coords_df_values = df[['Start_Lat', 'Start_Lng']].values
    print(f"   Shape: {coords_df_values.shape}")
    print(f"   Type: {type(coords_df_values)}")
    print(f"   Sample: {coords_df_values[0]}")
    print(f"   Sample type: {type(coords_df_values[0])}")
    
    try:
        for lat, lon in coords_df_values:
            print(f"   Direct unpack: {lat}, {lon}")
            break
        print("   ✅ Direct unpacking works")
    except Exception as e:
        print(f"   ❌ Direct unpacking failed: {e}")
    
    # Test 3: Individual coordinate access
    print("\n3. Testing individual coordinate access:")
    try:
        coord = coords_df_values[0]
        print(f"   coord = {coord}")
        print(f"   coord[0] = {coord[0]}")
        print(f"   coord[1] = {coord[1]}")
        lat, lon = coord[0], coord[1]
        print(f"   Manual extract: {lat}, {lon}")
        print("   ✅ Manual extraction works")
    except Exception as e:
        print(f"   ❌ Manual extraction failed: {e}")
    
    # Test 4: Try the problematic unpacking
    print("\n4. Testing problematic unpacking pattern:")
    try:
        for i, coord in enumerate(coords_df_values):
            if i < 3:  # Test first 3
                print(f"   coord[{i}]: {coord} (type: {type(coord)})")
                if hasattr(coord, '__len__'):
                    print(f"   Length: {len(coord)}")
                if hasattr(coord, 'shape'):
                    print(f"   Shape: {coord.shape}")
                
                # Try the exact pattern that's failing
                try:
                    lat, lon = coord[0], coord[1]
                    print(f"   ✅ Pattern works: {lat}, {lon}")
                except Exception as e:
                    print(f"   ❌ Pattern failed: {e}")
                    
                    # Try alternative extractions
                    try:
                        lat, lon = float(coord[0]), float(coord[1])
                        print(f"   ✅ Float cast works: {lat}, {lon}")
                    except Exception as e2:
                        print(f"   ❌ Float cast failed: {e2}")
        
        print("   ✅ Loop-based extraction works")
    except Exception as e:
        print(f"   ❌ Loop-based extraction failed: {e}")

def simulate_streamlit_data_flow():
    """Simulate the Streamlit data flow to find where the issue occurs."""
    print("\n🎭 Simulating Streamlit data flow...")
    
    # Simulate loading data like Streamlit does
    print("1. Creating DataFrame like CSV loader...")
    np.random.seed(42)
    test_data = np.random.uniform(low=[40.0, -75.0], high=[41.0, -74.0], size=(10, 2))
    df = pd.DataFrame(test_data, columns=['Start_Lat', 'Start_Lng'])
    print(f"   DataFrame shape: {df.shape}")
    
    # Simulate session state storage
    print("2. Simulating session state storage...")
    coordinates = df[['Start_Lat', 'Start_Lng']].values
    print(f"   Coordinates shape: {coordinates.shape}")
    print(f"   Coordinates type: {type(coordinates)}")
    
    # Simulate visualization access
    print("3. Simulating visualization access...")
    try:
        # This is how the visualization page accesses coords
        for i, coord in enumerate(coordinates):
            if i < 3:
                print(f"   coord[{i}]: {coord}")
                
                # Test direct unpacking (like original code)
                try:
                    lat, lon = coord
                    print(f"   ✅ Direct unpack: {lat}, {lon}")
                except Exception as e:
                    print(f"   ❌ Direct unpack failed: {e}")
                
                # Test indexed access (like our current approach)
                try:
                    lat, lon = coord[0], coord[1]
                    print(f"   ✅ Indexed access: {lat}, {lon}")
                except Exception as e:
                    print(f"   ❌ Indexed access failed: {e}")
    except Exception as e:
        print(f"   ❌ Coordinate iteration failed: {e}")

if __name__ == "__main__":
    debug_coordinate_formats()
    simulate_streamlit_data_flow()
    print("\n✅ Debug complete!")