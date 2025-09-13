"""
Interactive Streamlit GUI for spatial index comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.query.engine import QueryEngine, IndexType, QueryType
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator, QueryBenchmark

logger = logging.getLogger(__name__)


def safe_coordinate_extract(coord_array, index):
    """
    Safely extract latitude and longitude from coordinate array.
    Handles various data types and potential Streamlit serialization issues.
    """
    try:
        coord = coord_array[index]
        
        # Handle different possible formats
        if hasattr(coord, '__len__') and len(coord) >= 2:
            # Standard array/list with at least 2 elements
            return float(coord[0]), float(coord[1])
        elif hasattr(coord, 'shape') and len(coord.shape) > 0:
            # NumPy array
            return float(coord.flat[0]), float(coord.flat[1])
        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
            # List or tuple
            return float(coord[0]), float(coord[1])
        else:
            # Try direct indexing as fallback
            logger.warning(f"Unexpected coordinate format: {type(coord)} - {coord}")
            return float(coord[0]), float(coord[1])
            
    except Exception as e:
        logger.error(f"Coordinate extraction failed for index {index}: {e}")
        logger.error(f"Coordinate type: {type(coord_array[index]) if index < len(coord_array) else 'Index out of bounds'}")
        logger.error(f"Coordinate value: {coord_array[index] if index < len(coord_array) else 'N/A'}")
        logger.error(f"Array shape: {coord_array.shape if hasattr(coord_array, 'shape') else 'No shape'}")
        # Return default coordinates as fallback
        return 0.0, 0.0


class StreamlitApp:
    """
    Interactive Streamlit application for spatial index comparison.
    
    Provides a web-based interface for loading datasets, building indexes,
    executing queries, and comparing performance metrics.
    """
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.data_loader = DataLoader()
        
        # Store QueryEngine in session state to persist between page changes
        if 'query_engine' not in st.session_state:
            st.session_state.query_engine = QueryEngine()
        self.query_engine = st.session_state.query_engine
        
        self.evaluator = None
        
        # Initialize session state
        if 'dataset_loaded' not in st.session_state:
            st.session_state.dataset_loaded = False
        if 'indexes_built' not in st.session_state:
            st.session_state.indexes_built = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
    
    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="ZM R-Tree Research",
            page_icon="🗺️",
            layout="wide"
        )
        
        st.title("🗺️ Spatial Index Comparison: R-Tree vs Learned ZM Index")
        st.markdown("""
        Research prototype for empirical comparison of traditional R-Tree spatial indexing 
        with learned ZM (Z-order/Morton) indexes using machine learning approaches.
        """)
        
        # Create sidebar for navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Data Loading", "Index Building", "Query Execution", "Performance Evaluation", "Visualization"]
        )
        
        if page == "Data Loading":
            self._data_loading_page()
        elif page == "Index Building":
            self._index_building_page()
        elif page == "Query Execution":
            self._query_execution_page()
        elif page == "Performance Evaluation":
            self._performance_evaluation_page()
        elif page == "Visualization":
            self._visualization_page()
    
    def _data_loading_page(self) -> None:
        """Data loading and preprocessing page."""
        st.header("📊 Data Loading")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with spatial coordinates",
            type=['csv'],
            help="CSV file should contain latitude and longitude columns"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Configuration options
            col1, col2 = st.columns(2)
            
            with col1:
                lat_col = st.text_input("Latitude column name", value="Start_Lat")
                lon_col = st.text_input("Longitude column name", value="Start_Lng")
                sample_size = st.number_input(
                    "Sample size (0 for all data)", 
                    min_value=0, 
                    value=10000,
                    help="Limit dataset size for faster processing"
                )
            
            with col2:
                # Geographic filtering
                st.subheader("Geographic Filtering")
                filter_state = st.text_input("Filter by state (optional)")
                filter_city = st.text_input("Filter by city (optional)")
            
            if st.button("Load Dataset"):
                with st.spinner("Loading dataset..."):
                    try:
                        # Ensure sample_size is properly converted to int or None
                        actual_sample_size = None
                        if sample_size and int(sample_size) > 0:
                            actual_sample_size = int(sample_size)
                        
                        # Load data
                        df = self.data_loader.load_csv(
                            temp_path,
                            lat_col=lat_col,
                            lon_col=lon_col,
                            sample_size=actual_sample_size
                        )
                        
                        # Check if we have any data after loading
                        if len(df) == 0:
                            st.error("No valid data found after loading. Please check your CSV file and column names.")
                            return
                        
                        # Apply geographic filtering
                        if filter_state or filter_city:
                            geographic_filter = {}
                            if filter_state:
                                geographic_filter['state'] = filter_state
                            if filter_city:
                                geographic_filter['city'] = filter_city
                            
                            df = self.data_loader.geographic_subset(**geographic_filter)
                            
                            # Check if geographic filtering left us with data
                            if len(df) == 0:
                                st.error("No data remains after geographic filtering. Please adjust your filter criteria.")
                                return
                        
                        # Validate we have coordinate data
                        coords_preview = df[[lat_col, lon_col]].head()
                        st.info(f"Found {len(df)} records. Sample coordinates:")
                        st.dataframe(coords_preview)
                        
                        # Normalize coordinates and compute Morton codes
                        normalized_coords = self.data_loader.normalize_coordinates(df, lat_col, lon_col)
                        morton_codes = self.data_loader.compute_morton_codes(normalized_coords)
                        
                        # Store in session state
                        st.session_state.dataset = df
                        st.session_state.coordinates = df[[lat_col, lon_col]].values
                        st.session_state.normalized_coords = normalized_coords
                        st.session_state.morton_codes = morton_codes
                        st.session_state.dataset_loaded = True
                        
                        st.success(f"Dataset loaded successfully! {len(df)} records.")
                        
                    except ValueError as ve:
                        st.error(f"Data validation error: {ve}")
                        logger.error(f"Dataset validation error: {ve}")
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
                        logger.error(f"Dataset loading error: {e}", exc_info=True)
        
        # Display dataset information if loaded
        if st.session_state.dataset_loaded:
            st.subheader("📈 Dataset Information")
            
            df = st.session_state.dataset
            coords = st.session_state.coordinates
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Coordinate Range (Lat)", f"{coords[:, 0].min():.4f} - {coords[:, 0].max():.4f}")
            with col3:
                st.metric("Coordinate Range (Lon)", f"{coords[:, 1].min():.4f} - {coords[:, 1].max():.4f}")
            
            # Sample data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Coordinate distribution plots
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Latitude Distribution', 'Longitude Distribution'])
            
            fig.add_trace(
                go.Histogram(x=coords[:, 0], name="Latitude", nbinsx=50),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=coords[:, 1], name="Longitude", nbinsx=50),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
            
            # Add interactive map showing the loaded data points
            st.subheader("🗺️ Data Points on Map")
            
            # Map configuration controls
            col1, col2, col3 = st.columns(3)
            with col1:
                map_sample_size = st.slider(
                    "Points to show on map", 
                    min_value=100, 
                    max_value=min(5000, len(coords)), 
                    value=min(1000, len(coords)),
                    help="Limit number of points for better map performance"
                )
            with col2:
                marker_size = st.slider("Marker size", min_value=1, max_value=8, value=3)
            with col3:
                map_style = st.selectbox(
                    "Map style", 
                    ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Stamen Terrain"],
                    index=1
                )
            
            # Sample coordinates for map display
            if len(coords) > map_sample_size:
                sample_indices = np.random.choice(len(coords), map_sample_size, replace=False)
                map_coords = coords[sample_indices]
            else:
                map_coords = coords
            
            # Calculate map center and bounds
            center_lat = float(np.mean(map_coords[:, 0]))
            center_lon = float(np.mean(map_coords[:, 1]))
            
            # Create folium map
            if map_style == "OpenStreetMap":
                tile_layer = None
            elif map_style == "CartoDB positron":
                tile_layer = "CartoDB positron"
            elif map_style == "CartoDB dark_matter":
                tile_layer = "CartoDB dark_matter"
            elif map_style == "Stamen Terrain":
                tile_layer = "Stamen Terrain"
            else:
                tile_layer = None
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=10,
                tiles=tile_layer
            )
            
            # Add data points to map
            for i, coord in enumerate(map_coords):
                lat, lon = safe_coordinate_extract(map_coords, i)
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=marker_size,
                    popup=f"Point {i+1}: ({lat:.4f}, {lon:.4f})",
                    color='red',
                    fillColor='red',
                    fillOpacity=0.6
                ).add_to(m)
            
            # Add bounding box rectangle
            bounds = [
                [float(map_coords[:, 0].min()), float(map_coords[:, 1].min())],
                [float(map_coords[:, 0].max()), float(map_coords[:, 1].max())]
            ]
            
            folium.Rectangle(
                bounds=bounds,
                color='blue',
                fill=False,
                weight=2,
                popup=f"Data bounds: {len(coords)} total points"
            ).add_to(m)
            
            # Display the map
            map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
            
            # Show map statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Points on Map", len(map_coords))
            with col2:
                st.metric("Map Center", f"({center_lat:.4f}, {center_lon:.4f})")
            with col3:
                lat_range = map_coords[:, 0].max() - map_coords[:, 0].min()
                lon_range = map_coords[:, 1].max() - map_coords[:, 1].min()
                st.metric("Data Span", f"{lat_range:.4f}° × {lon_range:.4f}°")
            
            # If user clicked on map, show click coordinates
            if map_data['last_clicked'] is not None:
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lon = map_data['last_clicked']['lng']
                st.info(f"🎯 Clicked coordinates: ({clicked_lat:.6f}, {clicked_lon:.6f})")
                
                # Find nearest point to click
                distances = np.sqrt((coords[:, 0] - clicked_lat)**2 + (coords[:, 1] - clicked_lon)**2)
                nearest_idx = np.argmin(distances)
                nearest_lat, nearest_lon = safe_coordinate_extract(coords, nearest_idx)
                nearest_distance = distances[nearest_idx]
                
                st.write(f"📍 Nearest data point: ({nearest_lat:.6f}, {nearest_lon:.6f}) - Distance: {nearest_distance:.6f}°")
    
    def _index_building_page(self) -> None:
        """Index building and configuration page."""
        st.header("🏗️ Index Building")
        
        if not st.session_state.dataset_loaded:
            st.warning("Please load a dataset first.")
            return
        
        st.info(f"Dataset loaded: {len(st.session_state.coordinates)} points")
        
        # Index configuration
        st.subheader("Index Configuration")
        
        indexes_to_build = st.multiselect(
            "Select indexes to build",
            ["R-Tree", "ZM Linear", "ZM MLP"],
            default=["R-Tree", "ZM Linear"]
        )
        
        # Configuration for each index type
        config = {}
        
        if "R-Tree" in indexes_to_build:
            st.subheader("🌳 R-Tree Configuration")
            
            with st.expander("R-Tree Parameters Explanation", expanded=False):
                st.markdown("""
                - **Leaf Capacity**: Maximum spatial objects per leaf node (10-500)
                - **Overlap Factor**: Node splitting optimization (1-100, higher = better splits)
                - **Fill Factor**: Minimum node occupancy ratio (0.1-1.0)
                - **Split Algorithm**: Method for dividing overflowing nodes
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                leaf_capacity = st.slider("Leaf Capacity", 10, 500, 100, 
                    help="Higher values = fewer nodes, more objects per node")
                fill_factor = st.slider("Fill Factor", 0.1, 1.0, 0.8, 0.1,
                    help="Minimum node occupancy (0.8 = 80% full)")
            with col2:
                overlap_factor = st.slider("Overlap Factor", 1, 100, 32,
                    help="Higher values = better node splits, slower builds")
                split_algorithm = st.selectbox("Split Algorithm", 
                    ["linear", "quadratic", "rstar"], index=2,
                    help="R*-tree typically performs best")
            
            config["R-Tree"] = {
                "leaf_capacity": leaf_capacity,
                "near_minimum_overlap_factor": overlap_factor,
                "fill_factor": fill_factor,
                "split_algorithm": split_algorithm
            }
        
        if "ZM Linear" in indexes_to_build:
            st.subheader("📈 ZM Linear Configuration")
            
            with st.expander("Linear Model Parameters Explanation", expanded=False):
                st.markdown("""
                - **Polynomial Degree**: Model complexity (1=linear, 2=quadratic, etc.)
                - **Regularization**: Prevents overfitting (L1=Lasso, L2=Ridge)
                - **Alpha**: Regularization strength (higher = more regularization)
                - **Normalize Features**: Standardize input features for better convergence
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                degree = st.slider("Polynomial Degree", 1, 5, 1,
                    help="Higher degree = more complex model")
                regularization = st.selectbox("Regularization", 
                    ["none", "l1", "l2", "elastic"], index=2,
                    help="L2 (Ridge) typically works well")
            with col2:
                alpha = st.number_input("Regularization Strength", 0.0, 10.0, 1.0,
                    help="0.0 = no regularization, higher = more regularization")
                include_bias = st.checkbox("Include Bias", True,
                    help="Add constant offset term")
            
            normalize_features = st.checkbox("Normalize Features", True,
                help="Standardize inputs for better training")
            
            config["ZM Linear"] = {
                "degree": degree,
                "include_bias": include_bias,
                "regularization": regularization,
                "alpha": alpha,
                "normalize_features": normalize_features
            }
        
        if "ZM MLP" in indexes_to_build:
            st.subheader("🧠 ZM MLP Configuration")
            
            with st.expander("Neural Network Parameters Explanation", expanded=False):
                st.markdown("""
                - **Hidden Layers**: Network architecture (e.g., "64,32" = 2 layers)
                - **Activation**: Non-linear function between layers
                - **Learning Rate**: Optimization step size (0.001 typical)
                - **Batch Size**: Training examples processed together
                - **Dropout**: Regularization technique (0.2 = 20% neurons dropped)
                - **Early Stopping**: Stop training when validation stops improving
                """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                hidden_dims = st.text_input("Hidden Layers (comma-separated)", 
                    value="64,32", help="e.g., '128,64,32' for 3 layers")
                activation = st.selectbox("Activation Function", 
                    ["relu", "tanh", "sigmoid", "leaky_relu"], index=0,
                    help="ReLU typically works best")
                
            with col2:
                epochs = st.slider("Training Epochs", 10, 1000, 100,
                    help="More epochs = longer training, possibly better accuracy")
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 
                    format="%.4f", help="0.001 is a good starting point")
                
            with col3:
                batch_size = st.slider("Batch Size", 16, 1024, 128,
                    help="Power of 2 values work best (32, 64, 128, 256)")
                dropout = st.slider("Dropout Rate", 0.0, 0.8, 0.2, 0.1,
                    help="0.2-0.5 typical for regularization")
            
            col1, col2 = st.columns(2)
            with col1:
                optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0,
                    help="Adam typically works best for most problems")
            with col2:
                early_stopping = st.checkbox("Early Stopping", True,
                    help="Stop training when validation loss stops improving")
                if early_stopping:
                    patience = st.number_input("Early Stop Patience", 5, 50, 10,
                        help="Epochs to wait before stopping")
                else:
                    patience = None
            
            try:
                hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(",")]
            except:
                st.error("Invalid hidden dimensions format. Use comma-separated integers like '64,32'")
                hidden_dims_list = [64, 32]
            
            config["ZM MLP"] = {
                "hidden_dims": hidden_dims_list,
                "activation": activation,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dropout": dropout,
                "optimizer": optimizer,
                "early_stopping": early_stopping,
                "patience": patience
            }
        
        if st.button("Build Indexes"):
            with st.spinner("Building indexes..."):
                coordinates = st.session_state.coordinates
                morton_codes = st.session_state.morton_codes
                
                # Clear existing indexes
                self.query_engine.clear_all_indexes()
                
                build_results = {}
                
                for index_name in indexes_to_build:
                    try:
                        if index_name == "R-Tree":
                            self.query_engine.add_index(
                                "rtree",
                                IndexType.RTREE,
                                coordinates,
                                **config["R-Tree"]
                            )
                        elif index_name == "ZM Linear":
                            self.query_engine.add_index(
                                "zm_linear",
                                IndexType.ZM_LINEAR,
                                coordinates,
                                morton_codes,
                                **config["ZM Linear"]
                            )
                        elif index_name == "ZM MLP":
                            self.query_engine.add_index(
                                "zm_mlp",
                                IndexType.ZM_MLP,
                                coordinates,
                                morton_codes,
                                **config["ZM MLP"]
                            )
                        
                        build_results[index_name] = "Success"
                        
                    except Exception as e:
                        build_results[index_name] = f"Error: {e}"
                        logger.error(f"Error building {index_name}: {e}")
                
                st.session_state.indexes_built = True
                
                # Display build results
                if build_results:
                    st.subheader("Build Results")
                    for index_name, result in build_results.items():
                        if result == "Success":
                            st.success(f"✅ {index_name}: {result}")
                            
                            # Show R² score for learned indexes
                            if index_name in ["ZM Linear", "ZM MLP"]:
                                index_key = "zm_linear" if index_name == "ZM Linear" else "zm_mlp"
                                stats = self.query_engine.get_index_statistics()
                                if index_key in stats and 'r2_score' in stats[index_key]:
                                    r2_score = stats[index_key]['r2_score']
                                    if r2_score >= 0.8:
                                        st.success(f"   📊 Model R² Score: {r2_score:.4f} (Excellent fit)")
                                    elif r2_score >= 0.6:
                                        st.info(f"   📊 Model R² Score: {r2_score:.4f} (Good fit)")
                                    elif r2_score >= 0.4:
                                        st.warning(f"   📊 Model R² Score: {r2_score:.4f} (Moderate fit)")
                                    else:
                                        st.error(f"   📊 Model R² Score: {r2_score:.4f} (Poor fit)")
                        else:
                            st.error(f"❌ {index_name}: {result}")
                            logger.error(f"Error building {index_name}: {result}")
        
        # Display index statistics if any indexes were built successfully
        if st.session_state.indexes_built:
            st.subheader("📊 Index Statistics")
            
            stats = self.query_engine.get_index_statistics()
            if stats:
                stats_data = []
                
                for index_name, index_stats in stats.items():
                    row = {
                        "Index": index_name,
                        "Index Type": index_stats.get('index_type', 'Unknown'),
                        "Data Points": index_stats.get('num_points', 'N/A'),
                        "Build Time (s)": f"{index_stats.get('build_time', 0):.4f}",
                        "Memory Usage (MB)": f"{index_stats.get('memory_usage_mb', 0):.2f}"
                    }
                    
                    # Add R² score for learned indexes
                    if 'r2_score' in index_stats:
                        row["R² Score"] = f"{index_stats['r2_score']:.4f}"
                    
                    stats_data.append(row)
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, width='stretch')
            
            # Display R² score interpretation for learned indexes
            learned_indexes_with_r2 = [(name, stats[name]['r2_score']) 
                                     for name in stats 
                                     if 'r2_score' in stats[name]]
            
            if learned_indexes_with_r2:
                st.subheader("📈 Model Performance Analysis")
                
                for index_name, r2_score in learned_indexes_with_r2:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if r2_score >= 0.8:
                            st.success(f"**{index_name}**\nR² = {r2_score:.4f}")
                        elif r2_score >= 0.6:
                            st.info(f"**{index_name}**\nR² = {r2_score:.4f}")
                        elif r2_score >= 0.4:
                            st.warning(f"**{index_name}**\nR² = {r2_score:.4f}")
                        else:
                            st.error(f"**{index_name}**\nR² = {r2_score:.4f}")
                    
                    with col2:
                        if r2_score >= 0.8:
                            st.write("🎯 **Excellent fit** - Model explains >80% of variance in Morton code positions")
                        elif r2_score >= 0.6:
                            st.write("✅ **Good fit** - Model explains 60-80% of variance, should provide reliable approximations")
                        elif r2_score >= 0.4:
                            st.write("⚠️ **Moderate fit** - Model explains 40-60% of variance, may have accuracy issues")
                        else:
                            st.write("❌ **Poor fit** - Model explains <40% of variance, likely to have significant errors")
                
                # Overall recommendation based on best R² score
                best_r2 = max(r2_score for _, r2_score in learned_indexes_with_r2)
                
                if best_r2 >= 0.8:
                    st.success("🎉 **Recommendation**: Your learned indexes show excellent predictive power and should work well for spatial queries!")
                elif best_r2 >= 0.6:
                    st.info("👍 **Recommendation**: Your learned indexes show good predictive power. Consider comparing query accuracy with R-Tree results.")
                elif best_r2 >= 0.4:
                    st.warning("⚠️ **Recommendation**: Model performance is moderate. Consider tuning hyperparameters or using different features.")
                else:
                    st.error("🔧 **Recommendation**: Poor model performance detected. Try:\n- Increasing MLP hidden layer sizes\n- Using higher polynomial degree for linear model\n- Adding more training epochs\n- Checking if Morton codes correlate well with your spatial distribution")
    
    def _query_execution_page(self) -> None:
        """Interactive query execution page."""
        st.header("🔍 Query Execution")
        
        if not st.session_state.dataset_loaded:
            st.warning("⚠️ Please load a dataset first on the 'Data Loading' page.")
            return
        
        if not st.session_state.indexes_built:
            st.warning("⚠️ Please build indexes first on the 'Index Building' page.")
            return
        
        # Check if indexes are actually available
        available_indexes = self.query_engine.list_indexes()
        if not available_indexes:
            st.error("❌ No indexes are currently built. Please go to the 'Index Building' page and build some indexes first.")
            st.info("💡 Tip: Build at least one index (R-Tree, ZM Linear, or ZM MLP) before running queries.")
            return
        
        st.success(f"✅ Available indexes: {', '.join(available_indexes)}")
        
        # Query type selection
        query_type = st.selectbox(
            "Select Query Type",
            ["Point Query", "Range Query", "k-NN Query"]
        )
        
        # Get coordinate bounds from loaded data for smart defaults
        coords = st.session_state.coordinates
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        
        # Query parameters
        if query_type == "Point Query":
            st.subheader("🎯 Point Query Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                lat = st.number_input("Latitude", value=float(lat_center), format="%.6f",
                    help=f"Data range: {lat_min:.4f} to {lat_max:.4f}")
            with col2:
                lon = st.number_input("Longitude", value=float(lon_center), format="%.6f",
                    help=f"Data range: {lon_min:.4f} to {lon_max:.4f}")
            with col3:
                tolerance = st.number_input("Tolerance", value=0.001, format="%.6f",
                    help="Search radius around the point")
            
            if st.button("Execute Point Query", type="primary"):
                with st.spinner("Executing point query..."):
                    try:
                        # Store query parameters for visualization
                        st.session_state.last_query_params = {
                            'lat': lat, 'lon': lon, 'tolerance': tolerance
                        }
                        results = self.query_engine.point_query(lat, lon, tolerance=tolerance)
                        self._display_query_results(results, query_type)
                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
        
        elif query_type == "Range Query":
            st.subheader("📦 Range Query Parameters")
            col1, col2 = st.columns(2)
            with col1:
                min_lat = st.number_input("Min Latitude", value=float(lat_center - 0.01), format="%.6f",
                    help=f"Data range: {lat_min:.4f} to {lat_max:.4f}")
                max_lat = st.number_input("Max Latitude", value=float(lat_center + 0.01), format="%.6f",
                    help=f"Data range: {lat_min:.4f} to {lat_max:.4f}")
            with col2:
                min_lon = st.number_input("Min Longitude", value=float(lon_center - 0.01), format="%.6f",
                    help=f"Data range: {lon_min:.4f} to {lon_max:.4f}")
                max_lon = st.number_input("Max Longitude", value=float(lon_center + 0.01), format="%.6f",
                    help=f"Data range: {lon_min:.4f} to {lon_max:.4f}")
            
            if st.button("Execute Range Query", type="primary"):
                with st.spinner("Executing range query..."):
                    try:
                        # Store query parameters for visualization
                        st.session_state.last_query_params = {
                            'min_lat': min_lat, 'max_lat': max_lat, 
                            'min_lon': min_lon, 'max_lon': max_lon
                        }
                        results = self.query_engine.range_query(min_lat, max_lat, min_lon, max_lon)
                        self._display_query_results(results, query_type)
                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
        
        elif query_type == "k-NN Query":
            st.subheader("🔍 k-Nearest Neighbor Query Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                lat = st.number_input("Latitude", value=float(lat_center), format="%.6f",
                    help=f"Data range: {lat_min:.4f} to {lat_max:.4f}")
            with col2:
                lon = st.number_input("Longitude", value=float(lon_center), format="%.6f",
                    help=f"Data range: {lon_min:.4f} to {lon_max:.4f}")
            with col3:
                k = st.number_input("k (number of neighbors)", min_value=1, max_value=100, value=5,
                    help="Number of nearest points to find")
            
            if st.button("Execute k-NN Query", type="primary"):
                with st.spinner("Executing k-NN query..."):
                    try:
                        # Store query parameters for visualization
                        st.session_state.last_query_params = {
                            'lat': lat, 'lon': lon, 'k': k
                        }
                        results = self.query_engine.knn_query(lat, lon, k=k)
                        self._display_query_results(results, query_type)
                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
    
    def _display_query_results(self, results: Dict[str, Any], query_type: str) -> None:
        """Display query results with performance metrics and interactive visualization."""
        st.subheader("Query Results")
        
        # Performance comparison with accuracy indicators
        performance_data = []
        accuracy_issues = []
        
        for index_name, result in results.items():
            if 'error' not in result:
                # Get R² score if available
                stats = self.query_engine.get_index_statistics()
                index_key = index_name  # Use the actual index key
                r2_score = None
                if index_key in stats and 'r2_score' in stats[index_key]:
                    r2_score = stats[index_key]['r2_score']
                
                row_data = {
                    "Index": index_name,
                    "Type": result['index_type'],
                    "Results": result['count'],
                    "Query Time (ms)": f"{result['query_time_seconds'] * 1000:.4f}",
                    "Query Time (s)": f"{result['query_time_seconds']:.6f}"
                }
                
                # Add R² score if available (for learned indexes)
                if r2_score is not None:
                    row_data["R² Score"] = f"{r2_score:.4f}"
                    
                    # Flag potential accuracy issues based on R² score
                    if r2_score < 0.6:
                        accuracy_issues.append({
                            'index': index_name,
                            'r2_score': r2_score,
                            'issue': 'Low R² score may indicate poor query accuracy'
                        })
                
                performance_data.append(row_data)
        
        if performance_data:
            df_perf = pd.DataFrame(performance_data)
            st.dataframe(df_perf, width='stretch')
            
            # Show accuracy warnings if any
            if accuracy_issues:
                st.warning("⚠️ **Potential Accuracy Issues Detected:**")
                for issue in accuracy_issues:
                    st.write(f"- **{issue['index']}**: {issue['issue']} (R² = {issue['r2_score']:.4f})")
                st.info("💡 Consider validating results against R-Tree baseline or improving model training.")
            
            # Enhanced performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Query time comparison
                fig1 = px.bar(
                    df_perf, 
                    x="Index", 
                    y="Query Time (ms)",
                    color="Type",
                    title=f"{query_type} Performance Comparison",
                    color_discrete_map={
                        'RTREE': '#FF6B6B',
                        'ZM_LINEAR': '#4ECDC4', 
                        'ZM_MLP': '#45B7D1'
                    }
                )
                fig1.update_layout(
                    yaxis_title="Query Time (milliseconds)",
                    showlegend=True
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Results count comparison
                fig2 = px.bar(
                    df_perf,
                    x="Index",
                    y="Results", 
                    color="Type",
                    title="Results Count Comparison",
                    color_discrete_map={
                        'RTREE': '#FF6B6B',
                        'ZM_LINEAR': '#4ECDC4',
                        'ZM_MLP': '#45B7D1'  
                    }
                )
                fig2.update_layout(
                    yaxis_title="Number of Results",
                    showlegend=True
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Model performance indicator for learned indexes
            learned_indexes = [row for row in performance_data if "R² Score" in row]
            if learned_indexes:
                st.subheader("🎯 Model Performance Indicators")
                
                cols = st.columns(len(learned_indexes))
                for i, index_data in enumerate(learned_indexes):
                    with cols[i]:
                        r2_val = float(index_data["R² Score"])
                        
                        # Color-coded performance indicator
                        if r2_val >= 0.8:
                            st.success(f"**{index_data['Index']}**")
                            st.metric("R² Score", f"{r2_val:.4f}", "Excellent")
                            st.write("🎯 High accuracy expected")
                        elif r2_val >= 0.6:
                            st.info(f"**{index_data['Index']}**") 
                            st.metric("R² Score", f"{r2_val:.4f}", "Good")
                            st.write("✅ Reliable accuracy")
                        elif r2_val >= 0.4:
                            st.warning(f"**{index_data['Index']}**")
                            st.metric("R² Score", f"{r2_val:.4f}", "Moderate") 
                            st.write("⚠️ May have errors")
                        else:
                            st.error(f"**{index_data['Index']}**")
                            st.metric("R² Score", f"{r2_val:.4f}", "Poor")
                            st.write("❌ Likely inaccurate")
        
        # Add interactive map visualization
        self._create_query_results_map(results, query_type)
        
        # Add coordinate verification before validation
        self._debug_coordinate_existence(results, query_type)
        
        # Add validation check to compare results
        self._validate_query_results(results, query_type)
        
        # Detailed results for each index
        for index_name, result in results.items():
            with st.expander(f"{index_name} Details"):
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Results Count:** {result['count']}")
                        st.write(f"**Query Time:** {result['query_time_seconds']:.6f} seconds")
                        st.write(f"**Index Type:** {result['index_type']}")
                    
                    with col2:
                        # Show R² score and prediction quality for learned indexes
                        stats = self.query_engine.get_index_statistics()
                        if index_name in stats and 'r2_score' in stats[index_name]:
                            r2_score = stats[index_name]['r2_score']
                            st.write(f"**Model R² Score:** {r2_score:.4f}")
                            
                            if r2_score >= 0.8:
                                st.success("🎯 Excellent model fit")
                            elif r2_score >= 0.6:
                                st.info("✅ Good model fit") 
                            elif r2_score >= 0.4:
                                st.warning("⚠️ Moderate fit - check accuracy")
                            else:
                                st.error("❌ Poor fit - results may be unreliable")
                    
                    if result['results'] and len(result['results']) <= 100:
                        st.write("**Sample Results:**")
                        if query_type == "k-NN Query":
                            # For k-NN, show distances with ranking
                            knn_data = []
                            for i, (idx, dist) in enumerate(result['results'][:20]):
                                knn_data.append({
                                    "Rank": i+1,
                                    "Point Index": idx, 
                                    "Distance": f"{dist:.6f}"
                                })
                            st.dataframe(pd.DataFrame(knn_data))
                        else:
                            # For point/range queries, show indices in a nice format
                            result_indices = result['results'][:50]  # Limit display
                            if len(result_indices) <= 20:
                                st.write(f"Result indices: {result_indices}")
                            else:
                                st.write(f"First 20 results: {result_indices[:20]}")
                                st.write(f"... and {len(result['results']) - 20} more")
                    elif result['results']:
                        st.write(f"**Too many results to display** ({len(result['results'])} total)")
                        st.write(f"**Sample (first 10):** {result['results'][:10]}")
                    else:
                        st.write("**No results found**")
    
    def _create_query_results_map(self, results: Dict[str, Any], query_type: str) -> None:
        """Create interactive map showing query results for each index."""
        st.subheader("🗺️ Query Results Visualization")
        
        # Get coordinates for visualization
        coords = st.session_state.coordinates
        
        # Color scheme for different indexes
        index_colors = {
            'rtree': {'color': '#FF6B6B', 'name': 'R-Tree'},
            'zm_linear': {'color': '#4ECDC4', 'name': 'ZM Linear'},
            'zm_mlp': {'color': '#45B7D1', 'name': 'ZM MLP'},
        }
        
        # Create base map centered on data
        center_lat = float(np.mean(coords[:, 0]))
        center_lon = float(np.mean(coords[:, 1]))
        
        # Map configuration
        col1, col2 = st.columns(2)
        with col1:
            show_all_data = st.checkbox("Show background data points", value=True, 
                help="Display all data points in gray for context")
            max_bg_points = st.slider("Max background points", 100, 2000, 500,
                help="Limit background points for performance")
        with col2:
            marker_size = st.slider("Result marker size", 3, 10, 6)
            query_marker_size = st.slider("Query marker size", 5, 15, 10)
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=12,
            tiles="CartoDB positron"
        )
        
        # Add background data points if requested
        if show_all_data:
            # Sample data for background
            if len(coords) > max_bg_points:
                bg_indices = np.random.choice(len(coords), max_bg_points, replace=False)
                bg_coords = coords[bg_indices]
            else:
                bg_coords = coords
            
            for i, coord in enumerate(bg_coords):
                lat, lon = safe_coordinate_extract(bg_coords, i)
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=2,
                    color='lightgray',
                    fillColor='lightgray',
                    fillOpacity=0.3,
                    weight=1,
                    popup="Background data point"
                ).add_to(m)
        
        # Add query visualization based on query type
        self._add_query_visualization(m, query_type)
        
        # Add results for each index
        legend_html = self._add_index_results_to_map(m, results, coords, index_colors, marker_size)
        
        # Add legend
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display the map
        map_data = st_folium(m, width=700, height=600, returned_objects=["last_clicked"])
        
        # Results summary
        self._display_results_summary(results, index_colors)
    
    def _add_query_visualization(self, m: folium.Map, query_type: str) -> None:
        """Add query area/point visualization to the map."""
        if query_type == "Point Query":
            # Get query parameters from session state (stored during query execution)
            if hasattr(st.session_state, 'last_query_params'):
                params = st.session_state.last_query_params
                lat, lon, tolerance = params.get('lat', 0), params.get('lon', 0), params.get('tolerance', 0.001)
                
                # Add query point
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Query Point: ({lat:.6f}, {lon:.6f})",
                    icon=folium.Icon(color='red', icon='search', prefix='fa'),
                    tooltip="Query Point"
                ).add_to(m)
                
                # Add tolerance circle
                folium.Circle(
                    location=[lat, lon],
                    radius=tolerance * 111000,  # Convert degrees to meters (approximate)
                    color='red',
                    fillColor='red',
                    fillOpacity=0.1,
                    weight=2,
                    popup=f"Search radius: {tolerance:.6f}°"
                ).add_to(m)
        
        elif query_type == "Range Query":
            if hasattr(st.session_state, 'last_query_params'):
                params = st.session_state.last_query_params
                min_lat = params.get('min_lat', 0)
                max_lat = params.get('max_lat', 0)
                min_lon = params.get('min_lon', 0)
                max_lon = params.get('max_lon', 0)
                
                # Add query rectangle
                folium.Rectangle(
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    color='red',
                    fillColor='red',
                    fillOpacity=0.1,
                    weight=3,
                    popup=f"Query Range: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})"
                ).add_to(m)
        
        elif query_type == "k-NN Query":
            if hasattr(st.session_state, 'last_query_params'):
                params = st.session_state.last_query_params
                lat, lon, k = params.get('lat', 0), params.get('lon', 0), params.get('k', 5)
                
                # Add query point
                folium.Marker(
                    location=[lat, lon],
                    popup=f"k-NN Query Point: ({lat:.6f}, {lon:.6f})<br>k = {k}",
                    icon=folium.Icon(color='red', icon='search-plus', prefix='fa'),
                    tooltip=f"k-NN Query (k={k})"
                ).add_to(m)
    
    def _add_index_results_to_map(self, m: folium.Map, results: Dict[str, Any], 
                                  coords: np.ndarray, index_colors: Dict, marker_size: int) -> str:
        """Add result points for each index to the map with better visual separation."""
        coords = st.session_state.coordinates
        
        # DEBUG: Show full result counts before any limitations
        st.write(f"🔍 **FULL RESULTS DEBUG**:")
        for index_name, result in results.items():
            if 'error' not in result and result['results']:
                st.write(f"   - {index_name}: {len(result['results'])} total results")
                # Check if Point 5237 is in the FULL result set
                if isinstance(result['results'][0], tuple):  # k-NN results
                    point_indices = [idx for idx, dist in result['results']]
                else:  # Point/Range results
                    point_indices = result['results']
                
                if 5237 in point_indices:
                    position = point_indices.index(5237)
                    st.write(f"     ✅ Point 5237 found at position {position} in full {index_name} results")
                else:
                    st.write(f"     ❌ Point 5237 NOT found in {index_name} results")
        
        # Different marker styles for each index to handle overlapping points
        index_styles = {
            'rtree': {
                'color': '#FF6B6B', 
                'name': 'R-Tree', 
                'fillColor': '#FF6B6B',
                'weight': 3,
                'fillOpacity': 0.8,
                'radius_offset': 0
            },
            'zm_linear': {
                'color': '#4ECDC4', 
                'name': 'ZM Linear',
                'fillColor': '#4ECDC4', 
                'weight': 2,
                'fillOpacity': 0.6,
                'radius_offset': 2  # Slightly larger to show overlap
            },
            'zm_mlp': {
                'color': '#45B7D1', 
                'name': 'ZM MLP',
                'fillColor': '#45B7D1',
                'weight': 4,
                'fillOpacity': 0.4,
                'radius_offset': 4  # Even larger to show overlap
            },
        }
        
        # FIXED: Instead of limiting each index separately, collect ALL unique points
        # and track which indexes found each point
        all_result_points = set()  # All unique point indices found by any index
        index_result_sets = {}     # index_name -> set of point indices
        
        # First, collect all result sets
        for index_name, result in results.items():
            if 'error' in result or not result['results']:
                continue
                
            result_indices = result['results']
            if isinstance(result_indices[0], tuple):  # k-NN results with distances
                point_indices = [idx for idx, dist in result_indices]
                # For k-NN, store both indices and distances
                index_result_sets[index_name] = {
                    'points': set(point_indices),
                    'details': {idx: (i, dist) for i, (idx, dist) in enumerate(result_indices)}
                }
            else:  # Point/Range query results
                point_indices = result_indices
                index_result_sets[index_name] = {
                    'points': set(point_indices), 
                    'details': {idx: (i, None) for i, idx in enumerate(point_indices)}
                }
            
            all_result_points.update(point_indices)
        
        # Limit total points for map performance, but ensure we get overlapping points
        max_map_points = 1500  # Increased limit for better visualization
        
        if len(all_result_points) > max_map_points:
            st.warning(f"⚠️ Too many result points ({len(all_result_points)}) for optimal map performance. Showing overlapping points and a sample of others.")
            
            # Find points that appear in multiple indexes (overlapping points)
            overlapping_points = set()
            for point_idx in all_result_points:
                found_by_count = sum(1 for idx_data in index_result_sets.values() 
                                   if point_idx in idx_data['points'])
                if found_by_count > 1:
                    overlapping_points.add(point_idx)
            
            # Prioritize overlapping points + sample of others
            single_index_points = all_result_points - overlapping_points
            remaining_slots = max_map_points - len(overlapping_points)
            
            if remaining_slots > 0 and single_index_points:
                sampled_single = set(list(single_index_points)[:remaining_slots])
                points_to_show = overlapping_points | sampled_single
            else:
                points_to_show = overlapping_points
                
            st.info(f"📊 Showing {len(overlapping_points)} overlapping points + {len(points_to_show) - len(overlapping_points)} others = {len(points_to_show)} total")
        else:
            points_to_show = all_result_points
            st.info(f"📊 Showing all {len(points_to_show)} result points")
        
        # Track which points have been found by which indexes
        point_results = {}  # point_idx -> list of (index_name, rank, distance/None)
        
        st.write(f"\n🔍 **SMART SAMPLING DEBUG**:")
        
        # Build the visualization data for selected points
        for point_idx in points_to_show:
            if point_idx < len(coords):  # Validate point index
                point_results[point_idx] = []
                
                # Check which indexes found this point
                for index_name, idx_data in index_result_sets.items():
                    if point_idx in idx_data['points']:
                        rank, distance = idx_data['details'][point_idx]
                        point_results[point_idx].append((index_name, rank, distance))
        
        # Debug info about overlapping vs single-index points
        overlapping_in_view = sum(1 for point_data in point_results.values() if len(point_data) > 1)
        single_in_view = len(point_results) - overlapping_in_view
        
        st.write(f"   - Points with multiple indexes: {overlapping_in_view}")
        st.write(f"   - Points with single index: {single_in_view}")
        
        # Specific debug for Point 5237
        if 5237 in point_results:
            indexes_found = [x[0] for x, _, _ in point_results[5237]]
            st.write(f"   - ✅ Point 5237 will be shown, found by: {indexes_found}")
        else:
            st.write(f"   - ❌ Point 5237 not in visualization set")
        
        # Second pass: add markers with visual separation for overlapping points
        marker_count_debug = {}  # Track how many markers we actually create
        
        for point_idx, index_list in point_results.items():
            # Use direct coordinate extraction like original working code
            lat, lon = coords[point_idx]
            
            # Sort by index priority but ADD ALL LAYERS, not just the first one
            index_priority = {'rtree': 0, 'zm_linear': 1, 'zm_mlp': 2}
            index_list.sort(key=lambda x: index_priority.get(x[0], 99))
            
            # DEBUG: Add debugging for point 5237 specifically
            if point_idx == 5237:
                st.write(f"🔍 **DEBUG Point 5237 MAP CREATION**: Found by {len(index_list)} indexes: {[x[0] for x, _, _ in index_list]}")
                st.write(f"   - Coordinates: ({lat:.6f}, {lon:.6f})")
                st.write(f"   - About to create {len(index_list)} markers...")
                
            for layer_idx, (index_name, result_idx, distance) in enumerate(index_list):
                style = index_styles.get(index_name, index_styles['rtree'])
                
                # Track marker creation
                if index_name not in marker_count_debug:
                    marker_count_debug[index_name] = 0
                marker_count_debug[index_name] += 1
                
                # DEBUG: Specific logging for point 5237
                if point_idx == 5237:
                    st.write(f"   - Creating marker {layer_idx + 1}/{len(index_list)}: {style['name']} (color: {style['color']})")
                
                # Create popup text
                if distance is not None:  # k-NN result
                    popup_text = (f"<b>{style['name']}</b> Result #{result_idx+1}<br>"
                                f"Point Index: {point_idx}<br>"
                                f"Coordinates: ({lat:.6f}, {lon:.6f})<br>"
                                f"Distance: {distance:.6f}")
                    tooltip_text = f"{style['name']} - Rank {result_idx+1}"
                else:  # Point/Range result
                    popup_text = (f"<b>{style['name']}</b> Result<br>"
                                f"Point Index: {point_idx}<br>"
                                f"Coordinates: ({lat:.6f}, {lon:.6f})")
                    tooltip_text = f"{style['name']} Result"
                
                # Add overlap indicator to popup if multiple indexes found this point
                if len(index_list) > 1:
                    other_indexes = [x[0] for x in index_list if x[0] != index_name]
                    popup_text += f"<br><i>Also found by: {', '.join(other_indexes)}</i>"
                
                # Create marker with proper visual separation - all indexes get their marker
                # Use different radius offsets to show layering
                adjusted_radius = marker_size + style['radius_offset']
                
                # For overlapping points, make the markers slightly transparent so you can see through layers
                adjusted_opacity = style['fillOpacity'] if len(index_list) == 1 else style['fillOpacity'] * 0.7
                
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=adjusted_radius,
                    color=style['color'],
                    fillColor=style['fillColor'],
                    fillOpacity=adjusted_opacity,  # Adjusted opacity for overlapping
                    weight=style['weight'],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=tooltip_text
                ).add_to(m)
        
        # DEBUG: Show marker creation summary
        st.write(f"\n🎨 **Marker Creation Summary:**")
        for index_name, count in marker_count_debug.items():
            style = index_styles.get(index_name, {'name': index_name, 'color': '#999999'})
            st.write(f"   - {style['name']}: {count} markers created (color: {style.get('color', 'unknown')})")
        
        # Create improved legend HTML with better styling
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: rgba(255, 255, 255, 0.95); 
                    border: 2px solid #333333; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    z-index: 9999; 
                    font-size: 13px; 
                    font-family: Arial, sans-serif;
                    padding: 12px;
                    backdrop-filter: blur(3px);">
        <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
            📊 Query Results
        </h4>
        <div style="margin-bottom: 8px; color: #666;">
            <i class="fa fa-search" style="color: red; margin-right: 5px;"></i>
            <small>Query Location</small>
        </div>
        '''
        
        # Add legend entries for each index that returned results
        for index_name, result in results.items():
            if 'error' not in result and result['results']:
                style = index_styles.get(index_name, {'color': '#999999', 'name': index_name})
                
                # Create visual indicator that matches the map marker
                marker_style = (f"display: inline-block; width: 12px; height: 12px; "
                              f"border-radius: 50%; background-color: {style['fillColor']}; "
                              f"border: 2px solid {style['color']}; margin-right: 8px; "
                              f"vertical-align: middle;")
                
                legend_html += f'''
                <div style="margin-bottom: 6px; display: flex; align-items: center;">
                    <span style="{marker_style}"></span>
                    <span style="color: #333; font-weight: 500;">{style['name']}</span>
                    <span style="color: #666; margin-left: auto; font-size: 11px;">({result["count"]})</span>
                </div>
                '''
        
        # Add background data legend if shown
        if st.session_state.get('show_all_data', True):
            legend_html += '''
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 8px; height: 8px; 
                                 border-radius: 50%; background-color: lightgray; 
                                 margin-right: 8px; vertical-align: middle;"></span>
                    <span style="color: #666; font-size: 11px;">Background Data</span>
                </div>
            </div>
            '''
        
        # Add helpful note about overlapping points
        legend_html += '''
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
            <div style="color: #666; font-size: 10px; line-height: 1.3;">
                💡 <strong>Tip:</strong> Overlapping results shown with different sizes and transparency. Click markers for details.
            </div>
        </div>
        '''
        
        legend_html += '</div>'
        
        return legend_html
    
    def _display_results_summary(self, results: Dict[str, Any], index_colors: Dict) -> None:
        """Display a summary comparison of results from different indexes."""
        st.subheader("📊 Results Comparison")
        
        # Check for result consistency
        result_counts = []
        for index_name, result in results.items():
            if 'error' not in result:
                result_counts.append(result['count'])
        
        if result_counts and len(set(result_counts)) > 1:
            st.warning("⚠️ **Result Count Mismatch**: Different indexes returned different numbers of results. "
                      "This could indicate implementation differences or approximation errors in learned indexes.")
        
        # Results breakdown
        cols = st.columns(len([r for r in results.values() if 'error' not in r]))
        
        for i, (index_name, result) in enumerate(results.items()):
            if 'error' not in result:
                color_info = index_colors.get(index_name, {'color': '#999999', 'name': index_name})
                
                with cols[i]:
                    st.markdown(f"**{color_info['name']}**")
                    
                    # Performance indicator
                    if result['count'] > 0:
                        st.success("✅ Results found")
                    else:
                        st.info("ℹ️ No results found")
        
        # Add instructions for map interaction
        st.info("💡 **Map Interaction Tips:**\n"
               "- Click on result points to see details\n" 
               "- Red markers/areas show your query\n"
               "- Colored circles show results from each index\n"
               "- Gray dots are background data points for context")
    
    def _validate_query_results(self, results: Dict[str, Any], query_type: str) -> None:
        """Validate query results by comparing learned index results with R-Tree baseline."""
        st.subheader("🔍 Query Results Validation")
        
        if 'rtree' not in results:
            st.warning("⚠️ R-Tree results not available for validation.")
            return
        
        if 'error' in results['rtree']:
            st.error(f"❌ R-Tree query failed: {results['rtree']['error']}")
            return
        
        # Get R-Tree results as baseline
        if query_type == "k-NN Query":
            # For k-NN, compare point indices only (ignore distances)
            rtree_results = set([idx for idx, dist in results['rtree']['results']])
        else:
            rtree_results = set(results['rtree']['results'])
        
        st.write(f"**R-Tree baseline:** {len(rtree_results)} results")
        
        # Compare each learned index against R-Tree
        validation_data = []
        has_issues = False
        
        for index_name, result in results.items():
            if index_name == 'rtree' or 'error' in result:
                continue
            
            # Extract result indices for comparison
            if query_type == "k-NN Query":
                learned_results = set([idx for idx, dist in result['results']])
            else:
                learned_results = set(result['results'])
            
            # Calculate accuracy metrics
            intersection = rtree_results & learned_results
            missing_in_learned = rtree_results - learned_results
            extra_in_learned = learned_results - rtree_results
            
            accuracy = len(intersection) / len(rtree_results) if len(rtree_results) > 0 else 1.0
            precision = len(intersection) / len(learned_results) if len(learned_results) > 0 else 1.0
            recall = len(intersection) / len(rtree_results) if len(rtree_results) > 0 else 1.0
            
            validation_data.append({
                "Index": index_name,
                "Total Results": len(learned_results),
                "Correct Results": len(intersection),
                "Missing Results": len(missing_in_learned),
                "Extra Results": len(extra_in_learned),
                "Accuracy": f"{accuracy:.3f}",
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}"
            })
            
            # Check for significant accuracy issues
            if accuracy < 0.9 or len(missing_in_learned) > 0 or len(extra_in_learned) > 0:
                has_issues = True
        
        if validation_data:
            # Display validation table
            df_validation = pd.DataFrame(validation_data)
            st.dataframe(df_validation, width='stretch')
            
            # Overall assessment
            if not has_issues:
                st.success("✅ **Excellent!** All learned indexes match R-Tree results perfectly.")
            else:
                st.warning("⚠️ **Accuracy Issues Detected:** Some learned indexes have approximation errors.")
                
                # Detailed issue analysis
                with st.expander("🔍 Detailed Issue Analysis", expanded=True):
                    for index_name, result in results.items():
                        if index_name == 'rtree' or 'error' in result:
                            continue
                        
                        if query_type == "k-NN Query":
                            learned_results = set([idx for idx, dist in result['results']])
                        else:
                            learned_results = set(result['results'])
                        
                        missing = rtree_results - learned_results
                        extra = learned_results - rtree_results
                        
                        if missing or extra:
                            st.write(f"**{index_name} Issues:**")
                            if missing:
                                st.write(f"- Missing {len(missing)} results that R-Tree found: {list(missing)[:10]}{'...' if len(missing) > 10 else ''}")
                            if extra:
                                st.write(f"- Found {len(extra)} extra results not in R-Tree: {list(extra)[:10]}{'...' if len(extra) > 10 else ''}")
                            st.write("---")
                
                # Recommendations
                st.info("💡 **Possible Causes & Solutions:**\n"
                       "- **Poor model training:** Low R² score indicates bad predictions\n"
                       "- **Insufficient search radius:** Learned indexes might need larger search windows\n" 
                       "- **Data distribution mismatch:** Morton codes might not correlate well with your spatial data\n"
                       "- **Try:** Increase polynomial degree, more MLP epochs, or larger search windows")
        else:
            st.info("ℹ️ No learned indexes available for validation.")
    
    def _debug_coordinate_existence(self, results: Dict[str, Any], query_type: str) -> None:
        """Debug function to check if query coordinates exist in the dataset and analyze the mismatch."""
        st.subheader("🔍 Query Coordinate Analysis")
        
        # Validate spatial accuracy first
        self._validate_spatial_accuracy(results, query_type)
        
        # DETAILED SPATIAL INVESTIGATION
        st.subheader("🐛 Detailed Spatial Investigation")
        
        if not hasattr(st.session_state, 'last_query_params'):
            st.warning("No recent query parameters to validate.")
            return
        
        coords = st.session_state.coordinates
        params = st.session_state.last_query_params
        
        if query_type == "Point Query":
            query_lat = params.get('lat')
            query_lon = params.get('lon')
            tolerance = params.get('tolerance', 0.001)
            
            st.write(f"**Investigating Point Query:** ({query_lat:.6f}, {query_lon:.6f}) with tolerance {tolerance:.6f}")
            
            # Compare first 20 results from each index
            for index_name, result in results.items():
                if 'error' in result or not result['results']:
                    continue
                    
                st.write(f"\n**{index_name} - First 20 Results:**")
                
                sample_results = result['results'][:20]
                spatial_data = []
                
                for i, point_idx in enumerate(sample_results):
                    if point_idx < len(coords):
                        point_lat, point_lon = safe_coordinate_extract(coords, point_idx)
                        distance = np.sqrt((query_lat - point_lat)**2 + (query_lon - point_lon)**2)
                        within_tolerance = "✅" if distance <= tolerance else "❌"
                        
                        spatial_data.append({
                            "Rank": i+1,
                            "Point Index": int(point_idx),
                            "Latitude": f"{float(point_lat)::.6f}",
                            "Longitude": f"{float(point_lon)::.6f}",
                            "Distance": f"{float(distance)::.6f}",
                            "Within Tolerance": within_tolerance,
                            "Excess Distance": f"{float(max(0, distance - tolerance)):.6f}"
                        })
                
                if spatial_data:
                    df_spatial = pd.DataFrame(spatial_data)
                    st.dataframe(df_spatial, use_container_width=True)
                    
                    # Count valid vs invalid
                    valid_count = sum(1 for row in spatial_data if row["Within Tolerance"] == "✅")
                    invalid_count = len(spatial_data) - valid_count
                    
                    if invalid_count > 0:
                        st.error(f"❌ **{index_name}**: {invalid_count}/{len(spatial_data)} sample results are OUTSIDE tolerance!")
                    else:
                        st.success(f"✅ **{index_name}**: All {len(spatial_data)} sample results are within tolerance")
            
            # Direct comparison of result sets
            st.subheader("📊 Result Set Comparison")
            
            # Get all result sets
            result_sets = {}
            for index_name, result in results.items():
                if 'error' not in result and result['results']:
                    result_sets[index_name] = set(result['results'])
            
            if len(result_sets) >= 2:
                index_names = list(result_sets.keys())
                
                # Compare each pair
                for i in range(len(index_names)):
                    for j in range(i + 1, len(index_names)):
                        idx1, idx2 = index_names[i], index_names[j]
                        set1, set2 = result_sets[idx1], result_sets[idx2]
                        
                        intersection = set1 & set2
                        only_in_1 = set1 - set2
                        only_in_2 = set2 - set1
                        
                        st.write(f"\n**{idx1} vs {idx2}:**")
                        st.write(f"- Common results: {len(intersection)}")
                        st.write(f"- Only in {idx1}: {len(only_in_1)}")
                        st.write(f"- Only in {idx2}: {len(only_in_2)}")
                        st.write(f"- Overlap percentage: {len(intersection) / len(set1) * 100:.1f}%")
                        
                        # Show some examples of different results
                        if only_in_1:
                            st.write(f"- Sample unique to {idx1}: {list(only_in_1)[:10]}")
                        if only_in_2:
                            st.write(f"- Sample unique to {idx2}: {list(only_in_2)[:10]}")
            
            # Geographic distribution analysis
            st.subheader("🗺️ Geographic Distribution Analysis")
            
            for index_name, result in results.items():
                if 'error' in result or not result['results']:
                    continue
                
                sample_size = min(100, len(result['results']))
                sample_indices = result['results'][:sample_size]
                
                lats = [coords[idx][0] for idx in sample_indices if idx < len(coords)]
                lons = [coords[idx][1] for idx in sample_indices if idx < len(coords)]
                
                if lats and lons:
                    lat_center = np.mean(lats)
                    lon_center = np.mean(lons)
                    lat_std = np.std(lats)
                    lon_std = np.std(lons)
                    
                    st.write(f"**{index_name} Geographic Stats (sample of {len(lats)} points):**")
                    st.write(f"- Center: ({lat_center:.6f}, {lon_center:.6f})")
                    st.write(f"- Latitude spread (std): {lat_std:.6f}")
                    st.write(f"- Longitude spread (std): {lon_std:.6f}")
                    st.write(f"- Distance from query center: {np.sqrt((lat_center - query_lat)**2 + (lon_center - query_lon)**2):.6f}")
        
        # ADD DEBUG INFO FOR IDENTICAL COUNTS ISSUE
        st.subheader("🐛 Debug: Result Sets Detailed Comparison")
        st.write("**Investigating why indexes find same count but different points...**")
        
        if query_type == "Point Query" and len(results) >= 2:
            query_lat = params.get('lat')
            query_lon = params.get('lon') 
            tolerance = params.get('tolerance', 0.001)
            
            # Get result sets for comparison
            result_sets = {}
            for index_name, result in results.items():
                if 'error' not in result and result['results']:
                    result_sets[index_name] = set(result['results'])
                    st.write(f"**{index_name}**: {len(result['results'])} results")
            
            # Compare sets pairwise
            if len(result_sets) >= 2:
                index_names = list(result_sets.keys())
                
                # Find points that are unique to each index
                for i, index1 in enumerate(index_names):
                    for j in range(i + 1, len(index_names)):
                        index2 = index_names[j]
                        
                        set1 = result_sets[index1]
                        set2 = result_sets[index2] 
                        
                        common = set1 & set2
                        only_in_1 = set1 - set2
                        only_in_2 = set2 - set1
                        
                        st.write(f"\n**{index1} vs {index2} Comparison:**")
                        st.write(f"- Common points: {len(common)} ({len(common)/len(set1)*100:.1f}%)")
                        st.write(f"- Only in {index1}: {len(only_in_1)}")  
                        st.write(f"- Only in {index2}: {len(only_in_2)}")
                        
                        # Show specific examples of different points
                        if only_in_1:
                            st.write(f"\n**Points ONLY found by {index1}:**")
                            sample_unique_1 = list(only_in_1)[:10]  # Show first 10
                            
                            unique_data_1 = []
                            for idx in sample_unique_1:
                                if idx < len(coords):
                                    lat, lon = coords[idx]
                                    distance = np.sqrt((query_lat - lat)**2 + (query_lon - lon)**2)
                                    unique_data_1.append({
                                        "Point Index": idx,
                                        "Latitude": f"{lat:.6f}",
                                        "Longitude": f"{lon:.6f}",
                                        "Distance": f"{distance:.6f}",
                                        "Within Tolerance": "✅" if distance <= tolerance else "❌",
                                        "Why Unique": "Found by algo difference" if distance <= tolerance else "ERROR: Outside tolerance!"
                                    })
                            
                            if unique_data_1:
                                st.dataframe(pd.DataFrame(unique_data_1))
                        
                        if only_in_2:
                            st.write(f"\n**Points ONLY found by {index2}:**")
                            sample_unique_2 = list(only_in_2)[:10]  # Show first 10
                            
                            unique_data_2 = []
                            for idx in sample_unique_2:
                                if idx < len(coords):
                                    lat, lon = coords[idx]
                                    distance = np.sqrt((query_lat - lat)**2 + (query_lon - lon)**2)
                                    unique_data_2.append({
                                        "Point Index": idx,
                                        "Latitude": f"{lat:.6f}",
                                        "Longitude": f"{lon:.6f}",
                                        "Distance": f"{distance:.6f}",
                                        "Within Tolerance": "✅" if distance <= tolerance else "❌",
                                        "Why Unique": "Found by algo difference" if distance <= tolerance else "ERROR: Outside tolerance!"
                                    })
                            
                            if unique_data_2:
                                st.dataframe(pd.DataFrame(unique_data_2))
                
                # Investigate the specific point you mentioned
                st.write(f"\n**🔍 Investigating Point Index 5237:**")
                
                point_5237_lat, point_5237_lon = coords[5237]
                point_5237_distance = np.sqrt((query_lat - point_5237_lat)**2 + (query_lon - point_5237_lon)**2)
                
                st.write(f"- Coordinates: ({point_5237_lat:.6f}, {point_5237_lon:.6f})")
                st.write(f"- Distance from query: {point_5237_distance:.6f}")
                st.write(f"- Within tolerance ({tolerance:.6f}): {'✅ Yes' if point_5237_distance <= tolerance else '❌ No'}")
                
                # Check which indexes found this specific point
                found_by = []
                for index_name, result_set in result_sets.items():
                    if 5237 in result_set:
                        found_by.append(index_name)
                
                st.write(f"- Found by indexes: {found_by if found_by else 'None'}")
                
                if len(found_by) == 1:
                    st.error(f"🚨 **Point 5237 is ONLY found by {found_by[0]}!** This explains the red marker.")
                elif len(found_by) > 1:
                    st.info(f"✅ Point 5237 is found by multiple indexes: {found_by}")
                else:
                    st.warning("⚠️ Point 5237 not found by any index - this shouldn't happen!")
                
                # Check boundary cases - points very close to tolerance boundary
                st.write(f"\n**🎯 Boundary Analysis - Points near tolerance edge:**")
                
                boundary_analysis = []
                tolerance_margin = tolerance * 0.01  # 1% margin
                
                for index_name, result_set in result_sets.items():
                    boundary_points = []
                    for idx in list(result_set)[:100]:  # Check first 100 points
                        if idx < len(coords):
                            lat, lon = coords[idx]
                            distance = np.sqrt((query_lat - lat)**2 + (query_lon - lon)**2)
                            
                            # Check if point is close to tolerance boundary
                            if abs(distance - tolerance) <= tolerance_margin:
                                boundary_points.append({
                                    'idx': idx,
                                    'distance': distance,
                                    'margin': distance - tolerance
                                })
                    
                    if boundary_points:
                        boundary_analysis.append({
                            'index': index_name,
                            'boundary_points': len(boundary_points),
                            'sample': boundary_points[:5]  # Show first 5
                        })
                
                for analysis in boundary_analysis:
                    st.write(f"- **{analysis['index']}**: {analysis['boundary_points']} points near tolerance boundary")
                    if analysis['sample']:
                        for pt in analysis['sample']:
                            st.write(f"  - Point {pt['idx']}: distance={pt['distance']:.6f}, margin={pt['margin']:.6f}")
        
        # ...existing code...

    def _validate_spatial_accuracy(self, results: Dict[str, Any], query_type: str) -> None:
        """Validate spatial accuracy by checking which results are actually within tolerance/bounds."""
        st.subheader("🎯 Spatial Accuracy Validation")
        
        if not hasattr(st.session_state, 'last_query_params'):
            st.warning("No recent query parameters to validate.")
            return
        
        coords = st.session_state.coordinates
        params = st.session_state.last_query_params
        
        if query_type == "Point Query":
            query_lat = params.get('lat')
            query_lon = params.get('lon')
            tolerance = params.get('tolerance', 0.001)
            
            st.write(f"**Validating Point Query:** ({query_lat:.6f}, {query_lon:.6f}) with tolerance {tolerance:.6f}")
            
            validation_data = []
            
            for index_name, result in results.items():
                if 'error' in result or not result['results']:
                    continue
                
                # Check each result point
                valid_count = 0
                invalid_count = 0
                invalid_points = []
                
                for point_idx in result['results']:
                    if point_idx < len(coords):
                        point_lat, point_lon = safe_coordinate_extract(coords, point_idx)
                        # Calculate actual distance
                        distance = np.sqrt((query_lat - point_lat)**2 + (query_lon - point_lon)**2)
                        
                        if distance <= tolerance:
                            valid_count += 1
                        else:
                            invalid_count += 1
                            invalid_points.append({
                                'idx': point_idx,
                                'lat': point_lat,
                                'lon': point_lon,
                                'distance': distance,
                                'excess': distance - tolerance
                            })
                
                accuracy = valid_count / len(result['results']) if result['results'] else 0
                
                validation_data.append({
                    'Index': index_name,
                    'Total Results': len(result['results']),
                    'Valid Results': valid_count,
                    'Invalid Results': invalid_count,
                    'Accuracy': f"{accuracy:.1%}",
                    'Status': "✅ Perfect" if invalid_count == 0 else f"❌ {invalid_count} errors"
                })
                
                # Show detailed invalid points for problematic indexes
                if invalid_count > 0:
                    st.error(f"**{index_name}** has {invalid_count} invalid results:")
                    
                    # Show worst offenders
                    invalid_points.sort(key=lambda x: x['distance'], reverse=True)
                    for i, point in enumerate(invalid_points[:5]):  # Show top 5 worst
                        st.write(f"  - Point {point['idx']}: ({point['lat']:.6f}, {point['lon']:.6f}) "
                               f"Distance: {point['distance']:.6f} (excess: +{point['excess']:.6f})")
                    
                    if len(invalid_points) > 5:
                        st.write(f"  ... and {len(invalid_points) - 5} more invalid points")
            
            # Display validation summary
            if validation_data:
                st.subheader("📊 Accuracy Summary")
                df_validation = pd.DataFrame(validation_data)
                st.dataframe(df_validation, use_container_width=True)
                
                # Overall assessment
                perfect_indexes = [row for row in validation_data if row['Invalid Results'] == 0]
                if len(perfect_indexes) == len(validation_data):
                    st.success("🎉 **All indexes have perfect spatial accuracy!**")
                else:
                    st.error("⚠️ **Some indexes have spatial accuracy problems.**")
                    
                    # Find the most accurate index
                    most_accurate = max(validation_data, key=lambda x: float(x['Accuracy'].strip('%')) / 100)
                    st.info(f"**Most Accurate:** {most_accurate['Index']} ({most_accurate['Accuracy']})")
        
        elif query_type == "Range Query":
            min_lat = params.get('min_lat')
            max_lat = params.get('max_lat')
            min_lon = params.get('min_lon')
            max_lon = params.get('max_lon')
            
            st.write(f"**Validating Range Query:** [{min_lat:.6f}, {max_lat:.6f}] × [{min_lon:.6f}, {max_lon:.6f}]")
            
            validation_data = []
            
            for index_name, result in results.items():
                if 'error' in result or not result['results']:
                    continue
                
                valid_count = 0
                invalid_count = 0
                
                for point_idx in result['results']:
                    if point_idx < len(coords):
                        point_lat, point_lon = safe_coordinate_extract(coords, point_idx)
                        
                        # Check if point is within rectangle
                        if (min_lat <= point_lat <= max_lat and 
                            min_lon <= point_lon <= max_lon):
                            valid_count += 1
                        else:
                            invalid_count += 1
                
                accuracy = valid_count / len(result['results']) if result['results'] else 0
                
                validation_data.append({
                    'Index': index_name,
                    'Total Results': len(result['results']),
                    'Valid Results': valid_count,
                    'Invalid Results': invalid_count,
                    'Accuracy': f"{accuracy:.1%}",
                    'Status': "✅ Perfect" if invalid_count == 0 else f"❌ {invalid_count} errors"
                })
            
            if validation_data:
                st.dataframe(pd.DataFrame(validation_data), use_container_width=True)

    def _run_quick_benchmark(self, query_type: str, selected_indexes: List[str]) -> None:
        """Run a quick benchmark for the specified query type."""
        if not selected_indexes:
            st.error("Please select at least one index to benchmark.")
            return
        
        with st.spinner(f"Running {query_type} benchmark..."):
            try:
                if query_type == "point":
                    benchmark = QueryBenchmark(
                        QueryType.POINT,
                        num_queries=100,
                        tolerance=0.001,
                        description="Quick Point Query Test"
                    )
                elif query_type == "range":
                    benchmark = QueryBenchmark(
                        QueryType.RANGE,
                        num_queries=50,
                        selectivity=0.001,
                        description="Quick Range Query Test"
                    )
                elif query_type == "knn":
                    benchmark = QueryBenchmark(
                        QueryType.KNN,
                        num_queries=50,
                        k=5,
                        description="Quick k-NN Query Test"
                    )
                else:
                    st.error(f"Unknown query type: {query_type}")
                    return
                
                # Run the benchmark
                metrics = self.evaluator.run_benchmark(benchmark, index_names=selected_indexes)
                self._display_quick_benchmark_results(metrics, benchmark.description)
                
            except Exception as e:
                st.error(f"Quick benchmark failed: {e}")
                logger.error(f"Quick benchmark error: {e}", exc_info=True)

    def _performance_evaluation_page(self) -> None:
        """Performance evaluation and benchmarking page."""
        st.header("⚡ Performance Evaluation")
        
        if not st.session_state.indexes_built:
            st.warning("Please build indexes first.")
            return
        
        # Check available indexes
        available_indexes = self.query_engine.list_indexes()
        if not available_indexes:
            st.error("No indexes available for evaluation.")
            return
        
        st.success(f"✅ Available indexes: {', '.join(available_indexes)}")
        
        # Initialize evaluator
        self.evaluator = PerformanceEvaluator(self.query_engine)
        
        # Benchmark configuration section
        st.subheader("🎯 Benchmark Configuration")
        
        # Advanced configuration options
        with st.expander("📊 Benchmark Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Query Volumes**")
                num_point_queries = st.number_input("Point Queries", min_value=10, max_value=10000, value=1000,
                    help="Number of random point queries to execute")
                num_range_queries = st.number_input("Range Queries", min_value=10, max_value=5000, value=500,
                    help="Number of random range queries to execute")
                num_knn_queries = st.number_input("k-NN Queries", min_value=10, max_value=5000, value=500,
                    help="Number of random k-NN queries to execute")
            
            with col2:
                st.markdown("**Query Parameters**")
                k_values = st.multiselect("k values for k-NN", [1, 5, 10, 20, 50], default=[1, 5, 10],
                    help="Different k values to test for k-NN queries")
                selectivities = st.multiselect("Range selectivities (%)", [0.01, 0.1, 1.0, 5.0], default=[0.01, 0.1, 1.0],
                    help="Percentage of total area covered by range queries")
                point_tolerance = st.number_input("Point query tolerance", value=0.001, format="%.6f",
                    help="Search radius for point queries")
            
            with col3:
                st.markdown("**Evaluation Options**")
                include_accuracy = st.checkbox("Include accuracy validation", value=True,
                    help="Compare learned index results with R-Tree baseline")
                include_scalability = st.checkbox("Include scalability analysis", value=True,
                    help="Test performance with different data subset sizes")
                random_seed = st.number_input("Random seed", value=42, help="For reproducible benchmarks")
                
                # Index selection for evaluation
                selected_indexes = st.multiselect(
                    "Indexes to evaluate", 
                    available_indexes, 
                    default=available_indexes,
                    help="Select which indexes to include in the evaluation"
                )
        
        # Quick benchmark buttons
        st.subheader("🚀 Quick Benchmarks")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📍 Point Query Test", help="Quick point query benchmark"):
                self._run_quick_benchmark("point", selected_indexes)
        
        with col2:
            if st.button("📦 Range Query Test", help="Quick range query benchmark"):
                self._run_quick_benchmark("range", selected_indexes)
        
        with col3:
            if st.button("🎯 k-NN Query Test", help="Quick k-NN benchmark"):
                self._run_quick_benchmark("knn", selected_indexes)
        
        with col4:
            if st.button("🔍 Accuracy Test", help="Quick accuracy validation"):
                self._run_accuracy_validation(selected_indexes)
        
        # Comprehensive evaluation
        st.subheader("🏆 Comprehensive Evaluation")
        
        if st.button("🎯 Run Full Benchmark Suite", type="primary"):
            with st.spinner("Running comprehensive evaluation... This may take several minutes."):
                try:
                    # Create custom benchmark suite based on user configuration
                    benchmarks = []
                    
                    # Point queries
                    if num_point_queries > 0:
                        benchmarks.append(QueryBenchmark(
                            QueryType.POINT,
                            num_queries=num_point_queries,
                            tolerance=point_tolerance,
                            description=f"Point Query Benchmark ({num_point_queries} queries)"
                        ))
                    
                    # Range queries with different selectivities
                    for selectivity in selectivities:
                        if num_range_queries > 0:
                            benchmarks.append(QueryBenchmark(
                                QueryType.RANGE,
                                num_queries=num_range_queries,
                                selectivity=selectivity / 100.0,  # Convert percentage to fraction
                                description=f"Range Query Benchmark - {selectivity}% selectivity"
                            ))
                    
                    # k-NN queries with different k values
                    for k in k_values:
                        if num_knn_queries > 0:
                            benchmarks.append(QueryBenchmark(
                                QueryType.KNN,
                                num_queries=num_knn_queries,
                                k=k,
                                description=f"{k}-NN Query Benchmark"
                            ))
                    
                    # Run all benchmarks
                    all_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, benchmark in enumerate(benchmarks):
                        status_text.text(f"Running: {benchmark.description}")
                        progress_bar.progress((i + 1) / len(benchmarks))
                        
                        try:
                            metrics = self.evaluator.run_benchmark(benchmark, index_names=selected_indexes)
                            all_results[benchmark.description] = metrics
                        except Exception as e:
                            st.error(f"Error in benchmark {benchmark.description}: {e}")
                            all_results[benchmark.description] = {'error': str(e)}
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results and create comprehensive analysis
                    evaluation_results = {
                        'benchmarks': all_results,
                        'config': {
                            'indexes': selected_indexes,
                            'num_benchmarks': len(benchmarks),
                            'include_accuracy': include_accuracy,
                            'random_seed': random_seed
                        },
                        'timestamp': time.time()
                    }
                    
                    st.session_state.evaluation_results = evaluation_results
                    
                    st.success("✅ Comprehensive evaluation completed!")
                    
                    # Display comprehensive results
                    self._display_comprehensive_results(evaluation_results, include_accuracy)
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {e}")
                    logger.error(f"Comprehensive evaluation error: {e}", exc_info=True)
        
        # Display saved results if available
        if st.session_state.evaluation_results:
            st.subheader("📊 Previous Evaluation Results")
            
            if st.button("🔄 Show Detailed Analysis"):
                self._display_comprehensive_results(
                    st.session_state.evaluation_results, 
                    include_accuracy=True
                )
            
            # Export options
            st.subheader("💾 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download CSV Report"):
                    csv_data = self._generate_csv_report(st.session_state.evaluation_results)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"spatial_index_benchmark_{int(time.time())}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📄 Download JSON Report"):
                    json_data = self._generate_json_report(st.session_state.evaluation_results)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"spatial_index_benchmark_{int(time.time())}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("📈 Generate Summary Report"):
                    report = self._generate_summary_report(st.session_state.evaluation_results)
                    st.text_area("📋 Performance Summary Report", report, height=400)
    
    def _run_accuracy_validation(self, selected_indexes: List[str]) -> None:
        """Run a focused accuracy validation test."""
        if 'rtree' not in selected_indexes:
            st.error("R-Tree must be included for accuracy validation.")
            return
        
        with st.spinner("Running accuracy validation..."):
            try:
                # Run small benchmark with focus on accuracy
                benchmarks = [
                    QueryBenchmark(QueryType.POINT, num_queries=50, tolerance=0.001, description="Point Accuracy Test"),
                    QueryBenchmark(QueryType.RANGE, num_queries=25, selectivity=0.001, description="Range Accuracy Test"),
                    QueryBenchmark(QueryType.KNN, num_queries=25, k=5, description="k-NN Accuracy Test")
                ]
                
                accuracy_results = {}
                for benchmark in benchmarks:
                    metrics = self.evaluator.run_benchmark(benchmark, index_names=selected_indexes)
                    accuracy_results[benchmark.description] = metrics
                
                self._display_accuracy_validation_results(accuracy_results)
                
            except Exception as e:
                st.error(f"Accuracy validation failed: {e}")
    
    def _display_quick_benchmark_results(self, metrics: Dict[str, Any], benchmark_name: str) -> None:
        """Display results from a quick benchmark."""
        st.subheader(f"⚡ {benchmark_name} Results")
        
        if not metrics:
            st.warning("No results to display.")
            return
        
        # Performance comparison table
        perf_data = []
        for index_name, metric in metrics.items():
            perf_data.append({
                "Index": index_name,
                "Query Time (ms)": f"{metric.avg_query_time * 1000:.4f}",
                "Throughput (q/s)": f"{metric.throughput_queries_per_sec:.2f}",
                "Memory (MB)": f"{metric.memory_usage_mb:.2f}",
                "Build Time (s)": f"{metric.build_time:.4f}"
            })
        
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, width='stretch')
        
        # Quick visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(df_perf, x="Index", y="Query Time (ms)", 
                         title="Query Time Comparison",
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(df_perf, x="Index", y="Throughput (q/s)",
                         title="Throughput Comparison", 
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Performance winner announcement
        fastest_index = min(metrics.items(), key=lambda x: x[1].avg_query_time)
        st.success(f"🏆 **Fastest**: {fastest_index[0]} ({fastest_index[1].avg_query_time*1000:.2f} ms avg)")
    
    def _display_accuracy_validation_results(self, results: Dict[str, Any]) -> None:
        """Display focused accuracy validation results."""
        st.subheader("🎯 Accuracy Validation Results")
        
        if not results:
            st.warning("No accuracy results to display.")
            return
        
        # TODO: Implement accuracy comparison logic
        # This would compare learned index results against R-Tree baseline
        st.info("Accuracy validation implementation would compare learned index results with R-Tree baseline across different query types.")
        
        for benchmark_name, metrics in results.items():
            with st.expander(f"📊 {benchmark_name} Accuracy Analysis"):
                if metrics:
                    accuracy_data = []
                    for index_name, metric in metrics.items():
                        accuracy_data.append({
                            "Index": index_name,
                            "Precision": f"{metric.accuracy_metrics.get('precision', 1.0):.4f}",
                            "Recall": f"{metric.accuracy_metrics.get('recall', 1.0):.4f}",
                            "F1 Score": f"{metric.accuracy_metrics.get('f1_score', 1.0):.4f}",
                            "Error Rate": f"{metric.error_rate:.4f}"
                        })
                    
                    st.dataframe(pd.DataFrame(accuracy_data))
                else:
                    st.warning("No metrics available for this benchmark.")
    
    def _display_comprehensive_results(self, evaluation_results: Dict[str, Any], include_accuracy: bool) -> None:
        """Display comprehensive evaluation results with detailed analysis."""
        st.subheader("🏆 Comprehensive Evaluation Results")
        
        benchmarks = evaluation_results['benchmarks']
        config = evaluation_results.get('config', {})
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        successful_benchmarks = len([b for b in benchmarks.values() if 'error' not in b])
        total_benchmarks = len(benchmarks)
        
        with col1:
            st.metric("Total Benchmarks", total_benchmarks)
        with col2:
            st.metric("Successful Tests", successful_benchmarks)
        with col3:
            st.metric("Indexes Tested", len(config.get('indexes', [])))
        with col4:
            success_rate = (successful_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Aggregate performance analysis
        st.subheader("📊 Aggregate Performance Analysis")
        
        # Collect all metrics across benchmarks
        index_performance = {}
        
        for benchmark_name, benchmark_results in benchmarks.items():
            if 'error' not in benchmark_results:
                for index_name, metrics in benchmark_results.items():
                    if index_name not in index_performance:
                        index_performance[index_name] = {
                            'query_times': [],
                            'throughputs': [],
                            'build_time': metrics.build_time,
                            'memory_usage': metrics.memory_usage_mb,
                            'index_type': metrics.index_type
                        }
                    
                    index_performance[index_name]['query_times'].append(metrics.avg_query_time * 1000)  # Convert to ms
                    index_performance[index_name]['throughputs'].append(metrics.throughput_queries_per_sec)
        
        if index_performance:
            # Create comprehensive performance table
            perf_summary = []
            for index_name, data in index_performance.items():
                perf_summary.append({
                    "Index": index_name,
                    "Type": data['index_type'],
                    "Avg Query Time (ms)": f"{np.mean(data['query_times']):.4f}",
                    "Min Query Time (ms)": f"{np.min(data['query_times']):.4f}",
                    "Max Query Time (ms)": f"{np.max(data['query_times']):.4f}",
                    "Avg Throughput (q/s)": f"{np.mean(data['throughputs']):.2f}",
                    "Build Time (s)": f"{data['build_time']:.4f}",
                    "Memory Usage (MB)": f"{data['memory_usage']:.2f}"
                })
            
            df_summary = pd.DataFrame(perf_summary)
            st.dataframe(df_summary, width='stretch')
            
            # Performance visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Query time distribution
                query_time_data = []
                for index_name, data in index_performance.items():
                    for time_ms in data['query_times']:
                        query_time_data.append({
                            'Index': index_name,
                            'Query Time (ms)': time_ms,
                            'Type': data['index_type']
                        })
                
                df_times = pd.DataFrame(query_time_data)
                fig1 = px.box(df_times, x="Index", y="Query Time (ms)", color="Type",
                             title="Query Time Distribution Across All Benchmarks")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Memory vs Performance scatter
                memory_perf_data = []
                for index_name, data in index_performance.items():
                    memory_perf_data.append({
                        'Index': index_name,
                        'Memory Usage (MB)': data['memory_usage'],
                        'Avg Query Time (ms)': np.mean(data['query_times']),
                        'Type': data['index_type']
                    })
                
                df_scatter = pd.DataFrame(memory_perf_data)
                fig2 = px.scatter(df_scatter, x="Memory Usage (MB)", y="Avg Query Time (ms)",
                                 color="Type", size="Memory Usage (MB)", hover_name="Index",
                                 title="Memory Usage vs Query Performance")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Performance rankings
            st.subheader("🏅 Performance Rankings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**⚡ Fastest Query Time**")
                fastest = sorted(index_performance.items(), 
                               key=lambda x: np.mean(x[1]['query_times']))
                for i, (name, data) in enumerate(fastest[:3]):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{medal} {name}: {np.mean(data['query_times']):.2f} ms")
            
            with col2:
                st.markdown("**🚀 Highest Throughput**")
                highest_throughput = sorted(index_performance.items(),
                                          key=lambda x: np.mean(x[1]['throughputs']), reverse=True)
                for i, (name, data) in enumerate(highest_throughput[:3]):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{medal} {name}: {np.mean(data['throughputs']):.0f} q/s")
            
            with col3:
                st.markdown("**💾 Lowest Memory Usage**")
                lowest_memory = sorted(index_performance.items(),
                                     key=lambda x: x[1]['memory_usage'])
                for i, (name, data) in enumerate(lowest_memory[:3]):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{medal} {name}: {data['memory_usage']:.1f} MB")
        
        # Detailed benchmark results
        st.subheader("📋 Detailed Benchmark Results")
        
        for benchmark_name, benchmark_results in benchmarks.items():
            with st.expander(f"📊 {benchmark_name}"):
                if 'error' in benchmark_results:
                    st.error(f"❌ Benchmark failed: {benchmark_results['error']}")
                else:
                    # Individual benchmark performance table
                    bench_data = []
                    for index_name, metrics in benchmark_results.items():
                        bench_data.append({
                            "Index": index_name,
                            "Query Time (ms)": f"{metrics.avg_query_time * 1000:.4f}",
                            "Min Time (ms)": f"{metrics.min_query_time * 1000:.4f}",
                            "Max Time (ms)": f"{metrics.max_query_time * 1000:.4f}",
                            "Std Dev (ms)": f"{metrics.std_query_time * 1000:.4f}",
                            "Throughput (q/s)": f"{metrics.throughput_queries_per_sec:.2f}"
                        })
                    
                    st.dataframe(pd.DataFrame(bench_data))
                    
                    # Individual benchmark chart
                    fig = px.bar(pd.DataFrame(bench_data), x="Index", y="Query Time (ms)",
                               title=f"{benchmark_name} - Query Time Comparison")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on results
        self._generate_performance_recommendations(index_performance)
    
    def _generate_performance_recommendations(self, index_performance: Dict[str, Any]) -> None:
        """Generate performance recommendations based on benchmark results."""
        st.subheader("🎯 Performance Recommendations")
        
        if not index_performance:
            st.info("No performance data available for recommendations.")
            return
        
        # Analyze performance patterns
        recommendations = []
        
        # Find fastest index
        fastest_index = min(index_performance.items(), 
                          key=lambda x: np.mean(x[1]['query_times']))
        recommendations.append(
            f"🏆 **Best Overall Performance**: {fastest_index[0]} with average query time of "
            f"{np.mean(fastest_index[1]['query_times']):.2f} ms"
        )
        
        # Find most memory efficient
        most_efficient = min(index_performance.items(),
                           key=lambda x: x[1]['memory_usage'])
        recommendations.append(
            f"💾 **Most Memory Efficient**: {most_efficient[0]} using only "
            f"{most_efficient[1]['memory_usage']:.1f} MB"
        )
        
        # Analyze learned index performance
        learned_indexes = {k: v for k, v in index_performance.items() 
                          if v['index_type'] in ['ZM_LINEAR', 'ZM_MLP']}
        
        if learned_indexes:
            if len(learned_indexes) > 1:
                best_learned = min(learned_indexes.items(),
                                 key=lambda x: np.mean(x[1]['query_times']))
                recommendations.append(
                    f"🧠 **Best Learned Index**: {best_learned[0]} outperforms other learned methods"
                )
            
            # Compare with R-Tree if available
            if 'rtree' in index_performance:
                rtree_time = np.mean(index_performance['rtree']['query_times'])
                
                faster_learned = []
                for name, data in learned_indexes.items():
                    learned_time = np.mean(data['query_times'])
                    if learned_time < rtree_time:
                        speedup = rtree_time / learned_time
                        faster_learned.append((name, speedup))
                
                if faster_learned:
                    best_speedup = max(faster_learned, key=lambda x: x[1])
                    recommendations.append(
                        f"⚡ **Speed Improvement**: {best_speedup[0]} is {best_speedup[1]:.1f}x faster than R-Tree!"
                    )
                else:
                    recommendations.append(
                        f"⚠️ **Performance Gap**: All learned indexes are slower than R-Tree. "
                        "Consider tuning hyperparameters or improving model training."
                    )
        
        # Display recommendations
        for rec in recommendations:
            st.info(rec)
        
        # Usage recommendations
        st.markdown("**💡 Usage Recommendations:**")
        
        if 'rtree' in index_performance:
            st.write("- **R-Tree**: Best for guaranteed accuracy and complex spatial queries")
        
        if any(v['index_type'] == 'ZM_LINEAR' for v in index_performance.values()):
            st.write("- **ZM Linear**: Good for simple spatial distributions with linear patterns")
        
        if any(v['index_type'] == 'ZM_MLP' for v in index_performance.values()):
            st.write("- **ZM MLP**: Best for complex spatial distributions with non-linear patterns")
        
        st.write("- **Memory-constrained environments**: Choose the index with lowest memory usage")
        st.write("- **High-throughput applications**: Choose the index with highest queries/second")
        st.write("- **Accuracy-critical applications**: Always validate learned index results against R-Tree")
    
    def _generate_csv_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate CSV report from evaluation results."""
        import io
        
        output = io.StringIO()
        
        # Write header
        output.write("Benchmark,Index,Index_Type,Avg_Query_Time_ms,Min_Query_Time_ms,Max_Query_Time_ms,")
        output.write("Std_Query_Time_ms,Throughput_qps,Build_Time_s,Memory_Usage_MB,Error_Rate\n")
        
        # Write data
        for benchmark_name, benchmark_results in evaluation_results['benchmarks'].items():
            if 'error' not in benchmark_results:
                for index_name, metrics in benchmark_results.items():
                    output.write(f"{benchmark_name},{index_name},{metrics.index_type},")
                    output.write(f"{metrics.avg_query_time*1000:.4f},{metrics.min_query_time*1000:.4f},")
                    output.write(f"{metrics.max_query_time*1000:.4f},{metrics.std_query_time*1000:.4f},")
                    output.write(f"{metrics.throughput_queries_per_sec:.2f},{metrics.build_time:.4f},")
                    output.write(f"{metrics.memory_usage_mb:.2f},{metrics.error_rate:.4f}\n")
        
        return output.getvalue()
    
    def _generate_json_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate JSON report from evaluation results."""
        import json
        
        # Convert PerformanceMetrics objects to dictionaries for JSON serialization
        json_results = {
            'config': evaluation_results.get('config', {}),
            'timestamp': evaluation_results.get('timestamp', time.time()),
            'benchmarks': {}
        }
        
        for benchmark_name, benchmark_results in evaluation_results['benchmarks'].items():
            if 'error' in benchmark_results:
                json_results['benchmarks'][benchmark_name] = benchmark_results
            else:
                json_results['benchmarks'][benchmark_name] = {
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
        
        return json.dumps(json_results, indent=2)
    
    def _generate_summary_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        from datetime import datetime
        
        report = []
        report.append("=" * 80)
        report.append("SPATIAL INDEX PERFORMANCE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.fromtimestamp(evaluation_results.get('timestamp', time.time()))}")
        report.append("")
        
        config = evaluation_results.get('config', {})
        report.append("CONFIGURATION:")
        report.append(f"- Indexes Tested: {', '.join(config.get('indexes', []))}")
        report.append(f"- Total Benchmarks: {len(evaluation_results['benchmarks'])}")
        report.append(f"- Random Seed: {config.get('random_seed', 'N/A')}")
        report.append("")
        
        # Aggregate performance summary
        index_performance = {}
        successful_benchmarks = 0
        
        for benchmark_name, benchmark_results in evaluation_results['benchmarks'].items():
            if 'error' not in benchmark_results:
                successful_benchmarks += 1
                for index_name, metrics in benchmark_results.items():
                    if index_name not in index_performance:
                        index_performance[index_name] = {
                            'query_times': [],
                            'throughputs': [],
                            'build_time': metrics.build_time,
                            'memory_usage': metrics.memory_usage_mb,
                            'index_type': metrics.index_type
                        }
                    
                    index_performance[index_name]['query_times'].append(metrics.avg_query_time * 1000)
                    index_performance[index_name]['throughputs'].append(metrics.throughput_queries_per_sec)
        
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"- Successful Benchmarks: {successful_benchmarks}/{len(evaluation_results['benchmarks'])}")
        report.append("")
        
        if index_performance:
            report.append("INDEX PERFORMANCE OVERVIEW:")
            report.append("-" * 50)
            
            for index_name, data in index_performance.items():
                avg_time = np.mean(data['query_times'])
                avg_throughput = np.mean(data['throughputs'])
                
                report.append(f"{index_name} ({data['index_type']}):")
                report.append(f"  Average Query Time: {avg_time:.4f} ms")
                report.append(f"  Average Throughput: {avg_throughput:.2f} queries/sec")
                report.append(f"  Memory Usage: {data['memory_usage']:.2f} MB")
                report.append(f"  Build Time: {data['build_time']:.4f} seconds")
                report.append("")
            
            # Performance rankings
            fastest = min(index_performance.items(), key=lambda x: np.mean(x[1]['query_times']))
            highest_throughput = max(index_performance.items(), key=lambda x: np.mean(x[1]['throughputs']))
            lowest_memory = min(index_performance.items(), key=lambda x: x[1]['memory_usage'])
            
            report.append("PERFORMANCE LEADERS:")
            report.append(f"🏆 Fastest Query Time: {fastest[0]} ({np.mean(fastest[1]['query_times']):.4f} ms)")
            report.append(f"🚀 Highest Throughput: {highest_throughput[0]} ({np.mean(highest_throughput[1]['throughputs']):.2f} q/s)")
            report.append(f"💾 Lowest Memory Usage: {lowest_memory[0]} ({lowest_memory[1]['memory_usage']:.2f} MB)")
            report.append("")
        
        # Detailed benchmark results
        report.append("DETAILED BENCHMARK RESULTS:")
        report.append("=" * 50)
        
        for benchmark_name, benchmark_results in evaluation_results['benchmarks'].items():
            report.append(f"\n{benchmark_name}:")
            report.append("-" * len(benchmark_name))
            
            if 'error' in benchmark_results:
                report.append(f"❌ ERROR: {benchmark_results['error']}")
            else:
                for index_name, metrics in benchmark_results.items():
                    report.append(f"  {index_name}:")
                    report.append(f"    Query Time: {metrics.avg_query_time*1000:.4f} ms (±{metrics.std_query_time*1000:.4f})")
                    report.append(f"    Throughput: {metrics.throughput_queries_per_sec:.2f} queries/sec")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _visualization_page(self) -> None:
        """Data visualization and mapping page."""
        st.header("🗺️ Visualization")
        
        if not st.session_state.dataset_loaded:
            st.warning("Please load a dataset first.")
            return
        
        coordinates = st.session_state.coordinates
        
        # Sample data for visualization (limit for performance)
        max_points = st.slider("Max points to display", 100, min(10000, len(coordinates)), 5000)
        
        if len(coordinates) > max_points:
            indices = np.random.choice(len(coordinates), max_points, replace=False)
            vis_coords = coordinates[indices]
        else:
            vis_coords = coordinates
        
        # Create interactive map
        st.subheader("Interactive Map")
        
        # Calculate map center
        center_lat = float(np.mean(vis_coords[:, 0]))
        center_lon = float(np.mean(vis_coords[:, 1]))
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add points to map
        for i, coord in enumerate(vis_coords[::max(1, len(vis_coords)//1000)]):  # Subsample for performance
            lat, lon = safe_coordinate_extract(vis_coords, i)
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='red',
                fillColor='red'
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Coordinate scatter plot
        st.subheader("Coordinate Distribution")
        
        fig = px.scatter(
            x=vis_coords[:, 1],
            y=vis_coords[:, 0],
            labels={'x': 'Longitude', 'y': 'Latitude'},
            title=f"Spatial Distribution ({len(vis_coords)} points)",
            opacity=0.6
        )
        fig.update_traces(marker_size=2)
        st.plotly_chart(fig, width='stretch')


def main():
    """Main function to run the Streamlit app."""
    # Configure logging to show debug information
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/tmp/zm_rtree_debug.log')
        ]
    )
    
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()