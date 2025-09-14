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
        
        # Simple benchmark configuration
        st.subheader("🎯 Benchmark Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            num_point_queries = st.number_input("Point Queries", min_value=10, max_value=1000, value=100)
            num_range_queries = st.number_input("Range Queries", min_value=10, max_value=1000, value=100)
        with col2:
            num_knn_queries = st.number_input("k-NN Queries", min_value=10, max_value=1000, value=100)
            k_value = st.number_input("k for k-NN", min_value=1, max_value=50, value=5)
        
        # Quick benchmark buttons
        st.subheader("🚀 Quick Benchmarks")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📍 Point Query Test", help="Quick point query benchmark"):
                self._run_quick_benchmark("point", num_point_queries)
        
        with col2:
            if st.button("📦 Range Query Test", help="Quick range query benchmark"):
                self._run_quick_benchmark("range", num_range_queries)
        
        with col3:
            if st.button("🎯 k-NN Query Test", help="Quick k-NN benchmark"):
                self._run_quick_benchmark("knn", num_knn_queries, k_value)
        
        # Comprehensive evaluation
        if st.button("🏆 Run Full Benchmark Suite", type="primary"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    # Create benchmarks
                    benchmarks = [
                        QueryBenchmark(QueryType.POINT, num_queries=num_point_queries, 
                                     tolerance=0.001, description="Point Query Benchmark"),
                        QueryBenchmark(QueryType.RANGE, num_queries=num_range_queries,
                                     selectivity=0.01, description="Range Query Benchmark"),
                        QueryBenchmark(QueryType.KNN, num_queries=num_knn_queries,
                                     k=k_value, description=f"{k_value}-NN Query Benchmark")
                    ]
                    
                    # Run benchmarks
                    all_results = {}
                    progress_bar = st.progress(0)
                    
                    for i, benchmark in enumerate(benchmarks):
                        st.text(f"Running: {benchmark.description}")
                        progress_bar.progress((i + 1) / len(benchmarks))
                        
                        try:
                            metrics = self.evaluator.run_benchmark(benchmark, index_names=available_indexes)
                            all_results[benchmark.description] = metrics
                        except Exception as e:
                            st.error(f"Error in benchmark {benchmark.description}: {e}")
                            all_results[benchmark.description] = {'error': str(e)}
                    
                    progress_bar.empty()
                    
                    # Store and display results
                    st.session_state.evaluation_results = {
                        'benchmarks': all_results,
                        'timestamp': time.time()
                    }
                    
                    st.success("✅ Comprehensive evaluation completed!")
                    self._display_benchmark_results(all_results)
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {e}")
        
        # Display saved results if available
        if st.session_state.evaluation_results:
            st.subheader("📊 Previous Evaluation Results")
            if st.button("🔄 Show Results"):
                self._display_benchmark_results(st.session_state.evaluation_results['benchmarks'])
    
    def _run_quick_benchmark(self, benchmark_type: str, num_queries: int, k: int = 5) -> None:
        """Run a quick benchmark for a specific query type."""
        with st.spinner(f"Running {benchmark_type} benchmark..."):
            try:
                if benchmark_type == "point":
                    benchmark = QueryBenchmark(QueryType.POINT, num_queries=num_queries, 
                                             tolerance=0.001, description="Quick Point Query Test")
                elif benchmark_type == "range":
                    benchmark = QueryBenchmark(QueryType.RANGE, num_queries=num_queries,
                                             selectivity=0.01, description="Quick Range Query Test")
                elif benchmark_type == "knn":
                    benchmark = QueryBenchmark(QueryType.KNN, num_queries=num_queries,
                                             k=k, description=f"Quick {k}-NN Query Test")
                
                results = self.evaluator.run_benchmark(benchmark)
                self._display_quick_benchmark_results(results, benchmark.description)
                
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
    
    def _display_quick_benchmark_results(self, results: Dict[str, Any], benchmark_name: str) -> None:
        """Display results from a quick benchmark."""
        st.subheader(f"📊 {benchmark_name} Results")
        
        if 'error' in results:
            st.error(f"Benchmark failed: {results['error']}")
            return
        
        # Performance comparison table
        perf_data = []
        for index_name, metrics in results.items():
            perf_data.append({
                "Index": index_name,
                "Avg Query Time (ms)": f"{metrics.avg_query_time * 1000:.4f}",
                "Throughput (queries/sec)": f"{metrics.throughput_queries_per_sec:.2f}",
                "Total Queries": metrics.total_queries
            })
        
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, width='stretch')
        
        # Performance visualization
        fig = px.bar(df_perf, x="Index", y="Avg Query Time (ms)",
                    title=f"{benchmark_name} - Average Query Time",
                    color="Index")
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_benchmark_results(self, all_results: Dict[str, Any]) -> None:
        """Display comprehensive benchmark results."""
        st.subheader("📈 Comprehensive Benchmark Results")
        
        # Overall summary
        successful_benchmarks = sum(1 for r in all_results.values() if 'error' not in r)
        total_benchmarks = len(all_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Successful Benchmarks", f"{successful_benchmarks}/{total_benchmarks}")
        with col2:
            success_rate = (successful_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Results for each benchmark
        for benchmark_name, benchmark_results in all_results.items():
            with st.expander(f"📊 {benchmark_name}"):
                if 'error' in benchmark_results:
                    st.error(f"❌ Benchmark failed: {benchmark_results['error']}")
                else:
                    # Performance table
                    bench_data = []
                    for index_name, metrics in benchmark_results.items():
                        bench_data.append({
                            "Index": index_name,
                            "Query Time (ms)": f"{metrics.avg_query_time * 1000:.4f}",
                            "Std Dev (ms)": f"{metrics.std_query_time * 1000:.4f}",
                            "Throughput (q/s)": f"{metrics.throughput_queries_per_sec:.2f}",
                            "Total Queries": metrics.total_queries
                        })
                    
                    st.dataframe(pd.DataFrame(bench_data))
                    
                    # Performance chart
                    fig = px.bar(pd.DataFrame(bench_data), x="Index", y="Query Time (ms)",
                               title=f"{benchmark_name} - Query Time Comparison")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Overall performance summary
        self._display_performance_summary(all_results)
    
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
        """Add result points for each index to the map."""
        for index_name, result in results.items():
            if 'error' not in result and result['results']:
                color = index_colors.get(index_name, {}).get('color', 'blue')
                
                # Handle different result formats
                for item in result['results']:
                    # For k-NN queries, results are tuples of (index, distance)
                    # For point/range queries, results are just indices
                    if isinstance(item, tuple) and len(item) >= 2:
                        # k-NN result: (index, distance)
                        point_idx, distance = item[0], item[1]
                        popup_text = f"{index_name} Result: Point {point_idx}<br>Distance: {distance:.6f}"
                    else:
                        # Point/range query result: just index
                        point_idx = item
                        popup_text = f"{index_name} Result: Point {point_idx}"
                    
                    # Ensure point_idx is valid and within bounds
                    if isinstance(point_idx, (int, np.integer)) and 0 <= point_idx < len(coords):
                        lat, lon = safe_coordinate_extract(coords, point_idx)
                        folium.CircleMarker(
                            location=[float(lat), float(lon)],
                            radius=marker_size,
                            color=color,
                            fillColor=color,
                            fillOpacity=0.6,
                            weight=1,
                            popup=popup_text
                        ).add_to(m)
        
        # Create legend HTML
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border: 1px solid black; padding: 10px;">
        <h4>Legend</h4>
        '''
        for index_name, color_info in index_colors.items():
            legend_html += f'<i style="background:{color_info["color"]};width:10px;height:10px;display:inline-block;"></i> {color_info["name"]}<br>'
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
    
    def _display_performance_summary(self, all_results: Dict[str, Any]) -> None:
        """Display overall performance summary across all benchmarks."""
        st.subheader("🏆 Overall Performance Summary")
        
        # Aggregate performance data
        index_performance = {}
        
        for benchmark_name, benchmark_results in all_results.items():
            if 'error' not in benchmark_results:
                for index_name, metrics in benchmark_results.items():
                    if index_name not in index_performance:
                        index_performance[index_name] = {
                            'query_times': [],
                            'throughputs': []
                        }
                    
                    index_performance[index_name]['query_times'].append(metrics.avg_query_time * 1000)  # Convert to ms
                    index_performance[index_name]['throughputs'].append(metrics.throughput_queries_per_sec)
        
        if index_performance:
            # Performance comparison table
            summary_data = []
            for index_name, data in index_performance.items():
                summary_data.append({
                    "Index": index_name,
                    "Avg Query Time (ms)": f"{np.mean(data['query_times']):.4f}",
                    "Min Query Time (ms)": f"{np.min(data['query_times']):.4f}",
                    "Max Query Time (ms)": f"{np.max(data['query_times']):.4f}",
                    "Avg Throughput (q/s)": f"{np.mean(data['throughputs']):.2f}",
                    "Benchmarks": len(data['query_times'])
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, width='stretch')
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Average query time comparison
                fig1 = px.bar(df_summary, x="Index", y="Avg Query Time (ms)",
                             title="Average Query Time Across All Benchmarks")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Average throughput comparison
                fig2 = px.bar(df_summary, x="Index", y="Avg Throughput (q/s)",
                             title="Average Throughput Across All Benchmarks")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Performance insights
            best_time_idx = df_summary.loc[df_summary['Avg Query Time (ms)'].astype(float).idxmin(), 'Index']
            best_throughput_idx = df_summary.loc[df_summary['Avg Throughput (q/s)'].astype(float).idxmax(), 'Index']
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏃 **Fastest Index:** {best_time_idx}")
            with col2:
                st.success(f"🚀 **Highest Throughput:** {best_throughput_idx}")
        else:
            st.info("No performance data available for summary.")


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