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
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator

logger = logging.getLogger(__name__)


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
            for i, (lat, lon) in enumerate(map_coords):
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
                nearest_lat, nearest_lon = coords[nearest_idx]
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
                st.subheader("Build Results")
                for index_name, result in build_results.items():
                    if result == "Success":
                        st.success(f"✅ {index_name}: {result}")
                    else:
                        st.error(f"❌ {index_name}: {result}")
        
        # Display index statistics if built
        if st.session_state.indexes_built and self.query_engine.list_indexes():
            st.subheader("📊 Index Statistics")
            
            stats = self.query_engine.get_index_statistics()
            
            # Create comparison table
            stats_data = []
            for index_name, index_stats in stats.items():
                stats_data.append({
                    "Index": index_name,
                    "Type": index_stats.get("status", "unknown"),
                    "Build Time (s)": f"{index_stats.get('build_time_seconds', 0):.4f}",
                    "Memory Usage (MB)": f"{index_stats.get('memory_usage_mb', 0):.2f}",
                    "Data Points": index_stats.get('num_points', 0)
                })
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, width='stretch')
    
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
        
        # Add debug information section
        with st.expander("🔍 Debug Information", expanded=True):
            st.write("**Raw Query Results:**")
            for index_name, result in results.items():
                st.write(f"**{index_name}:**")
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.write(f"- Result count: {result['count']}")
                    st.write(f"- Query time: {result['query_time_seconds']:.6f}s")
                    st.write(f"- Index type: {result['index_type']}")
                    if result['results']:
                        if query_type == "k-NN Query" and isinstance(result['results'][0], tuple):
                            st.write(f"- Sample results (with distances): {result['results'][:5]}")
                        else:
                            st.write(f"- Sample result indices: {result['results'][:10]}")
                    st.write("---")
        
        # Performance comparison
        performance_data = []
        for index_name, result in results.items():
            if 'error' not in result:
                performance_data.append({
                    "Index": index_name,
                    "Type": result['index_type'],
                    "Results": result['count'],
                    "Query Time (ms)": f"{result['query_time_seconds'] * 1000:.4f}",
                    "Query Time (s)": f"{result['query_time_seconds']:.6f}"
                })
        
        if performance_data:
            df_perf = pd.DataFrame(performance_data)
            st.dataframe(df_perf, width='stretch')
            
            # Performance chart
            fig = px.bar(
                df_perf, 
                x="Index", 
                y="Query Time (ms)",
                color="Type",
                title=f"{query_type} Performance Comparison"
            )
            st.plotly_chart(fig, width='stretch')
        
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
                    st.write(f"**Results Count:** {result['count']}")
                    st.write(f"**Query Time:** {result['query_time_seconds']:.6f} seconds")
                    
                    if result['results'] and len(result['results']) <= 100:
                        st.write("**Sample Results:**")
                        if query_type == "k-NN Query":
                            # For k-NN, show distances
                            knn_data = [{"Index": idx, "Distance": dist} for idx, dist in result['results'][:20]]
                            st.dataframe(pd.DataFrame(knn_data))
                        else:
                            st.write(result['results'][:20])
    
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
            
            for lat, lon in bg_coords:
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
        query_bounds = self._add_query_visualization(m, query_type)
        
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
        
        # Track which points have been found by which indexes
        point_results = {}  # point_idx -> list of index names
        
        # First pass: collect all results and group by point index
        for index_name, result in results.items():
            if 'error' in result or not result['results']:
                continue
                
            # Add result points
            result_indices = result['results']
            if isinstance(result_indices[0], tuple):  # k-NN results with distances
                for i, (idx, distance) in enumerate(result_indices[:50]):  # Limit to 50 for performance
                    if idx < len(coords):
                        if idx not in point_results:
                            point_results[idx] = []
                        point_results[idx].append((index_name, i, distance))
            else:  # Point/Range query results
                for i, idx in enumerate(result_indices[:100]):  # Limit to 100 for performance
                    if idx < len(coords):
                        if idx not in point_results:
                            point_results[idx] = []
                        point_results[idx].append((index_name, i, None))
        
        # Second pass: add markers with visual separation for overlapping points
        for point_idx, index_list in point_results.items():
            lat, lon = coords[point_idx]
            
            # Sort by index priority (R-Tree first, then others)
            index_priority = {'rtree': 0, 'zm_linear': 1, 'zm_mlp': 2}
            index_list.sort(key=lambda x: index_priority.get(x[0], 99))
            
            for layer_idx, (index_name, result_idx, distance) in enumerate(index_list):
                style = index_styles.get(index_name, index_styles['rtree'])
                
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
                
                # Create marker with visual separation
                adjusted_radius = marker_size + style['radius_offset']
                
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=adjusted_radius,
                    color=style['color'],
                    fillColor=style['fillColor'],
                    fillOpacity=style['fillOpacity'],
                    weight=style['weight'],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=tooltip_text
                ).add_to(m)
        
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
                    st.markdown(f"🔹 **Results:** {result['count']}")
                    st.markdown(f"🔹 **Time:** {result['query_time_seconds']*1000:.2f} ms")
                    
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
        
        if not hasattr(st.session_state, 'last_query_params'):
            st.info("No recent query parameters to analyze.")
            return
        
        coords = st.session_state.coordinates
        params = st.session_state.last_query_params
        
        if query_type == "Point Query":
            query_lat = params.get('lat')
            query_lon = params.get('lon')
            tolerance = params.get('tolerance', 0.001)
            
            st.write(f"**Query Point:** ({query_lat:.6f}, {query_lon:.6f})")
            st.write(f"**Tolerance:** {tolerance:.6f}")
            
            # Find closest points in dataset
            distances = np.sqrt((coords[:, 0] - query_lat)**2 + (coords[:, 1] - query_lon)**2)
            sorted_indices = np.argsort(distances)
            
            st.write("**Closest 10 points in dataset:**")
            closest_data = []
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                dist = distances[idx]
                lat, lon = coords[idx]
                within_tolerance = "✅" if dist <= tolerance else "❌"
                closest_data.append({
                    "Index": idx,
                    "Latitude": f"{lat:.6f}",
                    "Longitude": f"{lon:.6f}",
                    "Distance": f"{dist:.6f}",
                    "Within Tolerance": within_tolerance
                })
            
            st.dataframe(pd.DataFrame(closest_data))
            
            # Check if any point should be found
            points_within_tolerance = np.sum(distances <= tolerance)
            st.write(f"**Expected results:** {points_within_tolerance} points within tolerance")
            
            if points_within_tolerance == 0:
                st.warning("⚠️ **No points exist within the specified tolerance!** Try increasing the tolerance value.")
            else:
                st.info(f"✅ **{points_within_tolerance} points should be found** by all indexes within tolerance {tolerance:.6f}")
        
        elif query_type == "k-NN Query":
            query_lat = params.get('lat')
            query_lon = params.get('lon')
            k = params.get('k', 5)
            
            st.write(f"**Query Point:** ({query_lat:.6f}, {query_lon:.6f})")
            st.write(f"**k:** {k}")
            
            # Find actual k nearest neighbors
            distances = np.sqrt((coords[:, 0] - query_lat)**2 + (coords[:, 1] - query_lon)**2)
            sorted_indices = np.argsort(distances)
            
            st.write(f"**True {k} nearest neighbors:**")
            true_knn_data = []
            for i in range(min(k, len(sorted_indices))):
                idx = sorted_indices[i]
                dist = distances[idx]
                lat, lon = coords[idx]
                true_knn_data.append({
                    "Rank": i+1,
                    "Index": idx,
                    "Latitude": f"{lat:.6f}",
                    "Longitude": f"{lon:.6f}",
                    "Distance": f"{dist:.6f}"
                })
            
            st.dataframe(pd.DataFrame(true_knn_data))
            
            # Compare with actual results from indexes
            st.write("**Comparison with index results:**")
            for index_name, result in results.items():
                if 'error' not in result and result['results']:
                    if isinstance(result['results'][0], tuple):  # k-NN results with distances
                        found_indices = [idx for idx, dist in result['results'][:k]]
                        expected_indices = sorted_indices[:k].tolist()
                        
                        matches = len(set(found_indices) & set(expected_indices))
                        accuracy = matches / k if k > 0 else 1.0
                        
                        if accuracy == 1.0:
                            st.success(f"✅ **{index_name}**: Perfect match ({matches}/{k})")
                        else:
                            st.error(f"❌ **{index_name}**: Only {matches}/{k} correct neighbors (accuracy: {accuracy:.1%})")
                            
                            # Show which ones are wrong
                            missing = set(expected_indices) - set(found_indices)
                            extra = set(found_indices) - set(expected_indices)
                            if missing:
                                st.write(f"  - Missing correct neighbors: {list(missing)}")
                            if extra:
                                st.write(f"  - Incorrect neighbors found: {list(extra)}")
        
        elif query_type == "Range Query":
            min_lat = params.get('min_lat')
            max_lat = params.get('max_lat')
            min_lon = params.get('min_lon')
            max_lon = params.get('max_lon')
            
            st.write(f"**Range:** [{min_lat:.6f}, {max_lat:.6f}] × [{min_lon:.6f}, {max_lon:.6f}]")
            
            # Find points that should be in range
            in_range = ((coords[:, 0] >= min_lat) & (coords[:, 0] <= max_lat) & 
                       (coords[:, 1] >= min_lon) & (coords[:, 1] <= max_lon))
            expected_indices = np.where(in_range)[0]
            
            st.write(f"**Expected results:** {len(expected_indices)} points in range")
            
            # Show sample of expected results
            if len(expected_indices) > 0:
                sample_size = min(10, len(expected_indices))
                sample_indices = expected_indices[:sample_size]
                expected_data = []
                for idx in sample_indices:
                    lat, lon = coords[idx]
                    expected_data.append({
                        "Index": idx,
                        "Latitude": f"{lat:.6f}",
                        "Longitude": f"{lon:.6f}"
                    })
                st.write("**Sample expected results:**")
                st.dataframe(pd.DataFrame(expected_data))
                
                # Compare with actual results
                for index_name, result in results.items():
                    if 'error' not in result:
                        found_indices = set(result['results'])
                        expected_set = set(expected_indices)
                        
                        matches = len(found_indices & expected_set)
                        accuracy = matches / len(expected_set) if len(expected_set) > 0 else 1.0
                        
                        if accuracy == 1.0 and len(found_indices) == len(expected_set):
                            st.success(f"✅ **{index_name}**: Perfect match ({matches}/{len(expected_set)})")
                        else:
                            st.error(f"❌ **{index_name}**: Found {len(found_indices)}, expected {len(expected_set)} (accuracy: {accuracy:.1%})")
            else:
                st.warning("⚠️ No points exist in the specified range!")
    
    def _performance_evaluation_page(self) -> None:
        """Performance evaluation and benchmarking page."""
        st.header("⚡ Performance Evaluation")
        
        if not st.session_state.indexes_built:
            st.warning("Please build indexes first.")
            return
        
        # Initialize evaluator
        self.evaluator = PerformanceEvaluator(self.query_engine)
        
        st.subheader("Benchmark Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            num_point_queries = st.number_input("Point Queries", min_value=10, max_value=10000, value=1000)
            num_range_queries = st.number_input("Range Queries", min_value=10, max_value=5000, value=500)
        with col2:
            num_knn_queries = st.number_input("k-NN Queries", min_value=10, max_value=5000, value=500)
            k_value = st.number_input("k for k-NN", min_value=1, max_value=50, value=5)
        
        if st.button("Run Comprehensive Evaluation"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    results = self.evaluator.comprehensive_evaluation()
                    st.session_state.evaluation_results = results
                    
                    st.success("Evaluation completed!")
                    
                    # Display summary
                    summary = results['summary']
                    st.subheader("📊 Evaluation Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Benchmarks", summary['total_benchmarks'])
                    with col2:
                        st.metric("Successful Benchmarks", summary['successful_benchmarks'])
                    
                    # Performance comparison table
                    if summary['index_comparison']:
                        st.subheader("Index Performance Comparison")
                        
                        comparison_data = []
                        for index_name, data in summary['index_comparison'].items():
                            comparison_data.append({
                                "Index": index_name,
                                "Type": data['index_type'],
                                "Build Time (s)": f"{data['build_time']:.4f}",
                                "Memory (MB)": f"{data['memory_usage_mb']:.2f}",
                                "Avg Query Time (ms)": f"{data['avg_query_time'] * 1000:.4f}",
                                "Throughput (queries/s)": f"{data['avg_throughput']:.2f}"
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, width='stretch')
                        
                        # Performance charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = px.bar(
                                df_comparison,
                                x="Index",
                                y="Avg Query Time (ms)",
                                color="Type",
                                title="Average Query Time Comparison"
                            )
                            st.plotly_chart(fig1, width='stretch')
                        
                        with col2:
                            fig2 = px.bar(
                                df_comparison,
                                x="Index",
                                y="Memory (MB)",
                                color="Type",
                                title="Memory Usage Comparison"
                            )
                            st.plotly_chart(fig2, width='stretch')
                
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
        
        # Display saved results
        if st.session_state.evaluation_results:
            st.subheader("📋 Detailed Report")
            
            report = self.evaluator.get_comparison_report()
            st.text_area("Performance Report", report, height=400)
    
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
        for lat, lon in vis_coords[::max(1, len(vis_coords)//1000)]:  # Subsample for performance
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