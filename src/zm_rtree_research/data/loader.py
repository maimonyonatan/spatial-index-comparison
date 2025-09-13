"""
Data loading and preprocessing utilities for spatial datasets.
"""

import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and preprocesses spatial datasets for R-Tree vs ZM Index comparison.
    
    Supports loading CSV files with latitude/longitude coordinates, geographic
    subsetting, coordinate normalization, and Morton/Z-address encoding.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Optional path to default data directory
        """
        self.data_path = data_path or Path("data")
        self.dataset: Optional[pd.DataFrame] = None
        self.normalized_coords: Optional[np.ndarray] = None
        self.bounds: Optional[Dict[str, float]] = None
        
    def load_csv(
        self, 
        filepath: Path,
        lat_col: str = "Start_Lat",
        lon_col: str = "Start_Lng",
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load spatial data from CSV file.
        
        Args:
            filepath: Path to CSV file
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            sample_size: Optional number of rows to sample
            random_state: Random seed for sampling
            
        Returns:
            Loaded DataFrame with spatial coordinates
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # Validate required columns
            if lat_col not in df.columns or lon_col not in df.columns:
                raise ValueError(f"Required columns {lat_col}, {lon_col} not found in dataset")
            
            # Convert coordinate columns to numeric, replacing invalid values with NaN
            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
            
            # Remove invalid coordinates
            initial_count = len(df)
            df = df.dropna(subset=[lat_col, lon_col])
            df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
            df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]
            
            valid_count = len(df)
            if valid_count < initial_count:
                logger.warning(f"Removed {initial_count - valid_count} rows with invalid coordinates")
            
            # Sample data if requested
            if sample_size is not None and int(sample_size) > 0 and int(sample_size) < len(df):
                df = df.sample(n=int(sample_size), random_state=random_state)
                logger.info(f"Sampled {sample_size} rows from dataset")
            
            self.dataset = df
            logger.info(f"Loaded {len(df)} spatial records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_us_accidents(
        self, 
        filepath: Optional[Path] = None,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load US Accidents dataset from Kaggle.
        
        Args:
            filepath: Optional path to dataset file
            sample_size: Optional number of rows to sample
            
        Returns:
            Loaded US Accidents DataFrame
        """
        if filepath is None:
            filepath = self.data_path / "US_Accidents_March23.csv"
        
        return self.load_csv(
            filepath=filepath,
            lat_col="Start_Lat",
            lon_col="Start_Lng", 
            sample_size=sample_size
        )
    
    def geographic_subset(
        self,
        state: Optional[str] = None,
        city: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> pd.DataFrame:
        """
        Create geographic subset of the dataset.
        
        Args:
            state: Filter by state name
            city: Filter by city name
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Filtered DataFrame
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_csv() first.")
        
        df = self.dataset.copy()
        
        if state and "State" in df.columns:
            df = df[df["State"] == state]
            logger.info(f"Filtered to state: {state} ({len(df)} records)")
        
        if city and "City" in df.columns:
            df = df[df["City"] == city]
            logger.info(f"Filtered to city: {city} ({len(df)} records)")
        
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            lat_col = "Start_Lat" if "Start_Lat" in df.columns else "lat"
            lon_col = "Start_Lng" if "Start_Lng" in df.columns else "lon"
            
            # Ensure coordinate columns are numeric
            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
            
            # Remove rows with NaN coordinates before filtering
            df = df.dropna(subset=[lat_col, lon_col])
            
            df = df[
                (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
                (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
            ]
            logger.info(f"Filtered to bounding box: {bbox} ({len(df)} records)")
        
        return df
    
    def normalize_coordinates(
        self,
        df: Optional[pd.DataFrame] = None,
        lat_col: str = "Start_Lat",
        lon_col: str = "Start_Lng"
    ) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range.
        
        Args:
            df: DataFrame to normalize (uses self.dataset if None)
            lat_col: Latitude column name
            lon_col: Longitude column name
            
        Returns:
            Normalized coordinates as (N, 2) array
        """
        if df is None:
            df = self.dataset
        
        if df is None:
            raise ValueError("No dataset provided")
        
        # Check if dataset is empty
        if len(df) == 0:
            raise ValueError("Dataset is empty - no valid coordinates found")
        
        # Validate coordinate columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Required columns {lat_col}, {lon_col} not found in dataset")
        
        coords = df[[lat_col, lon_col]].values
        
        # Check if coordinates array is empty
        if coords.size == 0:
            raise ValueError("No coordinate data available for normalization")
        
        # Calculate bounds
        min_lat, min_lon = coords.min(axis=0)
        max_lat, max_lon = coords.max(axis=0)
        
        # Check for valid coordinate ranges
        if min_lat == max_lat or min_lon == max_lon:
            logger.warning("Coordinates have zero range - adding small epsilon for normalization")
            # Add small epsilon to avoid division by zero
            if min_lat == max_lat:
                max_lat += 1e-6
            if min_lon == max_lon:
                max_lon += 1e-6
        
        self.bounds = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
        
        # Normalize to [0, 1]
        normalized = np.zeros_like(coords)
        normalized[:, 0] = (coords[:, 0] - min_lat) / (max_lat - min_lat)
        normalized[:, 1] = (coords[:, 1] - min_lon) / (max_lon - min_lon)
        
        self.normalized_coords = normalized
        logger.info(f"Normalized {len(normalized)} coordinate pairs")
        
        return normalized
    
    def compute_morton_codes(
        self,
        coords: Optional[np.ndarray] = None,
        precision: int = 16
    ) -> np.ndarray:
        """
        Compute Morton/Z-order codes for coordinates.
        
        Args:
            coords: Normalized coordinates (uses self.normalized_coords if None)
            precision: Number of bits per dimension
            
        Returns:
            Morton codes as 1D array
        """
        if coords is None:
            coords = self.normalized_coords
        
        if coords is None:
            raise ValueError("No normalized coordinates available")
        
        # Scale coordinates to integer range
        scale = (1 << precision) - 1
        x = (coords[:, 1] * scale).astype(np.uint64)  # longitude
        y = (coords[:, 0] * scale).astype(np.uint64)  # latitude
        
        # Interleave bits to create Morton codes
        morton_codes = np.zeros(len(coords), dtype=np.uint64)
        
        for i in range(precision):
            morton_codes |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        
        logger.info(f"Computed Morton codes for {len(morton_codes)} points")
        return morton_codes
    
    def prepare_dataset(
        self,
        filepath: Path,
        sample_size: Optional[int] = None,
        geographic_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            filepath: Path to dataset file
            sample_size: Optional number of rows to sample
            geographic_filter: Optional geographic filtering parameters
            
        Returns:
            Tuple of (original_df, normalized_coords, morton_codes)
        """
        # Load data
        df = self.load_csv(filepath, sample_size=sample_size)
        
        # Apply geographic filtering
        if geographic_filter:
            df = self.geographic_subset(**geographic_filter)
        
        # Normalize coordinates
        normalized = self.normalize_coordinates(df)
        
        # Compute Morton codes
        morton_codes = self.compute_morton_codes(normalized)
        
        return df, normalized, morton_codes
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            return {}
        
        stats = {
            "total_records": len(self.dataset),
            "bounds": self.bounds,
            "memory_usage_mb": self.dataset.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return stats