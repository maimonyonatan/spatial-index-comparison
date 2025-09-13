"""
R-Tree spatial index implementation.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from rtree import index
import time
import tempfile
import os

logger = logging.getLogger(__name__)


class RTreeIndex:
    """
    R-Tree spatial index implementation using the rtree library.
    
    Provides traditional spatial indexing for comparison with learned indexes.
    """
    
    def __init__(
        self, 
        leaf_capacity: int = 100, 
        near_minimum_overlap_factor: int = 32,
        fill_factor: float = 0.8,
        split_algorithm: str = "rstar",
        **kwargs
    ):
        """
        Initialize R-Tree index.
        
        Args:
            leaf_capacity: Maximum number of entries in a leaf node
            near_minimum_overlap_factor: Near minimum overlap factor for node splitting
            fill_factor: Minimum node occupancy ratio (0.1-1.0)
            split_algorithm: Split algorithm ("linear", "quadratic", "rstar")
            **kwargs: Additional parameters (for compatibility)
        """
        self.leaf_capacity = leaf_capacity
        self.near_minimum_overlap_factor = near_minimum_overlap_factor
        self.fill_factor = fill_factor
        self.split_algorithm = split_algorithm
        self.index: Optional[index.Index] = None
        self.coordinates: Optional[np.ndarray] = None
        self.build_time: Optional[float] = None
        self.memory_usage: Optional[float] = None
        self.temp_dir: Optional[str] = None
        
    def build(self, coordinates: np.ndarray) -> None:
        """
        Build R-Tree index from coordinates.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
        """
        logger.info(f"Building R-Tree Index for {len(coordinates)} points")
        start_time = time.perf_counter()
        
        self.coordinates = coordinates.copy()
        
        # Create temporary directory for index storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure R-Tree properties
        properties = index.Property()
        properties.leaf_capacity = self.leaf_capacity
        properties.near_minimum_overlap_factor = self.near_minimum_overlap_factor
        properties.storage = index.RT_Memory  # Use in-memory storage
        
        # Create R-Tree index
        self.index = index.Index(properties=properties)
        
        # Insert all points into the index
        for i, (lat, lon) in enumerate(coordinates):
            # R-Tree expects (left, bottom, right, top) bounding box
            # For points, left=right and bottom=top
            bbox = (lon, lat, lon, lat)
            self.index.insert(i, bbox)
        
        self.build_time = time.perf_counter() - start_time
        self._calculate_memory_usage()
        
        logger.info(f"R-Tree Index built in {self.build_time:.4f} seconds")
        logger.info(f"Memory usage: {self.memory_usage:.2f} MB")
    
    def point_query(self, lat: float, lon: float, tolerance: float = 1e-6) -> List[int]:
        """
        Perform point query using R-Tree index.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            tolerance: Tolerance for approximate matching
            
        Returns:
            List of point indices matching the query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Create search bounding box with tolerance
        bbox = (
            lon - tolerance,
            lat - tolerance,
            lon + tolerance,
            lat + tolerance
        )
        
        # Query R-Tree for intersecting points
        candidates = list(self.index.intersection(bbox))
        
        # Filter results by exact distance if needed
        results = []
        for idx in candidates:
            point_lat, point_lon = self.coordinates[idx]
            distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
            if distance <= tolerance:
                results.append(idx)
        
        return results
    
    def range_query(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> List[int]:
        """
        Perform range query using R-Tree index.
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            
        Returns:
            List of point indices within the range
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Create query bounding box
        bbox = (min_lon, min_lat, max_lon, max_lat)
        
        # Query R-Tree for intersecting points
        results = list(self.index.intersection(bbox))
        
        return results
    
    def knn_query(self, lat: float, lon: float, k: int = 1) -> List[Tuple[int, float]]:
        """
        Perform k-nearest neighbor query using R-Tree index.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of nearest neighbors to find
            
        Returns:
            List of (point_index, distance) tuples for k nearest neighbors
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # R-Tree's nearest method returns k nearest neighbors
        query_point = (lon, lat)  # R-Tree expects (x, y) format
        
        # Get nearest neighbors from R-Tree
        nearest_ids = list(self.index.nearest(query_point, k))
        
        # Calculate distances and create result tuples
        results = []
        for idx in nearest_ids:
            point_lat, point_lon = self.coordinates[idx]
            distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
            results.append((idx, distance))
        
        # Sort by distance (R-Tree nearest should already be sorted, but ensure it)
        results.sort(key=lambda x: x[1])
        
        return results
    
    def batch_point_queries(
        self, 
        query_points: np.ndarray, 
        tolerance: float = 1e-6
    ) -> List[List[int]]:
        """
        Perform batch point queries for multiple points.
        
        Args:
            query_points: Array of query coordinates, shape (N, 2)
            tolerance: Tolerance for approximate matching
            
        Returns:
            List of result lists, one for each query point
        """
        results = []
        for lat, lon in query_points:
            result = self.point_query(lat, lon, tolerance)
            results.append(result)
        return results
    
    def batch_range_queries(self, query_boxes: np.ndarray) -> List[List[int]]:
        """
        Perform batch range queries for multiple bounding boxes.
        
        Args:
            query_boxes: Array of bounding boxes, shape (N, 4)
                        Each row: [min_lat, max_lat, min_lon, max_lon]
            
        Returns:
            List of result lists, one for each query box
        """
        results = []
        for min_lat, max_lat, min_lon, max_lon in query_boxes:
            result = self.range_query(min_lat, max_lat, min_lon, max_lon)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics and performance metrics.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "not_built"}
        
        # Get R-Tree statistics if available
        stats = {
            "status": "built",
            "num_points": len(self.coordinates) if self.coordinates is not None else 0,
            "build_time_seconds": self.build_time,
            "memory_usage_mb": self.memory_usage,
            "leaf_capacity": self.leaf_capacity,
            "near_minimum_overlap_factor": self.near_minimum_overlap_factor,
        }
        
        # Try to get additional R-Tree specific statistics
        try:
            if hasattr(self.index, 'get_bounds'):
                bounds = self.index.get_bounds()
                stats["bounds"] = bounds
        except Exception as e:
            logger.debug(f"Could not get R-Tree bounds: {e}")
        
        return stats
    
    def _calculate_memory_usage(self) -> None:
        """Calculate approximate memory usage of the index."""
        try:
            memory_usage = 0.0
            
            # Memory for coordinate storage
            if self.coordinates is not None:
                memory_usage += self.coordinates.nbytes / (1024 * 1024)
            
            # Estimate R-Tree memory usage
            # This is a rough approximation since rtree doesn't expose memory stats directly
            if self.coordinates is not None:
                num_points = len(self.coordinates)
                # Rough estimate: each point takes ~100 bytes in R-Tree structure
                estimated_rtree_memory = (num_points * 100) / (1024 * 1024)
                memory_usage += estimated_rtree_memory
            
            self.memory_usage = memory_usage
            
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {e}")
            self.memory_usage = 0.0
    
    def clear(self) -> None:
        """Clear the index and free memory."""
        if self.index is not None:
            self.index.close()
        
        self.index = None
        self.coordinates = None
        self.build_time = None
        self.memory_usage = None
        
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory {self.temp_dir}: {e}")
        
        self.temp_dir = None
        logger.info("R-Tree Index cleared")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.clear()
        except Exception:
            pass  # Ignore errors during cleanup