"""
Learned ZM Index using Linear Regression.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import psutil
import os

logger = logging.getLogger(__name__)


class ZMLinearIndex:
    """
    Learned ZM (Z-order/Morton) Index using Linear Regression.
    
    Uses machine learning to predict Z-order positions for spatial queries,
    enabling efficient spatial operations through learned mappings.
    """
    
    def __init__(
        self, 
        degree: int = 1, 
        include_bias: bool = True,
        regularization: str = "l2",
        alpha: float = 1.0,
        normalize_features: bool = True,
        **kwargs
    ):
        """
        Initialize ZM Linear Index.
        
        Args:
            degree: Degree of polynomial features (1 for linear, >1 for polynomial)
            include_bias: Whether to include bias term in regression
            regularization: Regularization type ("none", "l1", "l2", "elastic")
            alpha: Regularization strength (higher = more regularization)
            normalize_features: Whether to normalize input features
            **kwargs: Additional parameters (for compatibility)
        """
        self.degree = degree
        self.include_bias = include_bias
        self.regularization = regularization
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.model: Optional[LinearRegression] = None
        self.poly_features: Optional[PolynomialFeatures] = None
        self.scaler = None
        self.coordinates: Optional[np.ndarray] = None
        self.morton_codes: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None
        self.build_time: Optional[float] = None
        self.memory_usage: Optional[float] = None
        
    def build(self, coordinates: np.ndarray, morton_codes: np.ndarray) -> None:
        """
        Build learned ZM index from coordinates and Morton codes.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
            morton_codes: Array of Morton/Z-order codes, shape (N,)
        """
        logger.info(f"Building ZM Linear Index for {len(coordinates)} points")
        start_time = time.perf_counter()
        
        self.coordinates = coordinates.copy()
        self.morton_codes = morton_codes.copy()
        
        # Sort data by Morton codes for efficient range queries
        self.sorted_indices = np.argsort(morton_codes)
        sorted_morton = morton_codes[self.sorted_indices]
        sorted_coords = coordinates[self.sorted_indices]
        
        # Prepare features (possibly with polynomial terms)
        if self.degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.degree, 
                include_bias=self.include_bias
            )
            X = self.poly_features.fit_transform(sorted_coords)
        else:
            X = sorted_coords
        
        # Train linear regression model to predict Morton code position
        self.model = LinearRegression(fit_intercept=self.include_bias and self.degree == 1)
        
        # Target is the position in sorted Morton order
        y = np.arange(len(sorted_morton))
        self.model.fit(X, y)
        
        self.build_time = time.perf_counter() - start_time
        self._calculate_memory_usage()
        
        logger.info(f"ZM Linear Index built in {self.build_time:.4f} seconds")
        logger.info(f"Memory usage: {self.memory_usage:.2f} MB")
        logger.info(f"Model R² score: {self.model.score(X, y):.4f}")
    
    def _predict_position(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Predict position in sorted Morton order for given coordinates.
        
        Args:
            coordinates: Array of coordinates, shape (N, 2)
            
        Returns:
            Predicted positions as array
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        if self.poly_features is not None:
            X = self.poly_features.transform(coordinates)
        else:
            X = coordinates
        
        positions = self.model.predict(X)
        # Clip to valid range
        return np.clip(positions, 0, len(self.sorted_indices) - 1)
    
    def point_query(self, lat: float, lon: float, tolerance: float = 1e-6) -> List[int]:
        """
        Perform point query using learned index.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            tolerance: Tolerance for approximate matching
            
        Returns:
            List of point indices matching the query
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        query_coords = np.array([[lat, lon]])
        predicted_pos = self._predict_position(query_coords)[0]
        
        logger.info(f"ZM Linear point query: lat={lat:.6f}, lon={lon:.6f}, tolerance={tolerance:.6f}")
        logger.info(f"Predicted position: {predicted_pos:.2f} (out of {len(self.sorted_indices)-1})")
        
        # Much more aggressive search for point queries - tolerance might be large
        # Start with a reasonable window, then expand if needed
        initial_search_radius = max(100, int(len(self.coordinates) * 0.05))  # Start with 5% of data
        
        results = []
        search_radius = initial_search_radius
        
        # Iterative search with expanding window if no results found
        while len(results) == 0 and search_radius <= len(self.coordinates) // 2:
            start_pos = max(0, int(predicted_pos) - search_radius)
            end_pos = min(len(self.sorted_indices), int(predicted_pos) + search_radius + 1)
            
            logger.info(f"Searching range [{start_pos}:{end_pos}] (radius={search_radius})")
            
            # Search for points within tolerance
            candidates_checked = 0
            for i in range(start_pos, end_pos):
                actual_idx = self.sorted_indices[i]
                point_lat, point_lon = self.coordinates[actual_idx]
                distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
                candidates_checked += 1
                
                # Debug: Log first few candidates for detailed analysis
                if candidates_checked <= 5:
                    logger.info(f"Candidate {candidates_checked}: idx={actual_idx}, coords=({point_lat:.6f}, {point_lon:.6f}), distance={distance:.6f}, tolerance={tolerance:.6f}")
                
                if distance <= tolerance:
                    results.append(actual_idx)
                    logger.debug(f"Found match: idx={actual_idx}, coords=({point_lat:.6f}, {point_lon:.6f}), dist={distance:.6f}")
                    
                # Debug: Check for very close matches that might be just outside tolerance
                elif distance <= tolerance * 10:  # Within 10x tolerance
                    logger.info(f"Near miss: idx={actual_idx}, coords=({point_lat:.6f}, {point_lon:.6f}), distance={distance:.6f} (tolerance={tolerance:.6f})")
            
            logger.info(f"Checked {candidates_checked} candidates, found {len(results)} matches")
            
            # If no results found and tolerance is small, try expanding search
            if len(results) == 0 and search_radius < len(self.coordinates) // 4:
                search_radius *= 2
                logger.info(f"No matches found, expanding search radius to {search_radius}")
            else:
                break
        
        logger.info(f"ZM Linear point query final: predicted_pos={predicted_pos:.1f}, search_radius={search_radius}, results={len(results)}")
        return results
    
    def range_query(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> List[int]:
        """
        Perform range query using learned index.
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            
        Returns:
            List of point indices within the range
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        logger.info(f"ZM Linear range query: [{min_lat:.6f}, {max_lat:.6f}] × [{min_lon:.6f}, {max_lon:.6f}]")
        
        # Predict positions for multiple points across the query region
        # Use a grid of points to better estimate the range of Morton positions
        grid_points = []
        for lat_factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for lon_factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
                lat = min_lat + lat_factor * (max_lat - min_lat)
                lon = min_lon + lon_factor * (max_lon - min_lon)
                grid_points.append([lat, lon])
        
        grid_coords = np.array(grid_points)
        predicted_positions = self._predict_position(grid_coords)
        
        # Use wider margins for range queries since spatial ranges can be scattered in Morton order
        min_predicted = np.min(predicted_positions)
        max_predicted = np.max(predicted_positions)
        range_span = max_predicted - min_predicted
        
        # Expand search window significantly for range queries
        margin = max(500, int(len(self.coordinates) * 0.1), int(range_span * 0.5))
        min_pos = max(0, int(min_predicted - margin))
        max_pos = min(len(self.sorted_indices), int(max_predicted + margin + 1))
        
        logger.info(f"ZM Linear range: predicted [{min_predicted:.1f}, {max_predicted:.1f}], searching [{min_pos}:{max_pos}] (margin={margin})")
        
        # Scan predicted range and filter by actual coordinates
        results = []
        candidates_checked = 0
        for i in range(min_pos, max_pos):
            actual_idx = self.sorted_indices[i]
            point_lat, point_lon = self.coordinates[actual_idx]
            candidates_checked += 1
            
            if (min_lat <= point_lat <= max_lat and 
                min_lon <= point_lon <= max_lon):
                results.append(actual_idx)
        
        logger.info(f"ZM Linear range query: checked {candidates_checked} candidates, found {len(results)} matches")
        return results
    
    def knn_query(self, lat: float, lon: float, k: int = 1) -> List[Tuple[int, float]]:
        """
        Perform k-nearest neighbor query using learned index.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of nearest neighbors to find
            
        Returns:
            List of (point_index, distance) tuples for k nearest neighbors
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        query_coords = np.array([[lat, lon]])
        predicted_pos = self._predict_position(query_coords)[0]
        
        # More aggressive search for k-NN to ensure accuracy
        # Start with a reasonable window, then expand if needed
        initial_search_radius = max(k * 10, int(len(self.coordinates) * 0.05))  # Increased from 0.01
        
        candidates = []
        search_radius = initial_search_radius
        
        # Iterative search with expanding window if not enough candidates found
        while len(candidates) < k * 3 and search_radius <= len(self.coordinates) // 2:
            candidates = []
            start_pos = max(0, int(predicted_pos) - search_radius)
            end_pos = min(len(self.sorted_indices), int(predicted_pos) + search_radius + 1)
            
            # Calculate distances for candidates
            for i in range(start_pos, end_pos):
                actual_idx = self.sorted_indices[i]
                point_lat, point_lon = self.coordinates[actual_idx]
                distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
                candidates.append((actual_idx, distance))
            
            # If we don't have enough candidates, expand search
            if len(candidates) < k * 2:
                search_radius *= 2
                logger.debug(f"Expanding k-NN search radius to {search_radius}")
            else:
                break
        
        # If we still don't have enough candidates, fall back to brute force
        if len(candidates) < k:
            logger.warning(f"ZM Linear k-NN: prediction failed, falling back to brute force search")
            candidates = []
            for i, (point_lat, point_lon) in enumerate(self.coordinates):
                distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
                candidates.append((i, distance))
        
        # Sort by distance and return top k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
    
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
    
    def batch_range_queries(
        self, 
        query_boxes: np.ndarray
    ) -> List[List[int]]:
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
        if self.model is None:
            return {"status": "not_built"}
        
        # Calculate model accuracy metrics
        if self.poly_features is not None:
            X = self.poly_features.transform(self.coordinates[self.sorted_indices])
        else:
            X = self.coordinates[self.sorted_indices]
        
        y = np.arange(len(self.sorted_indices))
        r2_score = self.model.score(X, y)
        
        stats = {
            "status": "built",
            "num_points": len(self.coordinates) if self.coordinates is not None else 0,
            "build_time_seconds": self.build_time,
            "memory_usage_mb": self.memory_usage,
            "degree": self.degree,
            "include_bias": self.include_bias,
            "r2_score": r2_score,
            "model_coefficients": self.model.coef_.tolist() if self.model.coef_ is not None else None,
            "model_intercept": float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None,
        }
        
        return stats
    
    def _calculate_memory_usage(self) -> None:
        """Calculate approximate memory usage of the index."""
        try:
            memory_usage = 0.0
            
            # Memory for data storage
            if self.coordinates is not None:
                memory_usage += self.coordinates.nbytes / (1024 * 1024)
            if self.morton_codes is not None:
                memory_usage += self.morton_codes.nbytes / (1024 * 1024)
            if self.sorted_indices is not None:
                memory_usage += self.sorted_indices.nbytes / (1024 * 1024)
            
            # Memory for model (rough estimate)
            if self.model is not None and hasattr(self.model, 'coef_'):
                memory_usage += self.model.coef_.nbytes / (1024 * 1024)
            
            self.memory_usage = memory_usage
            
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {e}")
            self.memory_usage = 0.0
    
    def clear(self) -> None:
        """Clear the index and free memory."""
        self.model = None
        self.poly_features = None
        self.coordinates = None
        self.morton_codes = None
        self.sorted_indices = None
        self.build_time = None
        self.memory_usage = None
        logger.info("ZM Linear Index cleared")