"""
Learned ZM Index using Linear Regression.
Implements the learned Z-order Model (ZM) index from Wang et al. 2019 paper.
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
    
    Implements the learned index approach from Wang et al. 2019 paper:
    "Learned Index for Spatial Queries"
    
    Uses Z-order curve mapping and machine learning to predict positions 
    for efficient spatial query processing.
    """
    
    def __init__(
        self, 
        degree: int = 1, 
        include_bias: bool = True,
        regularization: str = "l2",
        alpha: float = 1.0,
        normalize_features: bool = True,
        precision_bits: int = 16,
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
            precision_bits: Number of bits for Z-address computation
            **kwargs: Additional parameters (for compatibility)
        """
        self.degree = degree
        self.include_bias = include_bias
        self.regularization = regularization
        self.alpha = alpha
        self.normalize_features = normalize_features
        self.precision_bits = precision_bits
        self.model: Optional[LinearRegression] = None
        self.poly_features: Optional[PolynomialFeatures] = None
        self.scaler = None
        self.coordinates: Optional[np.ndarray] = None
        self.morton_codes: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None
        self.sorted_z_addresses: Optional[np.ndarray] = None
        self.build_time: Optional[float] = None
        self.memory_usage: Optional[float] = None
        self.min_error: float = 0.0  # worst over-prediction
        self.max_error: float = 0.0  # worst under-prediction
        
        # Coordinate bounds for Z-address computation
        self.min_coords: Optional[np.ndarray] = None
        self.max_coords: Optional[np.ndarray] = None
        self.coord_scale: Optional[np.ndarray] = None
        
    def _compute_z_address(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute Z-addresses (Morton codes) using bit interleaving.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
            
        Returns:
            Array of Z-addresses
        """
        if self.min_coords is None or self.max_coords is None:
            raise ValueError("Coordinate bounds not set. Call build() first.")
        
        # Normalize coordinates to [0, 2^precision_bits - 1]
        normalized = (coordinates - self.min_coords) / self.coord_scale
        quantized = np.clip(normalized * (2**self.precision_bits - 1), 0, 2**self.precision_bits - 1).astype(np.uint32)
        
        # Compute Z-addresses using bit interleaving
        z_addresses = np.zeros(len(coordinates), dtype=np.uint64)
        
        for i in range(len(coordinates)):
            x, y = quantized[i, 0], quantized[i, 1]
            z = 0
            for bit in range(self.precision_bits):
                z |= (x & (1 << bit)) << bit | (y & (1 << bit)) << (bit + 1)
            z_addresses[i] = z
            
        return z_addresses
        
    def build(self, coordinates: np.ndarray, morton_codes: np.ndarray) -> None:
        """
        Build learned ZM index from coordinates and Morton codes.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
            morton_codes: Array of Morton/Z-order codes, shape (N,) (for compatibility, will be recomputed)
        """
        logger.info(f"Building ZM Linear Index for {len(coordinates)} points")
        start_time = time.perf_counter()
        
        self.coordinates = coordinates.copy()
        
        # Compute coordinate bounds for Z-address computation
        self.min_coords = np.min(coordinates, axis=0)
        self.max_coords = np.max(coordinates, axis=0)
        self.coord_scale = self.max_coords - self.min_coords
        # Avoid division by zero
        self.coord_scale = np.where(self.coord_scale == 0, 1.0, self.coord_scale)
        
        # Compute Z-addresses using proper bit interleaving
        z_addresses = self._compute_z_address(coordinates)
        
        # Sort data by Z-addresses for efficient range queries
        self.sorted_indices = np.argsort(z_addresses)
        self.sorted_z_addresses = z_addresses[self.sorted_indices]
        
        # Prepare features for learning
        # Input: Z-addresses, Output: positions in sorted order
        X = self.sorted_z_addresses.reshape(-1, 1).astype(np.float64)
        
        if self.degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.degree, 
                include_bias=self.include_bias
            )
            X = self.poly_features.fit_transform(X)
        
        # Target is the position in sorted Z-address order
        y = np.arange(len(self.sorted_z_addresses)).astype(np.float64)
        
        # Train linear regression model: Z-address -> position
        self.model = LinearRegression(fit_intercept=self.include_bias and self.degree == 1)
        self.model.fit(X, y)
        
        # Calculate prediction errors for query bounds
        predictions = self.model.predict(X)
        errors = predictions - y
        self.min_error = np.min(errors)  # worst over-prediction (negative)
        self.max_error = np.max(errors)  # worst under-prediction (positive)
        
        self.build_time = time.perf_counter() - start_time
        self._calculate_memory_usage()
        
        logger.info(f"ZM Linear Index built in {self.build_time:.4f} seconds")
        logger.info(f"Memory usage: {self.memory_usage:.2f} MB")
        logger.info(f"Model R² score: {self.model.score(X, y):.4f}")
        logger.info(f"Prediction error range: [{self.min_error:.2f}, {self.max_error:.2f}]")
    
    def _predict_position(self, z_addresses: np.ndarray) -> np.ndarray:
        """
        Predict position in sorted Z-address order for given Z-addresses.
        
        Args:
            z_addresses: Array of Z-addresses
            
        Returns:
            Predicted positions as array
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        X = z_addresses.reshape(-1, 1).astype(np.float64)
        
        if self.poly_features is not None:
            X = self.poly_features.transform(X)
        
        positions = self.model.predict(X)
        # Clip to valid range
        return np.clip(positions, 0, len(self.sorted_indices) - 1)
    
    def _model_biased_search(self, predicted_pos: float, target_z_address: int) -> int:
        """
        Perform Model Biased Search (MBS) as described in the paper.
        
        Args:
            predicted_pos: Predicted position from the model
            target_z_address: Target Z-address to find
            
        Returns:
            Actual position of the target Z-address, or -1 if not found
        """
        # Calculate search bounds using min/max error
        min_pos = max(0, int(predicted_pos + self.min_error))
        max_pos = min(len(self.sorted_z_addresses) - 1, int(predicted_pos + self.max_error))
        
        # Binary search with model bias (start from predicted position)
        left, right = min_pos, max_pos
        start_pos = max(min_pos, min(max_pos, int(predicted_pos)))
        
        # Check predicted position first
        if self.sorted_z_addresses[start_pos] == target_z_address:
            return start_pos
            
        # Search left and right alternately
        for radius in range(1, max_pos - min_pos + 1):
            # Check right
            pos = start_pos + radius
            if pos <= max_pos and self.sorted_z_addresses[pos] == target_z_address:
                return pos
                
            # Check left  
            pos = start_pos - radius
            if pos >= min_pos and self.sorted_z_addresses[pos] == target_z_address:
                return pos
                
        return -1  # Not found
    
    def point_query(self, lat: float, lon: float, tolerance: float = 1e-6) -> List[int]:
        """
        Perform point query using learned ZM index.
        
        Args:
            lat: Query latitude
            lon: Query longitude
            tolerance: Tolerance for approximate matching
            
        Returns:
            List of point indices matching the query
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Compute Z-address for query point
        query_coords = np.array([[lat, lon]])
        query_z_address = self._compute_z_address(query_coords)[0]
        
        # Predict position using learned model
        predicted_pos = self._predict_position(np.array([query_z_address]))[0]
        
        logger.info(f"ZM Linear point query: lat={lat:.6f}, lon={lon:.6f}, tolerance={tolerance:.6f}")
        logger.info(f"Query Z-address: {query_z_address}, Predicted position: {predicted_pos:.2f}")
        
        results = []
        
        # For point queries with tolerance, we need to search spatially, not just for exact Z-address matches
        # Calculate search radius based on tolerance and error bounds
        
        # Base search radius: account for model prediction errors and spatial tolerance
        # Convert spatial tolerance to position search radius (rough estimate)
        coordinate_range = np.max(self.coord_scale)  # max coordinate range
        spatial_to_position_factor = len(self.coordinates) / (coordinate_range ** 2)  # rough conversion
        tolerance_radius = int(tolerance * np.sqrt(spatial_to_position_factor)) + 1
        
        # Account for model prediction errors
        error_margin = int(self.max_error - self.min_error) + 1
        search_radius = max(tolerance_radius, error_margin, 100)
        
        # Also expand search for larger tolerances
        if tolerance > 1e-4:  # For large tolerances, search more aggressively
            search_radius = max(search_radius, int(len(self.coordinates) * 0.05))
        
        logger.info(f"Using search radius: {search_radius} (tolerance_radius={tolerance_radius}, error_margin={error_margin})")
        
        # Search around predicted position
        min_pos = max(0, int(predicted_pos) - search_radius)
        max_pos = min(len(self.sorted_indices), int(predicted_pos) + search_radius + 1)
        
        logger.info(f"Searching range [{min_pos}:{max_pos}]")
        
        candidates_checked = 0
        for i in range(min_pos, max_pos):
            actual_idx = self.sorted_indices[i]
            point_lat, point_lon = self.coordinates[actual_idx][0], self.coordinates[actual_idx][1]  # Safe extraction
            distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
            candidates_checked += 1
            
            if distance <= tolerance:
                results.append(actual_idx)
                logger.debug(f"Found match: idx={actual_idx}, coords=({point_lat:.6f}, {point_lon:.6f}), distance={distance:.6f}")
        
        logger.info(f"ZM Linear point query: checked {candidates_checked} candidates, found {len(results)} matches")
        return results
    
    def range_query(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> List[int]:
        """
        Perform range query using learned ZM index following the paper's approach.
        
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
        
        # Compute Z-addresses for query region corners (following paper's approach)
        p_start = np.array([[min_lat, min_lon]])  # bottom-left
        p_end = np.array([[max_lat, max_lon]])    # top-right
        
        z_start = self._compute_z_address(p_start)[0]
        z_end = self._compute_z_address(p_end)[0]
        
        # Predict positions for start and end Z-addresses
        pos_start = self._predict_position(np.array([z_start]))[0]
        pos_end = self._predict_position(np.array([z_end]))[0]
        
        # Calculate search bounds with error margins
        min_pos = max(0, int(min(pos_start, pos_end) + self.min_error))
        max_pos = min(len(self.sorted_indices), int(max(pos_start, pos_end) + self.max_error + 1))
        
        logger.info(f"Z-addresses: start={z_start}, end={z_end}")
        logger.info(f"Predicted positions: start={pos_start:.1f}, end={pos_end:.1f}")
        logger.info(f"Search range: [{min_pos}:{max_pos}]")
        
        # Scan predicted range and filter by actual coordinates
        results = []
        candidates_checked = 0
        
        for i in range(min_pos, max_pos):
            actual_idx = self.sorted_indices[i]
            point_lat, point_lon = self.coordinates[actual_idx][0], self.coordinates[actual_idx][1]  # Safe extraction
            candidates_checked += 1
            
            # Check if point is within query rectangle
            if (min_lat <= point_lat <= max_lat and 
                min_lon <= point_lon <= max_lon):
                results.append(actual_idx)
        
        logger.info(f"ZM Linear range query: checked {candidates_checked} candidates, found {len(results)} matches")
        return results

    def knn_query(self, lat: float, lon: float, k: int = 1) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors using learned ZM index.
        
        Args:
            lat: Query latitude
            lon: Query longitude  
            k: Number of nearest neighbors
            
        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        if self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Compute Z-address for query point
        query_coords = np.array([[lat, lon]])
        query_z_address = self._compute_z_address(query_coords)[0]
        
        # Predict position
        predicted_pos = self._predict_position(np.array([query_z_address]))[0]
        
        # Search around predicted position for k nearest neighbors
        search_radius = max(k * 2, 100)
        candidates = []
        
        while len(candidates) < k * 5 and search_radius < len(self.coordinates) // 2:
            min_pos = max(0, int(predicted_pos) - search_radius)
            max_pos = min(len(self.sorted_indices), int(predicted_pos) + search_radius + 1)
            
            for i in range(min_pos, max_pos):
                actual_idx = self.sorted_indices[i]
                point_lat, point_lon = self.coordinates[actual_idx][0], self.coordinates[actual_idx][1]  # Safe extraction
                distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
                candidates.append((actual_idx, distance))
            
            if len(candidates) < k * 5:
                search_radius *= 2
            else:
                break
        
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
        for coord in query_points:
            lat, lon = coord[0], coord[1]  # Safely extract first two values
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
        for box in query_boxes:
            min_lat, max_lat, min_lon, max_lon = box[0], box[1], box[2], box[3]  # Safely extract first four values
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
        X = self.sorted_z_addresses.reshape(-1, 1).astype(np.float64)
        if self.poly_features is not None:
            X = self.poly_features.transform(X)
        
        y = np.arange(len(self.sorted_z_addresses))
        r2_score = self.model.score(X, y)
        
        stats = {
            "status": "built",
            "num_points": len(self.coordinates) if self.coordinates is not None else 0,
            "build_time_seconds": self.build_time,
            "memory_usage_mb": self.memory_usage,
            "degree": self.degree,
            "include_bias": self.include_bias,
            "precision_bits": self.precision_bits,
            "r2_score": r2_score,
            "min_error": self.min_error,
            "max_error": self.max_error,
            "error_range": self.max_error - self.min_error,
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
            if self.sorted_indices is not None:
                memory_usage += self.sorted_indices.nbytes / (1024 * 1024)
            if self.sorted_z_addresses is not None:
                memory_usage += self.sorted_z_addresses.nbytes / (1024 * 1024)
            
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
        self.sorted_z_addresses = None
        self.min_coords = None
        self.max_coords = None
        self.coord_scale = None
        self.build_time = None
        self.memory_usage = None
        logger.info("ZM Linear Index cleared")