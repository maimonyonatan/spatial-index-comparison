"""
Learned ZM Index using Multi-Layer Perceptron (MLP).
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import os

logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """Shallow MLP model for spatial coordinate to position mapping."""
    
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input dimension (2 for lat/lon coordinates)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (single output for position prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x).squeeze(-1)


class ZMMLPIndex:
    """
    Learned ZM (Z-order/Morton) Index using Multi-Layer Perceptron.
    
    Uses a shallow neural network to predict Z-order positions for spatial queries,
    providing more complex learned mappings than linear regression.
    """
    
    def __init__(
        self, 
        hidden_dims: List[int] = [64, 32],
        activation: str = "relu",
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        epochs: int = 100,
        optimizer: str = "adam",
        early_stopping: bool = True,
        patience: int = 10,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize ZM MLP Index.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "sigmoid", "leaky_relu")
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            epochs: Number of training epochs
            optimizer: Optimizer type ("adam", "sgd", "rmsprop")
            early_stopping: Whether to use early stopping
            patience: Early stopping patience (epochs to wait)
            device: Device to use ('cpu', 'cuda', or 'auto')
            **kwargs: Additional parameters (for compatibility)
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_type = optimizer
        self.early_stopping = early_stopping
        self.patience = patience if early_stopping else None
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model: Optional[MLPModel] = None
        self.coordinates: Optional[np.ndarray] = None
        self.morton_codes: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None
        self.coord_mean: Optional[np.ndarray] = None
        self.coord_std: Optional[np.ndarray] = None
        self.build_time: Optional[float] = None
        self.memory_usage: Optional[float] = None
        self.training_loss: Optional[List[float]] = None
        
    def build(self, coordinates: np.ndarray, morton_codes: np.ndarray) -> None:
        """
        Build learned ZM index using MLP.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
            morton_codes: Array of Morton/Z-order codes, shape (N,)
        """
        logger.info(f"Building ZM MLP Index for {len(coordinates)} points")
        start_time = time.perf_counter()
        
        self.coordinates = coordinates.copy()
        self.morton_codes = morton_codes.copy()
        
        # Sort data by Morton codes
        self.sorted_indices = np.argsort(morton_codes)
        sorted_coords = coordinates[self.sorted_indices]
        
        # Normalize coordinates for better training
        self.coord_mean = np.mean(sorted_coords, axis=0)
        self.coord_std = np.std(sorted_coords, axis=0)
        self.coord_std = np.where(self.coord_std == 0, 1.0, self.coord_std)  # Avoid division by zero
        
        normalized_coords = (sorted_coords - self.coord_mean) / self.coord_std
        
        # Prepare training data
        X = torch.FloatTensor(normalized_coords).to(self.device)
        y = torch.FloatTensor(np.arange(len(sorted_coords))).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = MLPModel(
            input_dim=2,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.model.train()
        self.training_loss = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_loss.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.model.eval()
        self.build_time = time.perf_counter() - start_time
        self._calculate_memory_usage()
        
        logger.info(f"ZM MLP Index built in {self.build_time:.4f} seconds")
        logger.info(f"Memory usage: {self.memory_usage:.2f} MB")
        logger.info(f"Final training loss: {self.training_loss[-1]:.6f}")
    
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
        
        # Normalize coordinates
        normalized_coords = (coordinates - self.coord_mean) / self.coord_std
        
        # Predict using model
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(normalized_coords).to(self.device)
            positions = self.model(X).cpu().numpy()
        
        # Clip to valid range
        return np.clip(positions, 0, len(self.sorted_indices) - 1)
    
    def point_query(self, lat: float, lon: float, tolerance: float = 1e-6) -> List[int]:
        """
        Perform point query using learned MLP index.
        
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
        
        logger.info(f"ZM MLP point query: lat={lat:.6f}, lon={lon:.6f}, tolerance={tolerance:.6f}")
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
            
            logger.info(f"MLP searching range [{start_pos}:{end_pos}] (radius={search_radius})")
            
            # Search for points within tolerance
            candidates_checked = 0
            for i in range(start_pos, end_pos):
                actual_idx = self.sorted_indices[i]
                point_lat, point_lon = self.coordinates[actual_idx]
                distance = np.sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
                candidates_checked += 1
                if distance <= tolerance:
                    results.append(actual_idx)
                    logger.debug(f"MLP found match: idx={actual_idx}, coords=({point_lat:.6f}, {point_lon:.6f}), dist={distance:.6f}")
            
            logger.info(f"MLP checked {candidates_checked} candidates, found {len(results)} matches")
            
            # If no results found and tolerance is small, try expanding search
            if len(results) == 0 and search_radius < len(self.coordinates) // 4:
                search_radius *= 2
                logger.info(f"MLP no matches found, expanding search radius to {search_radius}")
            else:
                break
        
        logger.info(f"ZM MLP point query final: predicted_pos={predicted_pos:.1f}, search_radius={search_radius}, results={len(results)}")
        return results
    
    def range_query(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> List[int]:
        """
        Perform range query using learned MLP index.
        
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
        
        logger.info(f"ZM MLP range query: [{min_lat:.6f}, {max_lat:.6f}] × [{min_lon:.6f}, {max_lon:.6f}]")
        
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
        
        logger.info(f"ZM MLP range: predicted [{min_predicted:.1f}, {max_predicted:.1f}], searching [{min_pos}:{max_pos}] (margin={margin})")
        
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
        
        logger.info(f"ZM MLP range query: checked {candidates_checked} candidates, found {len(results)} matches")
        return results
    
    def knn_query(self, lat: float, lon: float, k: int = 1) -> List[Tuple[int, float]]:
        """
        Perform k-nearest neighbor query using learned MLP index.
        
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
                logger.debug(f"Expanding k-NN search radius to {search_radius} for MLP")
            else:
                break
        
        # If we still don't have enough candidates, fall back to brute force
        if len(candidates) < k:
            logger.warning(f"ZM MLP k-NN: prediction failed, falling back to brute force search")
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
        """Perform batch point queries for multiple points."""
        results = []
        for lat, lon in query_points:
            result = self.point_query(lat, lon, tolerance)
            results.append(result)
        return results
    
    def batch_range_queries(self, query_boxes: np.ndarray) -> List[List[int]]:
        """Perform batch range queries for multiple bounding boxes."""
        results = []
        for min_lat, max_lat, min_lon, max_lon in query_boxes:
            result = self.range_query(min_lat, max_lat, min_lon, max_lon)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics and performance metrics."""
        if self.model is None:
            return {"status": "not_built"}
        
        stats = {
            "status": "built",
            "num_points": len(self.coordinates) if self.coordinates is not None else 0,
            "build_time_seconds": self.build_time,
            "memory_usage_mb": self.memory_usage,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": str(self.device),
            "final_training_loss": self.training_loss[-1] if self.training_loss else None,
            "num_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
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
            
            # Memory for model parameters
            if self.model is not None:
                param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
                memory_usage += param_memory / (1024 * 1024)
            
            self.memory_usage = memory_usage
            
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {e}")
            self.memory_usage = 0.0
    
    def clear(self) -> None:
        """Clear the index and free memory."""
        if self.model is not None:
            del self.model
        self.model = None
        self.coordinates = None
        self.morton_codes = None
        self.sorted_indices = None
        self.coord_mean = None
        self.coord_std = None
        self.build_time = None
        self.memory_usage = None
        self.training_loss = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ZM MLP Index cleared")