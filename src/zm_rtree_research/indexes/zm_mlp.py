"""
Learned ZM Index using Multi-Layer Perceptron (MLP).
Implements the learned Z-order Model (ZM) index from Wang et al. 2019 paper.
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
    """Multi-stage MLP model for Z-address to position mapping following the paper."""
    
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64, 32], dropout: float = 0.1, num_stages: int = 2):
        """
        Initialize multi-staged MLP model as described in the paper.
        
        Args:
            input_dim: Input dimension (1 for Z-addresses)
            hidden_dims: List of hidden layer dimensions per stage
            dropout: Dropout rate for regularization
            num_stages: Number of model stages (paper uses 2)
        """
        super(MLPModel, self).__init__()
        
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        
        # Create multi-staged architecture
        for stage in range(num_stages):
            stage_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                stage_layers.append(nn.Linear(prev_dim, hidden_dim))
                stage_layers.append(nn.ReLU())  # Paper uses ReLU
                stage_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            # Output layer for this stage
            stage_layers.append(nn.Linear(prev_dim, 1))
            
            self.stages.append(nn.Sequential(*stage_layers))
    
    def forward(self, x: torch.Tensor, stage: int = None) -> torch.Tensor:
        """
        Forward pass through the multi-staged network.
        
        Args:
            x: Input tensor (Z-addresses)
            stage: Specific stage to use (None for final stage)
        """
        if stage is None:
            stage = self.num_stages - 1
        
        return self.stages[stage](x).squeeze(-1)


class ZMMLPIndex:
    """
    Learned ZM (Z-order/Morton) Index using Multi-Layer Perceptron.
    
    Implements the learned index approach from Wang et al. 2019 paper:
    "Learned Index for Spatial Queries"
    
    Uses multi-staged neural networks to predict Z-order positions 
    for efficient spatial query processing.
    """
    
    def __init__(
        self, 
        hidden_dims: List[int] = [128, 64, 32],  # Larger network for better learning
        activation: str = "relu",
        dropout: float = 0.1,  # Reduced dropout to prevent underfitting
        learning_rate: float = 0.001,
        batch_size: int = 256,  # Larger batch size for stable training
        epochs: int = 200,  # More epochs for better convergence
        optimizer: str = "adam",
        early_stopping: bool = True,
        patience: int = 20,  # More patience for convergence
        device: str = "auto",
        num_stages: int = 1,  # Single stage for simpler learning
        precision_bits: int = 16,
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
            num_stages: Number of model stages (paper recommends 2)
            precision_bits: Number of bits for Z-address computation
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
        self.num_stages = num_stages
        self.precision_bits = precision_bits
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model: Optional[MLPModel] = None
        self.coordinates: Optional[np.ndarray] = None
        self.morton_codes: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None
        self.sorted_z_addresses: Optional[np.ndarray] = None
        self.build_time: Optional[float] = None
        self.memory_usage: Optional[float] = None
        self.training_loss: Optional[List[float]] = None
        self.min_error: float = 0.0  # worst over-prediction
        self.max_error: float = 0.0  # worst under-prediction
        self.r2_score: float = 0.0  # R² score for model performance
        
        # Coordinate bounds for Z-address computation
        self.min_coords: Optional[np.ndarray] = None
        self.max_coords: Optional[np.ndarray] = None
        self.coord_scale: Optional[np.ndarray] = None
        
        # Normalization parameters for neural network training
        self.z_min: float = 0.0
        self.z_range: float = 1.0
        self.pos_scale: float = 1.0
    
    def _compute_z_address(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute Z-addresses (Morton codes) using bit interleaving as described in the paper.
        
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
        Build learned ZM index using multi-staged MLP following the paper's approach.
        
        Args:
            coordinates: Array of (lat, lon) coordinates, shape (N, 2)
            morton_codes: Array of Morton/Z-order codes, shape (N,) (for compatibility, will be recomputed)
        """
        logger.info(f"Building ZM MLP Index for {len(coordinates)} points with {self.num_stages} stages")
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
        
        # Sort data by Z-addresses
        self.sorted_indices = np.argsort(z_addresses)
        self.sorted_z_addresses = z_addresses[self.sorted_indices]
        
        # CRITICAL FIX: Normalize input features for better training
        z_min = float(np.min(self.sorted_z_addresses))
        z_max = float(np.max(self.sorted_z_addresses))
        z_range = z_max - z_min if z_max > z_min else 1.0
        
        # Normalize Z-addresses to [0, 1] range for better neural network training
        normalized_z = (self.sorted_z_addresses - z_min) / z_range
        
        # Prepare training data: normalized Z-addresses -> normalized positions
        positions = np.arange(len(self.sorted_z_addresses), dtype=np.float32)
        normalized_positions = positions / (len(positions) - 1) if len(positions) > 1 else positions
        
        X = torch.FloatTensor(normalized_z.reshape(-1, 1)).to(self.device)
        y = torch.FloatTensor(normalized_positions).to(self.device)
        
        # Store normalization parameters
        self.z_min = z_min
        self.z_range = z_range
        self.pos_scale = len(positions) - 1 if len(positions) > 1 else 1
        
        # Create data loader with shuffling for better training
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize improved model architecture
        self.model = MLPModel(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            num_stages=self.num_stages
        ).to(self.device)
        
        # IMPROVED TRAINING SETUP
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)  # Removed verbose parameter
        
        # Training with early stopping
        self.model.train()
        self.training_loss = []
        best_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting MLP training: {self.epochs} epochs, batch_size={self.batch_size}")
        logger.info(f"Data normalization: Z-addresses [{z_min:.0f}, {z_max:.0f}] -> [0, 1]")
        logger.info(f"Position normalization: [0, {len(positions)-1}] -> [0, 1]")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_loss.append(avg_loss)
            scheduler.step(avg_loss)
            
            # Early stopping
            if self.early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                        break
            
            if epoch % 20 == 0 or epoch < 5:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        # Calculate prediction errors and R² score for MBS bounds
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
            y_np = y.cpu().numpy()
            
            # Denormalize predictions for error calculation
            denorm_predictions = predictions * self.pos_scale
            denorm_targets = y_np * self.pos_scale
            
            errors = denorm_predictions - denorm_targets
            self.min_error = float(np.min(errors))  # worst over-prediction (negative)
            self.max_error = float(np.max(errors))  # worst under-prediction (positive)
            
            # Calculate R² score on denormalized data
            ss_res = np.sum((denorm_targets - denorm_predictions) ** 2)
            ss_tot = np.sum((denorm_targets - np.mean(denorm_targets)) ** 2)
            self.r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        self.build_time = time.perf_counter() - start_time
        self._calculate_memory_usage()
        
        logger.info(f"ZM MLP Index built in {self.build_time:.4f} seconds")
        logger.info(f"Memory usage: {self.memory_usage:.2f} MB")
        logger.info(f"Final training loss: {self.training_loss[-1]:.6f}")
        logger.info(f"Prediction error range: [{self.min_error:.2f}, {self.max_error:.2f}]")
        logger.info(f"R² score: {self.r2_score:.6f}")
        
        if self.r2_score < 0.5:
            logger.warning("Low R² score detected. Consider increasing hidden layer sizes or training epochs.")
    
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
        
        # Normalize input using stored parameters
        normalized_z = (z_addresses - self.z_min) / self.z_range
        normalized_z = np.clip(normalized_z, 0.0, 1.0)  # Ensure valid range
        
        # Predict using model
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(normalized_z.reshape(-1, 1)).to(self.device)
            normalized_positions = self.model(X).cpu().numpy()
        
        # Denormalize predictions
        positions = normalized_positions * self.pos_scale
        
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
        Perform point query using learned ZM index following the paper's approach.
        
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
        
        logger.info(f"ZM MLP point query: lat={lat:.6f}, lon={lon:.6f}, tolerance={tolerance:.6f}")
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
        error_margin = int(abs(self.max_error - self.min_error)) + 1
        
        # Use the larger of the two radii
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
        
        logger.info(f"ZM MLP point query: checked {candidates_checked} candidates, found {len(results)} matches")
        return results
    
    def range_query(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> List[int]:
        """
        Perform range query using learned ZM index following Algorithm 1 from the paper.
        
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
        
        # Following Algorithm 1 from the paper:
        # Compute Z-addresses for query region corners
        p_start = np.array([[min_lat, min_lon]])  # bottom-left
        p_end = np.array([[max_lat, max_lon]])    # top-right
        
        z_start = self._compute_z_address(p_start)[0]
        z_end = self._compute_z_address(p_end)[0]
        
        # Predict positions for start and end Z-addresses
        pos_start = self._predict_position(np.array([z_start]))[0]
        pos_end = self._predict_position(np.array([z_end]))[0]
        
        # Calculate search bounds with error margins (as per paper's guarantee)
        min_pos = max(0, int(min(pos_start, pos_end) + self.min_error))
        max_pos = min(len(self.sorted_indices), int(max(pos_start, pos_end) + self.max_error + 1))
        
        logger.info(f"Z-addresses: start={z_start}, end={z_end}")
        logger.info(f"Predicted positions: start={pos_start:.1f}, end={pos_end:.1f}")
        logger.info(f"Search range: [{min_pos}:{max_pos}] (error range: [{self.min_error:.1f}, {self.max_error:.1f}])")
        
        # Algorithm 1: Scan predicted range and filter by actual coordinates
        results = []
        candidates_checked = 0
        
        for i in range(min_pos, max_pos):
            actual_idx = self.sorted_indices[i]
            point_lat, point_lon = self.coordinates[actual_idx][0], self.coordinates[actual_idx][1]  # Safe extraction
            candidates_checked += 1
            
            # Check if point is within query rectangle (Algorithm 1, line 8)
            if (min_lat <= point_lat <= max_lat and 
                min_lon <= point_lon <= max_lon):
                results.append(actual_idx)
        
        logger.info(f"ZM MLP range query: checked {candidates_checked} candidates, found {len(results)} matches")
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
        """Perform batch point queries for multiple points."""
        results = []
        for coord in query_points:
            lat, lon = coord[0], coord[1]  # Safely extract first two values
            result = self.point_query(lat, lon, tolerance)
            results.append(result)
        return results
    
    def batch_range_queries(self, query_boxes: np.ndarray) -> List[List[int]]:
        """Perform batch range queries for multiple bounding boxes."""
        results = []
        for box in query_boxes:
            min_lat, max_lat, min_lon, max_lon = box[0], box[1], box[2], box[3]  # Safely extract first four values
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
            "num_stages": self.num_stages,
            "precision_bits": self.precision_bits,
            "device": str(self.device),
            "final_training_loss": self.training_loss[-1] if self.training_loss else None,
            "num_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "min_error": self.min_error,
            "max_error": self.max_error,
            "error_range": self.max_error - self.min_error,
            "r2_score": self.r2_score,
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
        self.sorted_z_addresses = None
        self.min_coords = None
        self.max_coords = None
        self.coord_scale = None
        self.build_time = None
        self.memory_usage = None
        self.training_loss = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ZM MLP Index cleared")