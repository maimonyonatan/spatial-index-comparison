"""
Query engine for spatial operations across different index types.
"""

import logging
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

from zm_rtree_research.indexes.rtree_index import RTreeIndex
from zm_rtree_research.indexes.zm_linear import ZMLinearIndex
from zm_rtree_research.indexes.zm_mlp import ZMMLPIndex

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Enumeration of supported index types."""
    RTREE = "rtree"
    ZM_LINEAR = "zm_linear"
    ZM_MLP = "zm_mlp"


class QueryType(Enum):
    """Enumeration of supported query types."""
    POINT = "point"
    RANGE = "range"
    KNN = "knn"


class QueryEngine:
    """
    Unified query engine for spatial operations.
    
    Provides a consistent interface for executing spatial queries across
    different index implementations (R-Tree, ZM Linear, ZM MLP).
    """
    
    def __init__(self):
        """Initialize the query engine."""
        self.indexes: Dict[str, Any] = {}
        self.coordinates: Optional[np.ndarray] = None
        self.morton_codes: Optional[np.ndarray] = None
        
    def add_index(
        self, 
        name: str, 
        index_type: IndexType, 
        coordinates: np.ndarray,
        morton_codes: Optional[np.ndarray] = None,
        **index_kwargs
    ) -> None:
        """
        Add and build an index.
        
        Args:
            name: Unique name for the index
            index_type: Type of index to create
            coordinates: Array of (lat, lon) coordinates
            morton_codes: Morton codes (required for ZM indexes)
            **index_kwargs: Additional arguments for index construction
        """
        logger.info(f"Adding {index_type.value} index: {name}")
        
        if index_type == IndexType.RTREE:
            index = RTreeIndex(**index_kwargs)
            index.build(coordinates)
            
        elif index_type == IndexType.ZM_LINEAR:
            if morton_codes is None:
                raise ValueError("Morton codes required for ZM Linear index")
            index = ZMLinearIndex(**index_kwargs)
            index.build(coordinates, morton_codes)
            
        elif index_type == IndexType.ZM_MLP:
            if morton_codes is None:
                raise ValueError("Morton codes required for ZM MLP index")
            index = ZMMLPIndex(**index_kwargs)
            index.build(coordinates, morton_codes)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.indexes[name] = {
            'index': index,
            'type': index_type,
            'stats': index.get_statistics()
        }
        
        # Store coordinates and Morton codes for reference
        if self.coordinates is None:
            self.coordinates = coordinates.copy()
        if morton_codes is not None and self.morton_codes is None:
            self.morton_codes = morton_codes.copy()
        
        logger.info(f"Successfully added index: {name}")
    
    def point_query(
        self, 
        lat: float, 
        lon: float, 
        index_name: Optional[str] = None,
        tolerance: float = 1e-6,
        measure_time: bool = True
    ) -> Dict[str, Any]:
        """
        Execute point query on specified index(es).
        
        Args:
            lat: Query latitude
            lon: Query longitude
            index_name: Name of specific index (None for all indexes)
            tolerance: Tolerance for approximate matching
            measure_time: Whether to measure query time
            
        Returns:
            Dictionary with query results and timing information
        """
        if not self.indexes:
            raise ValueError("No indexes available")
        
        target_indexes = [index_name] if index_name else list(self.indexes.keys())
        results = {}
        
        for name in target_indexes:
            if name not in self.indexes:
                logger.warning(f"Index {name} not found, skipping")
                continue
            
            index_info = self.indexes[name]
            index = index_info['index']
            
            start_time = time.perf_counter() if measure_time else None
            
            try:
                query_results = index.point_query(lat, lon, tolerance)
                query_time = time.perf_counter() - start_time if measure_time else None
                
                results[name] = {
                    'results': query_results,
                    'count': len(query_results),
                    'query_time_seconds': query_time,
                    'index_type': index_info['type'].value
                }
                
            except Exception as e:
                logger.error(f"Error executing point query on {name}: {e}")
                results[name] = {
                    'error': str(e),
                    'index_type': index_info['type'].value
                }
        
        return results
    
    def range_query(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        index_name: Optional[str] = None,
        measure_time: bool = True
    ) -> Dict[str, Any]:
        """
        Execute range query on specified index(es).
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            index_name: Name of specific index (None for all indexes)
            measure_time: Whether to measure query time
            
        Returns:
            Dictionary with query results and timing information
        """
        if not self.indexes:
            raise ValueError("No indexes available")
        
        target_indexes = [index_name] if index_name else list(self.indexes.keys())
        results = {}
        
        for name in target_indexes:
            if name not in self.indexes:
                logger.warning(f"Index {name} not found, skipping")
                continue
            
            index_info = self.indexes[name]
            index = index_info['index']
            
            start_time = time.perf_counter() if measure_time else None
            
            try:
                query_results = index.range_query(min_lat, max_lat, min_lon, max_lon)
                query_time = time.perf_counter() - start_time if measure_time else None
                
                results[name] = {
                    'results': query_results,
                    'count': len(query_results),
                    'query_time_seconds': query_time,
                    'index_type': index_info['type'].value
                }
                
            except Exception as e:
                logger.error(f"Error executing range query on {name}: {e}")
                results[name] = {
                    'error': str(e),
                    'index_type': index_info['type'].value
                }
        
        return results
    
    def knn_query(
        self,
        lat: float,
        lon: float,
        k: int = 1,
        index_name: Optional[str] = None,
        measure_time: bool = True
    ) -> Dict[str, Any]:
        """
        Execute k-nearest neighbor query on specified index(es).
        
        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of nearest neighbors
            index_name: Name of specific index (None for all indexes)
            measure_time: Whether to measure query time
            
        Returns:
            Dictionary with query results and timing information
        """
        if not self.indexes:
            raise ValueError("No indexes available")
        
        target_indexes = [index_name] if index_name else list(self.indexes.keys())
        results = {}
        
        for name in target_indexes:
            if name not in self.indexes:
                logger.warning(f"Index {name} not found, skipping")
                continue
            
            index_info = self.indexes[name]
            index = index_info['index']
            
            start_time = time.perf_counter() if measure_time else None
            
            try:
                query_results = index.knn_query(lat, lon, k)
                query_time = time.perf_counter() - start_time if measure_time else None
                
                results[name] = {
                    'results': query_results,
                    'count': len(query_results),
                    'query_time_seconds': query_time,
                    'index_type': index_info['type'].value
                }
                
            except Exception as e:
                logger.error(f"Error executing kNN query on {name}: {e}")
                results[name] = {
                    'error': str(e),
                    'index_type': index_info['type'].value
                }
        
        return results
    
    def batch_queries(
        self,
        query_type: QueryType,
        query_params: List[Dict[str, Any]],
        index_name: Optional[str] = None,
        measure_time: bool = True
    ) -> Dict[str, Any]:
        """
        Execute batch queries on specified index(es).
        
        Args:
            query_type: Type of queries to execute
            query_params: List of query parameter dictionaries
            index_name: Name of specific index (None for all indexes)
            measure_time: Whether to measure total query time
            
        Returns:
            Dictionary with batch query results and timing information
        """
        if not self.indexes:
            raise ValueError("No indexes available")
        
        target_indexes = [index_name] if index_name else list(self.indexes.keys())
        results = {}
        
        for name in target_indexes:
            if name not in self.indexes:
                logger.warning(f"Index {name} not found, skipping")
                continue
            
            index_info = self.indexes[name]
            index = index_info['index']
            
            start_time = time.perf_counter() if measure_time else None
            batch_results = []
            
            try:
                for params in query_params:
                    if query_type == QueryType.POINT:
                        result = index.point_query(**params)
                    elif query_type == QueryType.RANGE:
                        result = index.range_query(**params)
                    elif query_type == QueryType.KNN:
                        result = index.knn_query(**params)
                    else:
                        raise ValueError(f"Unsupported query type: {query_type}")
                    
                    batch_results.append(result)
                
                total_time = time.perf_counter() - start_time if measure_time else None
                total_results = sum(len(r) for r in batch_results)
                
                results[name] = {
                    'results': batch_results,
                    'num_queries': len(query_params),
                    'total_results': total_results,
                    'total_time_seconds': total_time,
                    'avg_time_per_query': total_time / len(query_params) if total_time else None,
                    'index_type': index_info['type'].value
                }
                
            except Exception as e:
                logger.error(f"Error executing batch queries on {name}: {e}")
                results[name] = {
                    'error': str(e),
                    'index_type': index_info['type'].value
                }
        
        return results
    
    def compare_indexes(
        self,
        query_type: QueryType,
        query_params: Dict[str, Any],
        index_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple indexes on the same query.
        
        Args:
            query_type: Type of query to execute
            query_params: Query parameters
            index_names: List of index names to compare (None for all)
            
        Returns:
            Dictionary with comparative results
        """
        target_indexes = index_names if index_names else list(self.indexes.keys())
        
        if query_type == QueryType.POINT:
            results = self.point_query(index_name=None, **query_params)
        elif query_type == QueryType.RANGE:
            results = self.range_query(index_name=None, **query_params)
        elif query_type == QueryType.KNN:
            results = self.knn_query(index_name=None, **query_params)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Filter to target indexes
        filtered_results = {k: v for k, v in results.items() if k in target_indexes}
        
        # Add comparison metrics
        if len(filtered_results) > 1:
            times = [r.get('query_time_seconds', float('inf')) 
                    for r in filtered_results.values() if 'error' not in r]
            if times:
                fastest_time = min(times)
                for name, result in filtered_results.items():
                    if 'query_time_seconds' in result and result['query_time_seconds'] is not None:
                        result['speedup_factor'] = result['query_time_seconds'] / fastest_time
        
        return filtered_results
    
    def get_index_statistics(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for specified index(es).
        
        Args:
            index_name: Name of specific index (None for all indexes)
            
        Returns:
            Dictionary with index statistics
        """
        if index_name:
            if index_name not in self.indexes:
                raise ValueError(f"Index {index_name} not found")
            return {index_name: self.indexes[index_name]['stats']}
        else:
            return {name: info['stats'] for name, info in self.indexes.items()}
    
    def clear_index(self, index_name: str) -> None:
        """
        Clear and remove an index.
        
        Args:
            index_name: Name of index to clear
        """
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} not found")
        
        self.indexes[index_name]['index'].clear()
        del self.indexes[index_name]
        logger.info(f"Cleared index: {index_name}")
    
    def clear_all_indexes(self) -> None:
        """Clear all indexes."""
        for name in list(self.indexes.keys()):
            self.clear_index(name)
        self.coordinates = None
        self.morton_codes = None
        logger.info("Cleared all indexes")
    
    def list_indexes(self) -> List[str]:
        """Get list of available index names."""
        return list(self.indexes.keys())
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of query engine state.
        
        Returns:
            Dictionary with engine summary
        """
        return {
            'num_indexes': len(self.indexes),
            'index_names': list(self.indexes.keys()),
            'index_types': [info['type'].value for info in self.indexes.values()],
            'data_points': len(self.coordinates) if self.coordinates is not None else 0,
            'has_morton_codes': self.morton_codes is not None
        }