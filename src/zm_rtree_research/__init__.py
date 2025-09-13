"""
ZM R-Tree Research: Empirical comparison of R-Tree vs. Learned ZM Index for spatial queries.

This package provides a comprehensive research prototype for comparing the performance
of traditional R-Tree spatial indexing with learned ZM (Z-order/Morton) indexes using
machine learning approaches.

Main Components:
- data: Data loading and preprocessing utilities
- indexes: R-Tree and Learned ZM Index implementations
- query: Query engine for spatial operations
- evaluation: Performance evaluation and metrics
- gui: Streamlit-based interactive interface
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.indexes.rtree_index import RTreeIndex
from zm_rtree_research.indexes.zm_linear import ZMLinearIndex
from zm_rtree_research.indexes.zm_mlp import ZMMLPIndex
from zm_rtree_research.query.engine import QueryEngine
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator

__all__ = [
    "DataLoader",
    "RTreeIndex", 
    "ZMLinearIndex",
    "ZMMLPIndex",
    "QueryEngine",
    "PerformanceEvaluator",
]