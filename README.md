# ZM R-Tree Research: Spatial Index Comparison

A comprehensive research prototype for empirical comparison of traditional R-Tree spatial indexing with learned ZM (Z-order/Morton) indexes using machine learning approaches.

## 🎯 Overview

This project implements and compares three spatial indexing approaches:

1. **R-Tree Index**: Traditional hierarchical spatial data structure
2. **ZM Linear Index**: Learned index using linear regression on Morton codes
3. **ZM MLP Index**: Learned index using shallow neural networks on Morton codes

## 🚀 Features

- **Multiple Index Types**: R-Tree, Linear Regression, and MLP-based learned indexes
- **Comprehensive Evaluation**: Performance benchmarking across different query types
- **Interactive GUI**: Streamlit-based web interface for exploration and analysis
- **Query Types**: Point queries, range queries, and k-NN queries
- **Performance Metrics**: Build time, memory usage, query latency, and throughput
- **Visualization**: Interactive maps and performance charts

## 📋 Requirements

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (optional, for MLP training acceleration)

## 🛠 Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone <repository-url>
cd zm_rtree_project
```

3. **Install dependencies and create virtual environment**:
```bash
uv sync
```

## 🏃 Quick Start

### 1. Run Basic Example
```bash
uv run python examples/basic_example.py
```

### 2. Launch Interactive GUI
```bash
uv run streamlit run src/zm_rtree_research/gui/app.py
```

The GUI will open in your web browser at `http://localhost:8501`

## 📊 Usage Examples

### Python API Usage

```python
from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.query.engine import QueryEngine, IndexType
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator

# Load spatial dataset
loader = DataLoader()
df = loader.load_csv("examples/sample_data.csv", sample_size=10000)

# Prepare coordinates and Morton codes
coordinates = df[['lat', 'lon']].values
normalized_coords = loader.normalize_coordinates(df, 'lat', 'lon')
morton_codes = loader.compute_morton_codes(normalized_coords)

# Build indexes
engine = QueryEngine()
engine.add_index("rtree", IndexType.RTREE, coordinates)
engine.add_index("zm_linear", IndexType.ZM_LINEAR, coordinates, morton_codes)
engine.add_index("zm_mlp", IndexType.ZM_MLP, coordinates, morton_codes, epochs=50)

# Execute queries
results = engine.range_query(39.0, 41.0, -76.0, -74.0)
knn_results = engine.knn_query(40.0, -75.0, k=10)

# Performance evaluation
evaluator = PerformanceEvaluator(engine)
evaluation = evaluator.comprehensive_evaluation()
```

### Running with Your Own Dataset

```python
# Using the GUI (recommended)
uv run streamlit run src/zm_rtree_research/gui/app.py

# Or programmatically
from zm_rtree_research.data.loader import DataLoader

loader = DataLoader()
df = loader.load_csv("your_data.csv", 
                    lat_col="latitude", 
                    lon_col="longitude", 
                    sample_size=5000)
```

## 🏗 Architecture

### Core Components

- **`data/loader.py`**: Data loading and preprocessing utilities
- **`indexes/`**: Spatial index implementations
  - `rtree_index.py`: Traditional R-Tree using rtree library
  - `zm_linear.py`: Linear regression on Morton codes
  - `zm_mlp.py`: Neural network on Morton codes
- **`query/engine.py`**: Unified query interface
- **`evaluation/evaluator.py`**: Performance evaluation framework
- **`gui/app.py`**: Streamlit web interface

### Index Implementations

#### R-Tree Index
- Uses the `rtree` library with libspatialindex
- Traditional hierarchical spatial data structure
- Optimized for range and nearest neighbor queries

#### ZM Linear Index
- Maps 2D coordinates to 1D Morton codes (Z-order curve)
- Uses linear/polynomial regression to predict positions
- Leverages scikit-learn for model training

#### ZM MLP Index
- Shallow neural network (MLP) for position prediction
- Uses PyTorch for training and inference
- Supports GPU acceleration

## 📈 Evaluation Metrics

The system evaluates indexes across multiple dimensions:

### Performance Metrics
- **Build Time**: Time to construct the index
- **Memory Usage**: RAM consumption of the index structure
- **Query Latency**: Average time per query execution
- **Throughput**: Queries processed per second

### Query Types
- **Point Queries**: Exact location lookups with tolerance
- **Range Queries**: Rectangular region queries with varying selectivity
- **k-NN Queries**: Nearest neighbor searches with different k values

### Benchmarks
- Configurable number of point, range, and k-NN queries
- Multiple selectivity levels for range queries
- Different k values for k-NN queries

## 🖥 GUI Features

The Streamlit interface provides:

1. **Data Loading**: Upload and explore spatial datasets
2. **Index Building**: Configure and build different index types
3. **Query Execution**: Interactive query interface with real-time results
4. **Performance Evaluation**: Comprehensive benchmarking with visualizations
5. **Visualization**: Interactive maps and performance charts

## 📁 Project Structure

```
zm_rtree_project/
├── src/zm_rtree_research/           # Main package
│   ├── data/                        # Data loading utilities
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── indexes/                     # Index implementations
│   │   ├── __init__.py
│   │   ├── rtree_index.py
│   │   ├── zm_linear.py
│   │   └── zm_mlp.py
│   ├── query/                       # Query engine
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── evaluation/                  # Performance evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── gui/                         # Streamlit interface
│       ├── __init__.py
│       └── app.py
├── examples/                        # Example scripts
│   ├── basic_example.py
│   └── sample_data.csv             # Generated sample dataset
├── tests/                          # Unit tests
├── docs/                           # Documentation
├── data/                           # Additional datasets (optional)
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                         # Locked dependencies
└── README.md                       # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run basic example to verify installation
uv run python examples/basic_example.py

# Run tests (if available)
uv run pytest tests/ -v

# Check package imports
uv run python -c "from zm_rtree_research.gui.app import main; print('✅ Package imports successfully')"
```

## 🔧 Development

For development work:

```bash
# Install with development dependencies (if defined)
uv sync --dev

# Run with specific Python version
uv run --python 3.11 streamlit run src/zm_rtree_research/gui/app.py

# Add new dependencies
uv add package_name

# Update dependencies
uv sync --upgrade
```

## 📊 Expected Results

Based on spatial indexing literature, expected performance characteristics:

### Build Time
1. **R-Tree**: Moderate (O(n log n))
2. **ZM Linear**: Fast (O(n))
3. **ZM MLP**: Slowest (neural network training)

### Query Performance
- **Point Queries**: R-Tree typically fastest for exact matches
- **Range Queries**: Depends on selectivity and data distribution
- **k-NN Queries**: R-Tree optimized, learned indexes competitive for large k

### Memory Usage
1. **ZM Linear**: Lowest (small regression model)
2. **ZM MLP**: Low (compact neural network)
3. **R-Tree**: Higher (tree structure overhead)

### Accuracy
- **R-Tree**: 100% accurate (exact results)
- **ZM Learned Indexes**: High accuracy (>95%) with good R² scores
- **Trade-off**: Slight accuracy loss for memory/speed gains

## 🔬 Research Applications

This system enables research into:

- **Learned Index Effectiveness**: Performance vs. traditional structures
- **Data Distribution Impact**: How spatial clustering affects learned indexes
- **Query Pattern Analysis**: Performance across different query workloads
- **Trade-off Analysis**: Build time vs. query performance vs. memory usage
- **Scalability Studies**: Performance with varying dataset sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run basic example: `uv run python examples/basic_example.py`
5. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're using `uv run` prefix
2. **GUI not loading**: Check that port 8501 is available
3. **CUDA errors**: MLP training will fall back to CPU automatically
4. **Memory issues**: Reduce sample size in data loading

### Getting Help

For issues and questions:

1. Run the basic example first: `uv run python examples/basic_example.py`
2. Check that GUI launches: `uv run streamlit run src/zm_rtree_research/gui/app.py`
3. Search existing GitHub issues
4. Create a new issue with detailed description and reproduction steps

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Kraska, T., et al. (2018). "The Case for Learned Index Structures." SIGMOD.
- Guttman, A. (1984). "R-trees: A Dynamic Index Structure for Spatial Searching." SIGMOD.
- Morton, G. M. (1966). "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing."

## 🏆 Acknowledgments

- Built with modern Python spatial libraries (rtree, geopandas, shapely)
- Machine learning powered by scikit-learn and PyTorch
- Interactive interface using Streamlit and Plotly
- Efficient package management with uv
- Sample data generation using realistic coordinate distributions