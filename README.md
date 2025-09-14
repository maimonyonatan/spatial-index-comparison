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

- Python 3.12+
- CUDA-capable GPU (optional, for MLP training acceleration)

## 🛠 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd zm_rtree_project
```

2. **Install dependencies using uv**:
```bash
uv sync
```

3. **Activate the virtual environment**:
```bash
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

## 🏃 Quick Start

### 1. Run Basic Example
```bash
python examples/basic_example.py
```

### 2. Launch Interactive GUI
```bash
python run_gui.py
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
df = loader.load_csv("data/accidents.csv", sample_size=10000)

# Prepare coordinates and Morton codes
coordinates = df[['Start_Lat', 'Start_Lng']].values
normalized_coords = loader.normalize_coordinates(df, 'Start_Lat', 'Start_Lng')
morton_codes = loader.compute_morton_codes(normalized_coords)

# Build indexes
engine = QueryEngine()
engine.add_index("rtree", IndexType.RTREE, coordinates)
engine.add_index("zm_linear", IndexType.ZM_LINEAR, coordinates, morton_codes)
engine.add_index("zm_mlp", IndexType.ZM_MLP, coordinates, morton_codes)

# Execute queries
results = engine.range_query(39.0, 41.0, -76.0, -74.0)
knn_results = engine.knn_query(40.0, -75.0, k=10)

# Performance evaluation
evaluator = PerformanceEvaluator(engine)
evaluation = evaluator.comprehensive_evaluation()
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
│   ├── indexes/                     # Index implementations
│   ├── query/                       # Query engine
│   ├── evaluation/                  # Performance evaluation
│   └── gui/                         # Streamlit interface
├── examples/                        # Example scripts
├── tests/                           # Unit tests
├── docs/                           # Documentation
├── data/                           # Sample datasets
├── run_gui.py                      # GUI launcher
└── pyproject.toml                  # Project configuration
```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
uv sync --dev

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=zm_rtree_research --cov-report=html
```

## 📊 Expected Results

Based on spatial indexing literature, expected performance characteristics:

### Build Time
1. **R-Tree**: Moderate (O(n log n))
2. **ZM Linear**: Fast (O(n))
3. **ZM MLP**: Slowest (neural network training)

### Query Performance
- **Point Queries**: R-Tree typically fastest
- **Range Queries**: Depends on selectivity and data distribution
- **k-NN Queries**: R-Tree optimized, learned indexes competitive

### Memory Usage
1. **ZM Linear**: Lowest (small model)
2. **ZM MLP**: Low (compact neural network)
3. **R-Tree**: Higher (tree structure overhead)

## 🔬 Research Applications

This system enables research into:

- **Learned Index Effectiveness**: Performance vs. traditional structures
- **Data Distribution Impact**: How spatial clustering affects learned indexes
- **Query Pattern Analysis**: Performance across different query workloads
- **Trade-off Analysis**: Build time vs. query performance vs. memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Kraska, T., et al. (2018). "The Case for Learned Index Structures." SIGMOD.
- Guttman, A. (1984). "R-trees: A Dynamic Index Structure for Spatial Searching." SIGMOD.
- Morton, G. M. (1966). "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing."

## 🆘 Support

For issues and questions:

1. Check the documentation and examples
2. Search existing GitHub issues
3. Create a new issue with detailed description and reproduction steps

## 🏆 Acknowledgments

- Built with modern Python spatial libraries (rtree, geopandas, shapely)
- Machine learning powered by scikit-learn and PyTorch
- Interactive interface using Streamlit and Plotly
- Efficient package management with uv