# Technical Diagrams and System Architecture

## Overview

This document provides comprehensive visual documentation of the ZM R-Tree Research project, including class diagrams, system architecture, data flow patterns, and performance evaluation frameworks. These diagrams serve as essential reference materials for understanding the system design, implementation relationships, and experimental methodology.

## Table of Contents

1. [System Architecture Diagram](#system-architecture-diagram)
2. [Class Diagrams](#class-diagrams)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Spatial Index Structure Diagrams](#spatial-index-structure-diagrams)
5. [Performance Evaluation Framework](#performance-evaluation-framework)
6. [Query Processing Pipeline](#query-processing-pipeline)
7. [Morton Code Generation Process](#morton-code-generation-process)
8. [Neural Network Architecture](#neural-network-architecture)

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Layer"
        A[US Accidents Dataset<br/>7.7M Records] --> B[Data Loader<br/>CSV Processing]
        B --> C[Coordinate Normalization<br/>& Validation]
    end
    
    subgraph "Spatial Index Layer"
        C --> D[Traditional R-Tree<br/>Hierarchical Structure]
        C --> E[ZM Linear Index<br/>Linear Regression]
        C --> F[ZM MLP Index<br/>Neural Network]
    end
    
    subgraph "Query Processing Layer"
        G[Query Engine] --> H[Point Queries]
        G --> I[Range Queries]
        G --> J[k-NN Queries]
        
        D --> G
        E --> G
        F --> G
    end
    
    subgraph "Evaluation Framework"
        K[Performance Evaluator] --> L[Query Time Metrics]
        K --> M[Memory Usage Analysis]
        K --> N[Index Construction Time]
        K --> O[Accuracy Measurements]
        
        G --> K
    end
    
    subgraph "Visualization & Interface"
        P[GUI Application<br/>Streamlit] --> Q[Interactive Maps]
        P --> R[Performance Charts]
        P --> S[Real-time Comparisons]
        
        K --> P
        G --> P
    end
    
    subgraph "Output & Documentation"
        T[Research Results<br/>Performance Data] --> U[Academic Paper]
        T --> V[Technical Documentation]
        T --> W[Benchmark Reports]
        
        K --> T
    end
```

---

## Class Diagrams

### Core Index Classes

```mermaid
classDiagram
    class SpatialIndex {
        <<abstract>>
        +insert(point: Point) void
        +search(query: Query) List~Point~
        +range_query(bounds: Rectangle) List~Point~
        +knn_query(point: Point, k: int) List~Point~
        +get_stats() IndexStats
    }
    
    class RTreeIndex {
        -root: RTreeNode
        -max_entries: int
        -min_entries: int
        +insert(point: Point) void
        +search(query: Query) List~Point~
        +split_node(node: RTreeNode) void
        +choose_leaf(point: Point) RTreeNode
        +adjust_tree(node: RTreeNode) void
    }
    
    class ZMLinearIndex {
        -model: LinearRegression
        -morton_codes: List~int~
        -points: List~Point~
        -is_trained: bool
        +train(points: List~Point~) void
        +insert(point: Point) void
        +search(query: Query) List~Point~
        +predict_position(morton_code: int) int
    }
    
    class ZMMLPIndex {
        -model: MLPRegressor
        -morton_codes: List~int~
        -points: List~Point~
        -is_trained: bool
        -hidden_layers: Tuple
        +train(points: List~Point~) void
        +insert(point: Point) void
        +search(query: Query) List~Point~
        +predict_position(morton_code: int) int
    }
    
    class Point {
        +x: float
        +y: float
        +id: str
        +morton_code: int
        +__init__(x: float, y: float, id: str)
        +to_morton() int
        +distance_to(other: Point) float
    }
    
    class Rectangle {
        +min_x: float
        +min_y: float
        +max_x: float
        +max_y: float
        +contains(point: Point) bool
        +intersects(other: Rectangle) bool
        +area() float
    }
    
    SpatialIndex <|-- RTreeIndex
    SpatialIndex <|-- ZMLinearIndex
    SpatialIndex <|-- ZMMLPIndex
    
    RTreeIndex --> Point
    ZMLinearIndex --> Point
    ZMMLPIndex --> Point
    
    SpatialIndex --> Rectangle
```

### Data Processing Classes

```mermaid
classDiagram
    class DataLoader {
        -file_path: str
        -coordinate_bounds: CoordinateBounds
        +load_data() List~Point~
        +validate_coordinates(lat: float, lng: float) bool
        +normalize_coordinates(lat: float, lng: float) Tuple
        +get_statistics() DataStats
    }
    
    class CoordinateBounds {
        +min_lat: float
        +max_lat: float
        +min_lng: float
        +max_lng: float
        +normalize(lat: float, lng: float) Tuple
        +denormalize(x: float, y: float) Tuple
    }
    
    class DataStats {
        +total_points: int
        +coordinate_ranges: CoordinateBounds
        +mean_coordinates: Tuple
        +std_coordinates: Tuple
        +spatial_distribution: Dict
    }
    
    class MortonEncoder {
        +precision_bits: int
        +encode(x: float, y: float) int
        +decode(morton_code: int) Tuple
        +interleave_bits(x: int, y: int) int
        +separate_bits(morton_code: int) Tuple
    }
    
    DataLoader --> CoordinateBounds
    DataLoader --> DataStats
    DataLoader --> MortonEncoder
    DataLoader --> Point
```

### Query Processing Classes

```mermaid
classDiagram
    class QueryEngine {
        -indexes: Dict~str, SpatialIndex~
        +register_index(name: str, index: SpatialIndex) void
        +execute_point_query(point: Point, tolerance: float) QueryResult
        +execute_range_query(bounds: Rectangle) QueryResult
        +execute_knn_query(point: Point, k: int) QueryResult
        +benchmark_query(query: Query, iterations: int) BenchmarkResult
    }
    
    class Query {
        <<abstract>>
        +query_type: str
        +timestamp: datetime
        +execute(index: SpatialIndex) QueryResult
    }
    
    class PointQuery {
        +target_point: Point
        +tolerance: float
        +execute(index: SpatialIndex) QueryResult
    }
    
    class RangeQuery {
        +query_bounds: Rectangle
        +execute(index: SpatialIndex) QueryResult
    }
    
    class KNNQuery {
        +center_point: Point
        +k: int
        +execute(index: SpatialIndex) QueryResult
    }
    
    class QueryResult {
        +results: List~Point~
        +execution_time: float
        +query_type: str
        +result_count: int
        +metadata: Dict
    }
    
    class BenchmarkResult {
        +query_results: List~QueryResult~
        +average_time: float
        +min_time: float
        +max_time: float
        +std_time: float
        +throughput: float
    }
    
    Query <|-- PointQuery
    Query <|-- RangeQuery
    Query <|-- KNNQuery
    
    QueryEngine --> Query
    QueryEngine --> QueryResult
    QueryEngine --> BenchmarkResult
    QueryEngine --> SpatialIndex
```

### Evaluation Framework Classes

```mermaid
classDiagram
    class PerformanceEvaluator {
        -indexes: Dict~str, SpatialIndex~
        -test_queries: List~Query~
        +add_index(name: str, index: SpatialIndex) void
        +run_benchmark_suite() EvaluationReport
        +measure_construction_time(index: SpatialIndex, data: List~Point~) float
        +measure_memory_usage(index: SpatialIndex) MemoryStats
        +generate_performance_report() PerformanceReport
    }
    
    class EvaluationReport {
        +index_performance: Dict~str, IndexPerformance~
        +query_performance: Dict~str, QueryPerformance~
        +construction_metrics: ConstructionMetrics
        +memory_metrics: MemoryMetrics
        +timestamp: datetime
        +generate_summary() str
        +export_to_csv(file_path: str) void
    }
    
    class IndexPerformance {
        +index_name: str
        +construction_time: float
        +memory_usage: MemoryStats
        +query_times: Dict~str, List~float~~
        +accuracy_metrics: AccuracyMetrics
    }
    
    class QueryPerformance {
        +query_type: str
        +average_time: float
        +throughput: float
        +accuracy: float
        +result_distribution: Dict
    }
    
    class MemoryStats {
        +peak_memory: int
        +average_memory: int
        +memory_efficiency: float
        +index_size: int
        +overhead_ratio: float
    }
    
    class AccuracyMetrics {
        +precision: float
        +recall: float
        +f1_score: float
        +error_rate: float
        +false_positives: int
        +false_negatives: int
    }
    
    PerformanceEvaluator --> EvaluationReport
    EvaluationReport --> IndexPerformance
    EvaluationReport --> QueryPerformance
    IndexPerformance --> MemoryStats
    IndexPerformance --> AccuracyMetrics
```

---

## Data Flow Diagrams

### Data Processing Pipeline

```mermaid
flowchart TD
    A[Raw CSV Data<br/>US_Accidents_March23.csv] --> B[Data Validation<br/>Check Coordinates]
    B --> C{Valid Data?}
    C -->|Yes| D[Coordinate Normalization<br/>Scale to [0,1]]
    C -->|No| E[Error Logging<br/>Skip Invalid Records]
    
    D --> F[Morton Code Generation<br/>Z-order Curve Encoding]
    F --> G[Point Object Creation<br/>Spatial Data Structure]
    
    G --> H[Index Construction Phase]
    H --> I[R-Tree Building<br/>Hierarchical Structure]
    H --> J[ZM Linear Training<br/>Linear Regression Model]
    H --> K[ZM MLP Training<br/>Neural Network Model]
    
    I --> L[Query Processing Ready]
    J --> L
    K --> L
    
    L --> M[Performance Evaluation<br/>Benchmark Execution]
    M --> N[Results Analysis<br/>Statistical Comparison]
    N --> O[Report Generation<br/>Documentation & Visualization]
```

### Query Execution Flow

```mermaid
flowchart LR
    A[Query Request] --> B{Query Type?}
    
    B -->|Point Query| C[Point Query Handler]
    B -->|Range Query| D[Range Query Handler]
    B -->|k-NN Query| E[k-NN Query Handler]
    
    C --> F[R-Tree Processing]
    C --> G[ZM Linear Processing]
    C --> H[ZM MLP Processing]
    
    D --> F
    D --> G
    D --> H
    
    E --> F
    E --> G
    E --> H
    
    F --> I[Result Collection]
    G --> I
    H --> I
    
    I --> J[Performance Metrics<br/>Time, Memory, Accuracy]
    J --> K[Formatted Response<br/>Query Results + Metrics]
```

---

## Spatial Index Structure Diagrams

### R-Tree Hierarchical Structure

```mermaid
graph TD
    subgraph "R-Tree Structure"
        A[Root Node<br/>Level 2] --> B[Internal Node 1<br/>Level 1]
        A --> C[Internal Node 2<br/>Level 1]
        A --> D[Internal Node 3<br/>Level 1]
        
        B --> E[Leaf Node 1<br/>Points 1-50]
        B --> F[Leaf Node 2<br/>Points 51-100]
        
        C --> G[Leaf Node 3<br/>Points 101-150]
        C --> H[Leaf Node 4<br/>Points 151-200]
        
        D --> I[Leaf Node 5<br/>Points 201-250]
        D --> J[Leaf Node 6<br/>Points 251-300]
    end
    
    subgraph "Bounding Rectangles"
        K[MBR Root<br/>Entire Dataset]
        L[MBR Internal 1<br/>Ohio Region]
        M[MBR Internal 2<br/>California Region]
        N[MBR Internal 3<br/>Other States]
    end
    
    A -.-> K
    B -.-> L
    C -.-> M
    D -.-> N
```

### ZM Index Array Structure

```mermaid
graph LR
    subgraph "Sorted Morton Array"
        A[Index 0<br/>Morton: 1001<br/>Point: P₁] --> B[Index 1<br/>Morton: 1247<br/>Point: P₂]
        B --> C[Index 2<br/>Morton: 1389<br/>Point: P₃]
        C --> D[Index 3<br/>Morton: 1456<br/>Point: P₄]
        D --> E[...]
        E --> F[Index n-1<br/>Morton: 9876<br/>Point: Pₙ]
    end
    
    subgraph "ML Model Prediction"
        G[Input: Morton Code] --> H[Linear/MLP Model]
        H --> I[Predicted Array Index]
        I --> J[Binary Search Range<br/>±ε positions]
    end
    
    G -.-> A
    I -.-> C
```

---

## Performance Evaluation Framework

### Benchmark Test Suite

```mermaid
flowchart TB
    subgraph "Test Data Preparation"
        A[Load Dataset<br/>3,680 Points] --> B[Generate Test Queries<br/>Point, Range, k-NN]
        B --> C[Create Index Instances<br/>R-Tree, ZM-Linear, ZM-MLP]
    end
    
    subgraph "Construction Benchmarks"
        D[Measure Build Time] --> E[Memory Usage Tracking]
        E --> F[Index Size Analysis]
    end
    
    subgraph "Query Benchmarks"
        G[Point Query Tests<br/>1000 iterations] --> H[Range Query Tests<br/>500 iterations]
        H --> I[k-NN Query Tests<br/>300 iterations]
    end
    
    subgraph "Performance Metrics"
        J[Execution Time<br/>Average, Min, Max, StdDev] --> K[Throughput<br/>Queries per Second]
        K --> L[Memory Efficiency<br/>Peak Usage, Overhead]
        L --> M[Accuracy Metrics<br/>Precision, Recall]
    end
    
    subgraph "Statistical Analysis"
        N[Comparative Analysis<br/>Index Performance] --> O[Statistical Significance<br/>t-tests, ANOVA]
        O --> P[Performance Rankings<br/>Best/Worst Cases]
    end
    
    C --> D
    C --> G
    G --> J
    J --> N
```

### Results Visualization Pipeline

```mermaid
graph LR
    A[Raw Performance Data<br/>JSON/CSV Format] --> B[Data Aggregation<br/>Statistical Processing]
    B --> C{Visualization Type}
    
    C -->|Time Series| D[Line Charts<br/>Query Time vs Dataset Size]
    C -->|Comparative| E[Bar Charts<br/>Index Performance Comparison]
    C -->|Distribution| F[Box Plots<br/>Query Time Distribution]
    C -->|Correlation| G[Scatter Plots<br/>Memory vs Performance]
    C -->|Spatial| H[Geographic Maps<br/>Query Result Visualization]
    
    D --> I[Interactive Dashboard<br/>Streamlit GUI]
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J[Research Report<br/>Academic Documentation]
    I --> K[Technical Presentation<br/>Stakeholder Communication]
```

---

## Query Processing Pipeline

### Point Query Processing

```mermaid
sequenceDiagram
    participant Client
    participant QueryEngine
    participant RTree as R-Tree Index
    participant ZMLinear as ZM Linear Index
    participant ZMMlp as ZM MLP Index
    
    Client->>QueryEngine: execute_point_query(point, tolerance)
    
    par R-Tree Processing
        QueryEngine->>RTree: search(point_query)
        RTree->>RTree: traverse_tree(point)
        RTree->>RTree: check_leaf_nodes()
        RTree-->>QueryEngine: results + metrics
    and ZM Linear Processing
        QueryEngine->>ZMLinear: search(point_query)
        ZMLinear->>ZMLinear: compute_morton_code(point)
        ZMLinear->>ZMLinear: predict_position(morton)
        ZMLinear->>ZMLinear: binary_search(predicted_pos)
        ZMLinear-->>QueryEngine: results + metrics
    and ZM MLP Processing
        QueryEngine->>ZMMlp: search(point_query)
        ZMMlp->>ZMMlp: compute_morton_code(point)
        ZMMlp->>ZMMlp: neural_predict(morton)
        ZMMlp->>ZMMlp: binary_search(predicted_pos)
        ZMMlp-->>QueryEngine: results + metrics
    end
    
    QueryEngine->>QueryEngine: aggregate_results()
    QueryEngine->>QueryEngine: compute_performance_metrics()
    QueryEngine-->>Client: comparative_results
```

### Range Query Processing

```mermaid
flowchart TD
    A[Range Query Request<br/>Rectangle Bounds] --> B{Index Type}
    
    B -->|R-Tree| C[R-Tree Range Processing]
    B -->|ZM Linear| D[ZM Linear Range Processing]
    B -->|ZM MLP| E[ZM MLP Range Processing]
    
    C --> C1[Check Root MBR<br/>Intersection Test]
    C1 --> C2[Recursive Descent<br/>Internal Nodes]
    C2 --> C3[Leaf Node Filtering<br/>Point-in-Rectangle Test]
    
    D --> D1[Compute Morton Range<br/>Rectangle to Z-order]
    D1 --> D2[Predict Range Positions<br/>Linear Model]
    D2 --> D3[Scan Predicted Range<br/>Refine Results]
    
    E --> E1[Compute Morton Range<br/>Rectangle to Z-order]
    E1 --> E2[Neural Position Prediction<br/>MLP Model]
    E2 --> E3[Scan Predicted Range<br/>Refine Results]
    
    C3 --> F[Result Aggregation]
    D3 --> F
    E3 --> F
    
    F --> G[Performance Metrics<br/>Time, Accuracy, Memory]
    G --> H[Query Response<br/>Points + Statistics]
```

---

## Morton Code Generation Process

### Z-Order Curve Encoding

```mermaid
flowchart TD
    subgraph "Coordinate Input"
        A[Normalized Coordinates<br/>x ∈ [0,1], y ∈ [0,1]] --> B[Scale to Integer Range<br/>x' ∈ [0, 2ⁿ-1]]
    end
    
    subgraph "Bit Interleaving Process"
        B --> C[Binary Representation<br/>x' = b₁b₂...bₙ<br/>y' = c₁c₂...cₙ]
        C --> D[Bit Interleaving<br/>morton = c₁b₁c₂b₂...cₙbₙ]
    end
    
    subgraph "Morton Code Output"
        D --> E[Final Morton Code<br/>Integer Value]
        E --> F[Z-Order Position<br/>Spatial Locality Preserved]
    end
    
    subgraph "Example Calculation"
        G[Point (0.25, 0.75)<br/>x=0.25, y=0.75] --> H[Scale (n=4 bits)<br/>x'=4, y'=12]
        H --> I[Binary<br/>x'=0100₂, y'=1100₂]
        I --> J[Interleave<br/>morton=11010000₂=208₁₀]
    end
```

### Spatial Locality Visualization

```mermaid
graph TB
    subgraph "2D Space Quadrants"
        A[Q1: NE<br/>11xx] --> B[Q2: NW<br/>10xx]
        C[Q3: SW<br/>00xx] --> D[Q4: SE<br/>01xx]
    end
    
    subgraph "Z-Order Traversal"
        E[Start: 00<br/>SW Corner] --> F[01<br/>SE Corner]
        F --> G[10<br/>NW Corner]
        G --> H[11<br/>NE Corner]
    end
    
    subgraph "Morton Code Sequence"
        I[000...000<br/>Min Value] --> J[Increasing<br/>Morton Codes]
        J --> K[111...111<br/>Max Value]
    end
    
    E -.-> I
    H -.-> K
```

---

## Neural Network Architecture

### ZM MLP Index Structure

```mermaid
graph TB
    subgraph "Input Layer"
        A[Morton Code<br/>Single Integer Input<br/>Range: [0, 2³²-1]]
    end
    
    subgraph "Hidden Layers"
        B[Hidden Layer 1<br/>128 Neurons<br/>ReLU Activation]
        C[Hidden Layer 2<br/>64 Neurons<br/>ReLU Activation]
        D[Hidden Layer 3<br/>32 Neurons<br/>ReLU Activation]
    end
    
    subgraph "Output Layer"
        E[Output Layer<br/>1 Neuron<br/>Linear Activation<br/>Predicted Array Index]
    end
    
    subgraph "Training Process"
        F[Training Data<br/>Morton Code → Array Index<br/>Pairs]
        G[Loss Function<br/>Mean Squared Error]
        H[Optimizer<br/>Adam Algorithm]
        I[Regularization<br/>L2 Penalty]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    
    F --> G
    G --> H
    H --> I
    
    subgraph "Model Parameters"
        J[Learning Rate: 0.001<br/>Batch Size: 32<br/>Epochs: 100<br/>Validation Split: 0.2]
    end
```

### Training Data Flow

```mermaid
flowchart LR
    A[Spatial Points<br/>Coordinates] --> B[Morton Encoding<br/>Z-order Conversion]
    B --> C[Array Sorting<br/>Morton Code Order]
    C --> D[Training Pairs<br/>Morton → Index]
    
    D --> E[Train/Test Split<br/>80/20 Division]
    E --> F[Model Training<br/>Supervised Learning]
    F --> G[Validation<br/>Performance Testing]
    
    G --> H{Convergence?}
    H -->|No| I[Hyperparameter Tuning<br/>Architecture Adjustment]
    H -->|Yes| J[Final Model<br/>Ready for Deployment]
    
    I --> F
    J --> K[Index Integration<br/>Query Processing]
```

---

## System Integration Diagram

### Complete System Overview

```mermaid
graph TB
    subgraph "External Dependencies"
        A[NumPy<br/>Numerical Computing] --> B[scikit-learn<br/>ML Algorithms]
        C[Pandas<br/>Data Processing] --> D[Streamlit<br/>Web Interface]
        E[Matplotlib<br/>Visualization] --> F[Plotly<br/>Interactive Charts]
    end
    
    subgraph "Core System Components"
        G[Data Layer<br/>CSV Processing] --> H[Index Layer<br/>Spatial Structures]
        H --> I[Query Layer<br/>Search Operations]
        I --> J[Evaluation Layer<br/>Performance Analysis]
        J --> K[Interface Layer<br/>User Interaction]
    end
    
    subgraph "Output Interfaces"
        L[Research Documentation<br/>Academic Papers] --> M[Technical Reports<br/>Implementation Details]
        N[Interactive Dashboard<br/>Real-time Analysis] --> O[Export Capabilities<br/>CSV, JSON, Images]
    end
    
    A --> G
    B --> H
    C --> G
    D --> K
    E --> J
    F --> K
    
    K --> N
    J --> L
    J --> M
    N --> O
```

---

## Conclusion

These technical diagrams provide a comprehensive visual reference for the ZM R-Tree Research project, covering all major system components, data flows, and architectural relationships. The diagrams serve multiple purposes:

1. **Design Documentation**: Clear visualization of system architecture and component relationships
2. **Implementation Guide**: Detailed class structures and method interfaces for development
3. **Educational Resource**: Visual learning aids for understanding spatial indexing concepts
4. **Research Communication**: Professional diagrams for academic presentations and publications
5. **Maintenance Reference**: System overview for future development and optimization

The modular design illustrated in these diagrams enables independent development and testing of each component while maintaining clear interfaces and data flow patterns. This architectural approach supports both current research objectives and future extensibility for additional spatial indexing algorithms and evaluation metrics.

---

*This technical diagram documentation complements the comprehensive ZM R-Tree Research project documentation, providing essential visual references for system understanding, implementation, and research communication.*