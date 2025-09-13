"""
Command-line interface for the ZM R-Tree research system.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

from zm_rtree_research.data.loader import DataLoader
from zm_rtree_research.query.engine import QueryEngine, IndexType
from zm_rtree_research.evaluation.evaluator import PerformanceEvaluator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """ZM R-Tree Research: Compare spatial index performance."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--lat-col', default='Start_Lat', help='Latitude column name')
@click.option('--lon-col', default='Start_Lng', help='Longitude column name')
@click.option('--sample-size', type=int, help='Number of records to sample')
@click.option('--state', help='Filter by state')
@click.option('--city', help='Filter by city')
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory for results')
def benchmark(
    data_file: Path,
    lat_col: str,
    lon_col: str,
    sample_size: Optional[int],
    state: Optional[str],
    city: Optional[str],
    output_dir: Optional[Path]
):
    """Run comprehensive benchmark on a dataset."""
    click.echo(f"🚀 Starting benchmark with dataset: {data_file}")
    
    try:
        # Load data
        loader = DataLoader()
        click.echo("📊 Loading dataset...")
        df = loader.load_csv(data_file, lat_col, lon_col, sample_size)
        
        # Apply geographic filtering
        if state or city:
            geographic_filter = {}
            if state:
                geographic_filter['state'] = state
            if city:
                geographic_filter['city'] = city
            df = loader.geographic_subset(**geographic_filter)
        
        # Prepare data
        click.echo("🔧 Processing coordinates...")
        coordinates = df[[lat_col, lon_col]].values
        normalized_coords = loader.normalize_coordinates(df, lat_col, lon_col)
        morton_codes = loader.compute_morton_codes(normalized_coords)
        
        click.echo(f"✅ Dataset ready: {len(coordinates)} points")
        
        # Build indexes
        query_engine = QueryEngine()
        
        click.echo("🏗️ Building R-Tree index...")
        query_engine.add_index("rtree", IndexType.RTREE, coordinates)
        
        click.echo("🏗️ Building ZM Linear index...")
        query_engine.add_index("zm_linear", IndexType.ZM_LINEAR, coordinates, morton_codes)
        
        click.echo("🏗️ Building ZM MLP index...")
        query_engine.add_index("zm_mlp", IndexType.ZM_MLP, coordinates, morton_codes, epochs=50)
        
        # Run evaluation
        click.echo("⚡ Running performance evaluation...")
        evaluator = PerformanceEvaluator(query_engine)
        results = evaluator.comprehensive_evaluation(output_dir)
        
        # Display results
        click.echo("\n" + "="*80)
        click.echo("📊 BENCHMARK RESULTS")
        click.echo("="*80)
        
        report = evaluator.get_comparison_report()
        click.echo(report)
        
        if output_dir:
            click.echo(f"\n💾 Detailed results saved to: {output_dir}")
        
        click.echo("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during benchmark: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--lat-col', default='Start_Lat', help='Latitude column name')
@click.option('--lon-col', default='Start_Lng', help='Longitude column name')
@click.option('--sample-size', type=int, help='Number of records to sample')
@click.option('--index-type', type=click.Choice(['rtree', 'zm_linear', 'zm_mlp']), 
              default='rtree', help='Index type to use')
@click.option('--query-lat', type=float, required=True, help='Query latitude')
@click.option('--query-lon', type=float, required=True, help='Query longitude')
@click.option('--k', type=int, default=5, help='Number of nearest neighbors')
def knn_query(
    data_file: Path,
    lat_col: str,
    lon_col: str,
    sample_size: Optional[int],
    index_type: str,
    query_lat: float,
    query_lon: float,
    k: int
):
    """Perform k-nearest neighbor query."""
    click.echo(f"🔍 Performing {k}-NN query at ({query_lat}, {query_lon})")
    
    try:
        # Load and prepare data
        loader = DataLoader()
        df = loader.load_csv(data_file, lat_col, lon_col, sample_size)
        coordinates = df[[lat_col, lon_col]].values
        
        # Build index
        query_engine = QueryEngine()
        
        if index_type == 'rtree':
            query_engine.add_index("index", IndexType.RTREE, coordinates)
        elif index_type == 'zm_linear':
            normalized_coords = loader.normalize_coordinates(df, lat_col, lon_col)
            morton_codes = loader.compute_morton_codes(normalized_coords)
            query_engine.add_index("index", IndexType.ZM_LINEAR, coordinates, morton_codes)
        elif index_type == 'zm_mlp':
            normalized_coords = loader.normalize_coordinates(df, lat_col, lon_col)
            morton_codes = loader.compute_morton_codes(normalized_coords)
            query_engine.add_index("index", IndexType.ZM_MLP, coordinates, morton_codes, epochs=50)
        
        # Execute query
        results = query_engine.knn_query(query_lat, query_lon, k, index_name="index")
        result = results["index"]
        
        if 'error' in result:
            click.echo(f"❌ Query failed: {result['error']}", err=True)
            sys.exit(1)
        
        # Display results
        click.echo(f"\n📊 Query Results ({index_type.upper()}):")
        click.echo(f"Query time: {result['query_time_seconds']:.6f} seconds")
        click.echo(f"Results found: {result['count']}")
        click.echo("\nNearest neighbors:")
        
        for i, (idx, distance) in enumerate(result['results'][:10], 1):
            point_lat, point_lon = coordinates[idx][0], coordinates[idx][1]  # Safe extraction
            click.echo(f"  {i}. Point {idx}: ({point_lat:.6f}, {point_lon:.6f}) - Distance: {distance:.6f}")
        
        if len(result['results']) > 10:
            click.echo(f"  ... and {len(result['results']) - 10} more")
        
    except Exception as e:
        click.echo(f"❌ Error during query: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--lat-col', default='Start_Lat', help='Latitude column name')
@click.option('--lon-col', default='Start_Lng', help='Longitude column name')
@click.option('--sample-size', type=int, help='Number of records to sample')
@click.option('--min-lat', type=float, required=True, help='Minimum latitude')
@click.option('--max-lat', type=float, required=True, help='Maximum latitude')
@click.option('--min-lon', type=float, required=True, help='Minimum longitude')
@click.option('--max-lon', type=float, required=True, help='Maximum longitude')
def range_query(
    data_file: Path,
    lat_col: str,
    lon_col: str,
    sample_size: Optional[int],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
):
    """Perform range query and compare all index types."""
    click.echo(f"🔍 Performing range query: [{min_lat}, {max_lat}] x [{min_lon}, {max_lon}]")
    
    try:
        # Load and prepare data
        loader = DataLoader()
        df = loader.load_csv(data_file, lat_col, lon_col, sample_size)
        coordinates = df[[lat_col, lon_col]].values
        normalized_coords = loader.normalize_coordinates(df, lat_col, lon_col)
        morton_codes = loader.compute_morton_codes(normalized_coords)
        
        # Build all indexes
        query_engine = QueryEngine()
        
        click.echo("Building indexes...")
        query_engine.add_index("rtree", IndexType.RTREE, coordinates)
        query_engine.add_index("zm_linear", IndexType.ZM_LINEAR, coordinates, morton_codes)
        query_engine.add_index("zm_mlp", IndexType.ZM_MLP, coordinates, morton_codes, epochs=50)
        
        # Execute query
        results = query_engine.range_query(min_lat, max_lat, min_lon, max_lon)
        
        # Display results
        click.echo(f"\n📊 Range Query Results:")
        click.echo("-" * 60)
        
        for index_name, result in results.items():
            if 'error' in result:
                click.echo(f"{index_name.upper():>12}: ERROR - {result['error']}")
            else:
                click.echo(f"{index_name.upper():>12}: {result['count']:>6} results in {result['query_time_seconds']:.6f}s")
        
        # Show fastest index
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            fastest = min(valid_results.items(), key=lambda x: x[1]['query_time_seconds'])
            click.echo(f"\n🏆 Fastest: {fastest[0].upper()} ({fastest[1]['query_time_seconds']:.6f}s)")
        
    except Exception as e:
        click.echo(f"❌ Error during query: {e}", err=True)
        sys.exit(1)


@cli.command()
def gui():
    """Launch the interactive Streamlit GUI."""
    click.echo("🚀 Launching Streamlit GUI...")
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the path to the GUI module
        gui_module = "zm_rtree_research.gui.app"
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "-m", gui_module]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        click.echo("\n👋 GUI closed by user")
    except Exception as e:
        click.echo(f"❌ Error launching GUI: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option('--lat-col', default='Start_Lat', help='Latitude column name')
@click.option('--lon-col', default='Start_Lng', help='Longitude column name')
@click.option('--sample-size', type=int, default=1000, help='Number of records to sample')
def info(data_file: Path, lat_col: str, lon_col: str, sample_size: int):
    """Display dataset information and statistics."""
    click.echo(f"📊 Analyzing dataset: {data_file}")
    
    try:
        loader = DataLoader()
        df = loader.load_csv(data_file, lat_col, lon_col, sample_size)
        coordinates = df[[lat_col, lon_col]].values
        
        click.echo(f"\n📈 Dataset Statistics:")
        click.echo(f"  Total records: {len(df):,}")
        click.echo(f"  Latitude range: {coordinates[:, 0].min():.6f} to {coordinates[:, 0].max():.6f}")
        click.echo(f"  Longitude range: {coordinates[:, 1].min():.6f} to {coordinates[:, 1].max():.6f}")
        click.echo(f"  Coordinate span: {coordinates[:, 0].max() - coordinates[:, 0].min():.6f} x {coordinates[:, 1].max() - coordinates[:, 1].min():.6f}")
        
        # Additional columns info
        if len(df.columns) > 2:
            click.echo(f"  Additional columns: {', '.join([col for col in df.columns if col not in [lat_col, lon_col]])}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        click.echo(f"  Memory usage: {memory_mb:.2f} MB")
        
        # Show sample data
        click.echo(f"\n📋 Sample Data:")
        click.echo(df.head().to_string(index=False))
        
    except Exception as e:
        click.echo(f"❌ Error analyzing dataset: {e}", err=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()