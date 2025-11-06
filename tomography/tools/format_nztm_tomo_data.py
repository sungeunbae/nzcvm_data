#!/usr/bin/env python3
"""
Convert raw tomography data to sparse HDF5 format.

Input formats: CSV, Parquet, NetCDF
Output: Sparse HDF5 with structure:
  - Root: x_nztm, y_nztm (unique coordinate arrays)
  - For each elevation: structured array with (x_idx, y_idx, vp, vs, rho)

Advantages:
  - Compact storage (no empty grid cells)
  - Fast elevation-based queries
  - Suitable as input to interpolate_nztm_tomo_data.py
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_data(input_file: Path) -> pd.DataFrame:
    """Load data from CSV, Parquet, or NetCDF."""
    suffix = input_file.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(input_file)
    elif suffix == '.parquet':
        return _load_parquet_chunked(input_file)
    elif suffix in ['.nc', '.ncdf', '.netcdf']:
        import xarray as xr
        ds = xr.open_dataset(input_file)
        return ds.to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv, .parquet, or .nc")


def _load_parquet_chunked(input_file: Path) -> pd.DataFrame:
    """Load parquet using row groups for memory efficiency."""
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(input_file)
    chunks = []
    
    print(f"   Loading {input_file.name} ({input_file.stat().st_size / 1e9:.2f} GB)...")
    for row_group_idx in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_groups([row_group_idx]).to_pandas()
        chunks.append(chunk)
        if (row_group_idx + 1) % 5 == 0 or row_group_idx == parquet_file.num_row_groups - 1:
            print(f"     Row group {row_group_idx + 1}/{parquet_file.num_row_groups} ({len(chunk):,} rows)")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Total rows: {len(df):,}")
    return df


def prepare_data(
    df: pd.DataFrame,
    x_nztm_col_idx: int,
    y_nztm_col_idx: int,
    depth_col_idx: int,
    vp_col_idx: int,
    vs_col_idx: int,
    rho_col_idx: int,
    already_elev: bool = False
) -> pd.DataFrame:
    """Prepare data by selecting columns by index."""
    df = df.copy()
    
    df_selected = pd.DataFrame({
        'x_nztm': df.iloc[:, x_nztm_col_idx].values,
        'y_nztm': df.iloc[:, y_nztm_col_idx].values,
        'elevation': df.iloc[:, depth_col_idx].values,
        'vp': df.iloc[:, vp_col_idx].values,
        'vs': df.iloc[:, vs_col_idx].values,
        'rho': df.iloc[:, rho_col_idx].values
    })
    
    # Convert depth to elevation if needed
    if not already_elev:
        df_selected['elevation'] = -1 * df_selected['elevation']
    
    return df_selected


def write_hdf5_sparse(
    output_path: Path,
    df: pd.DataFrame,
    compression_level: int = 4
) -> None:
    """
    Write data to sparse HDF5 format.
    
    Structure:
      Root: x_nztm (1D), y_nztm (1D)
      For each elevation: structured array with columns (x_idx, y_idx, vp, vs, rho)
    """
    
    print(f"\n   Preparing sparse HDF5 format...")
    
    # Get unique coordinates
    x_unique = np.sort(df['x_nztm'].unique()).astype(np.float32)
    y_unique = np.sort(df['y_nztm'].unique()).astype(np.float32)
    elevations = sorted(df['elevation'].unique())
    
    print(f"   Unique X coords: {len(x_unique):,}")
    print(f"   Unique Y coords: {len(y_unique):,}")
    print(f"   Unique elevations: {len(elevations)}")
    print(f"   Total data points: {len(df):,}")
    
    # Create lookup dictionaries
    x_to_idx = {x: i for i, x in enumerate(x_unique)}
    y_to_idx = {y: i for i, y in enumerate(y_unique)}
    
    print(f"\n   Writing HDF5...")
    
    with h5py.File(output_path, "w") as hf:
        # Store coordinate axes at root
        hf.create_dataset("x_nztm", data=x_unique, dtype=np.float32)
        hf.create_dataset("y_nztm", data=y_unique, dtype=np.float32)
        
        # Store data for each elevation
        for elev in elevations:
            df_e = df[np.isclose(df["elevation"], elev, atol=1e-3)]
            n_points = len(df_e)
            
            print(f"     Elevation {elev:7.2f} km: {n_points:,} points", end="")
            
            if n_points > 0:
                # Create structured array
                dt = np.dtype([
                    ('x_idx', np.int32),
                    ('y_idx', np.int32),
                    ('vp', np.float32),
                    ('vs', np.float32),
                    ('rho', np.float32)
                ])
                
                data_array = np.zeros(n_points, dtype=dt)
                
                for i, (idx, row) in enumerate(df_e.iterrows()):
                    data_array[i]['x_idx'] = x_to_idx[row['x_nztm']]
                    data_array[i]['y_idx'] = y_to_idx[row['y_nztm']]
                    data_array[i]['vp'] = row['vp']
                    data_array[i]['vs'] = row['vs']
                    data_array[i]['rho'] = row['rho']
                
                # Store as group with structured array
                elev_key = f"{elev:.1f}"
                grp = hf.create_group(elev_key)
                grp.create_dataset(
                    "data",
                    data=data_array,
                    compression="gzip",
                    compression_opts=compression_level
                )
                print(f" -> Stored")
            else:
                print()
    
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n   Output: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")


def main() -> None:
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description="Convert raw tomography data to sparse HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sparse format is efficient for storage and point-based access. Use the output
with interpolate_nztm_tomo_data.py to create gridded data for visualization.

Examples:
  # CSV input
  python %(prog)s data.csv --x-nztm-col 11 --y-nztm-col 12 \\
    --depth-col 8 --vp-col 0 --vs-col 2 --rho-col 3
  
  # Parquet input (memory efficient chunked loading)
  python %(prog)s data.parquet --x-nztm-col 13 --y-nztm-col 14 \\
    --depth-col 2 --vp-col 7 --vs-col 8 --rho-col 10
  
  # If depth column is already elevation (positive up):
  python %(prog)s data.csv --x-nztm-col 0 --y-nztm-col 1 \\
    --depth-col 2 --vp-col 3 --vs-col 4 --rho-col 5 --already-elev
        """
    )
    parser.add_argument("input_file", type=Path, help="Input file (CSV, Parquet, or NetCDF)")
    
    # Column mapping by index
    parser.add_argument("--x-nztm-col", type=int, required=True, help="NZTM X coordinate column index")
    parser.add_argument("--y-nztm-col", type=int, required=True, help="NZTM Y coordinate column index")
    parser.add_argument("--depth-col", type=int, required=True, help="Depth column index")
    parser.add_argument("--vp-col", type=int, required=True, help="Vp column index")
    parser.add_argument("--vs-col", type=int, required=True, help="Vs column index")
    parser.add_argument("--rho-col", type=int, required=True, help="Rho column index")
    parser.add_argument("--already-elev", action="store_true", help="Depth is already elevation, don't multiply by -1")
    
    # Output options
    parser.add_argument("--output-name", type=Path, default=None, help="Output filename")
    parser.add_argument("--compression", type=int, default=4, choices=range(0, 10), 
                       help="Gzip compression level (default: 4)")
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output_name is None:
        stem = args.input_file.stem
        output_file = args.input_file.parent / f"{stem}_sparse.h5"
    else:
        output_file = args.output_name
    
    print("=" * 70)
    print("TOMOGRAPHY DATA FORMATTER (SPARSE NZTM HDF5)")
    print("=" * 70)
    
    # Load
    print(f"\n   Reading {args.input_file.suffix.lower()}...")
    df = load_data(args.input_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Print first few values
    print("\n   First row sample:")
    for col_idx in range(min(8, len(df.columns))):
        val = df.iloc[0, col_idx]
        print(f"     Col {col_idx}: {val}")
    
    # Prepare
    print("\n   Preparing data...")
    print(f"   Using columns: x={args.x_nztm_col}, y={args.y_nztm_col}, depth={args.depth_col}")
    print(f"                  vp={args.vp_col}, vs={args.vs_col}, rho={args.rho_col}")
    
    try:
        df = prepare_data(
            df,
            x_nztm_col_idx=args.x_nztm_col,
            y_nztm_col_idx=args.y_nztm_col,
            depth_col_idx=args.depth_col,
            vp_col_idx=args.vp_col,
            vs_col_idx=args.vs_col,
            rho_col_idx=args.rho_col,
            already_elev=args.already_elev
        )
    except IndexError:
        print(f"\n   ERROR: Column index out of range!")
        print(f"   File has {len(df.columns)} columns (0-{len(df.columns)-1})")
        raise
    
    # Validate NZTM coordinates
    x_range = df['x_nztm'].max() - df['x_nztm'].min()
    y_range = df['y_nztm'].max() - df['y_nztm'].min()
    
    if x_range < 100 or y_range < 100:
        print(f"\n   ERROR: NZTM coordinates too small!")
        print(f"   X range: {x_range:.0f} m, Y range: {y_range:.0f} m")
        raise ValueError("Invalid NZTM coordinates")
    
    print(f"   NZTM X range: {df['x_nztm'].min():.0f} to {df['x_nztm'].max():.0f} m")
    print(f"   NZTM Y range: {df['y_nztm'].min():.0f} to {df['y_nztm'].max():.0f} m")
    print(f"   Elevation range: {df['elevation'].min():.1f} to {df['elevation'].max():.1f} km")
    
    # Write
    write_hdf5_sparse(output_file, df, args.compression)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Next: python interpolate_nztm_tomo_data.py {output_file} --spacing 1.4")
    print("=" * 70)


if __name__ == "__main__":
    main()
