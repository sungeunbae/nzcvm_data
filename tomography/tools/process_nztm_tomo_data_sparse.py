#!/usr/bin/env python3
"""
Process sparse tomography measurement data: store points as-is without grid expansion.

For data where measurements are at irregular locations (e.g., 5,066 measurement stations
with multiple depth levels). Stores data efficiently without creating massive empty grids.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_parquet_data_chunked(input_file: Path) -> pd.DataFrame:
    """Load parquet file using row groups (memory efficient)."""
    parquet_file = pq.ParquetFile(input_file)
    chunks = []
    
    print(f"   Loading {input_file.name}...")
    for row_group_idx in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_groups([row_group_idx]).to_pandas()
        chunks.append(chunk)
        if (row_group_idx + 1) % 5 == 0 or row_group_idx == parquet_file.num_row_groups - 1:
            print(f"     Row group {row_group_idx + 1}/{parquet_file.num_row_groups} ({len(chunk):,} rows)")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Total rows loaded: {len(df):,}")
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
    output_path: str | Path,
    df: pd.DataFrame,
    compression_level: int = 4
) -> None:
    """
    Write sparse measurement data to HDF5.
    
    Structure:
    - Root coordinates: x_nztm_unique, y_nztm_unique
    - For each elevation: table with (x_idx, y_idx, vp, vs, rho)
    """
    
    print(f"\n   Preparing sparse HDF5 format...")
    
    # Get unique coordinates
    x_unique = np.sort(df['x_nztm'].unique()).astype(np.float32)
    y_unique = np.sort(df['y_nztm'].unique()).astype(np.float32)
    elevations = sorted(df['elevation'].unique())
    
    print(f"   Unique X coords: {len(x_unique):,}")
    print(f"   Unique Y coords: {len(y_unique):,}")
    print(f"   Unique elevations: {len(elevations):,}")
    print(f"   Total data points: {len(df):,}")
    
    # Create x, y lookup dictionaries for fast indexing
    x_to_idx = {x: i for i, x in enumerate(x_unique)}
    y_to_idx = {y: i for i, y in enumerate(y_unique)}
    
    print(f"\n   Writing HDF5...")
    
    with h5py.File(output_path, "w") as hf:
        # Store coordinate axes
        hf.create_dataset("x_nztm", data=x_unique, dtype=np.float32)
        hf.create_dataset("y_nztm", data=y_unique, dtype=np.float32)
        
        # Store data for each elevation
        for elev in elevations:
            df_e = df[np.isclose(df["elevation"], elev, atol=1e-3)]
            n_points = len(df_e)
            
            print(f"     Elevation {elev:7.2f} km: {n_points:,} points", end="")
            
            if n_points > 0:
                # Create compound dtype for the table
                dt = np.dtype([
                    ('x_idx', np.int32),
                    ('y_idx', np.int32),
                    ('vp', np.float32),
                    ('vs', np.float32),
                    ('rho', np.float32)
                ])
                
                # Create structured array
                data_array = np.zeros(n_points, dtype=dt)
                
                # Fill with data
                for i, (idx, row) in enumerate(df_e.iterrows()):
                    x_idx = x_to_idx[row['x_nztm']]
                    y_idx = y_to_idx[row['y_nztm']]
                    
                    data_array[i]['x_idx'] = x_idx
                    data_array[i]['y_idx'] = y_idx
                    data_array[i]['vp'] = row['vp']
                    data_array[i]['vs'] = row['vs']
                    data_array[i]['rho'] = row['rho']
                
                # Store in HDF5
                elev_key = f"{elev:.1f}"
                grp = hf.create_group(elev_key)
                grp.create_dataset(
                    "data",
                    data=data_array,
                    compression="gzip",
                    compression_opts=compression_level
                )
                print(f" -> Stored as table")
            else:
                print()
    
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n   Output: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")
    
    print(f"\n   To read this sparse HDF5 format:")
    print(f"""
    import h5py
    import numpy as np
    
    with h5py.File('{output_path}', 'r') as f:
        x_nztm = f['x_nztm'][:]
        y_nztm = f['y_nztm'][:]
        
        # Read data at a specific elevation
        data_table = f['-50.0']['data'][:]  # Example elevation
        
        for row in data_table:
            x = x_nztm[row['x_idx']]
            y = y_nztm[row['y_idx']]
            vp = row['vp']
            vs = row['vs']
            rho = row['rho']
            # Use the data
    """)


def main() -> None:
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description="Process sparse tomography measurements: store as point data without grid expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
For scattered measurement data (e.g., 5,000 stations with 200 depth levels each).
Stores data efficiently without creating massive empty grid arrays.

Example:
  python %(prog)s DB2025_NZTM.parquet \\
    --x-nztm-col 13 --y-nztm-col 14 --depth-col 2 \\
    --vp-col 7 --vs-col 8 --rho-col 10
        """
    )
    parser.add_argument("input_file", type=Path, help="Input parquet file")
    
    # Column mapping by index
    parser.add_argument("--x-nztm-col", type=int, required=True, help="NZTM X coordinate column index")
    parser.add_argument("--y-nztm-col", type=int, required=True, help="NZTM Y coordinate column index")
    parser.add_argument("--depth-col", type=int, required=True, help="Depth column index")
    parser.add_argument("--vp-col", type=int, required=True, help="Vp column index")
    parser.add_argument("--vs-col", type=int, required=True, help="Vs column index")
    parser.add_argument("--rho-col", type=int, required=True, help="Density column index")
    parser.add_argument("--already-elev", action="store_true", help="Depth is already elevation")
    
    # Output options
    parser.add_argument("--output-name", type=Path, default=None, help="Output HDF5 filename")
    parser.add_argument("--compression", type=int, default=4, choices=range(0, 10), help="Gzip compression level")
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output_name is None:
        stem = args.input_file.stem
        output_file = args.input_file.parent / f"{stem}_sparse.h5"
    else:
        output_file = args.output_name
    
    print("=" * 70)
    print("SPARSE TOMOGRAPHY DATA PROCESSOR (NZTM)")
    print("=" * 70)
    
    # Load data
    print(f"\n   Reading parquet file...")
    df = load_parquet_data_chunked(args.input_file)
    
    # Prepare
    print(f"\n   Preparing data...")
    print(f"   Using columns: x_col={args.x_nztm_col}, y_col={args.y_nztm_col}, depth_col={args.depth_col}")
    print(f"                  vp_col={args.vp_col}, vs_col={args.vs_col}, rho_col={args.rho_col}")
    
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
    
    # Validate
    x_range = df['x_nztm'].max() - df['x_nztm'].min()
    y_range = df['y_nztm'].max() - df['y_nztm'].min()
    
    if x_range < 100 or y_range < 100:
        print(f"\n   ERROR: NZTM coordinates too small!")
        raise ValueError("Invalid NZTM coordinate range")
    
    print(f"   Found {len(df['elevation'].unique())} elevation levels")
    print(f"   Elevation range: {df['elevation'].min():.1f} to {df['elevation'].max():.1f} km")
    print(f"   NZTM X range: {df['x_nztm'].min():.0f} to {df['x_nztm'].max():.0f} m")
    print(f"   NZTM Y range: {df['y_nztm'].min():.0f} to {df['y_nztm'].max():.0f} m")
    
    # Write
    write_hdf5_sparse(output_file, df, args.compression)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
