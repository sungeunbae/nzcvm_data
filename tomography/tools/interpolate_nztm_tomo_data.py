#!/usr/bin/env python3
"""
Interpolate sparse tomography HDF5 to gridded HDF5 format.

Input: Sparse HDF5 from format_nztm_tomo_data.py
Output: Gridded HDF5 suitable for tomo_nztm_map.py visualization

Gridded format:
  - Root: x_nztm (1D), y_nztm (1D)
  - For each elevation: 2D grids (vp, vs, rho) of shape (ny, nx)
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def load_sparse_hdf5(sparse_file: Path) -> tuple[dict, dict]:
    """
    Load sparse HDF5 data.
    
    Returns: (coordinates_dict, data_dict)
      coordinates_dict: {'x_nztm': array, 'y_nztm': array}
      data_dict: {elevation: dataframe}
    """
    coordinates = {}
    data_by_elev = {}
    
    with h5py.File(sparse_file, "r") as f:
        # Load coordinate arrays
        coordinates['x_nztm'] = f['x_nztm'][:].astype(float)
        coordinates['y_nztm'] = f['y_nztm'][:].astype(float)
        
        # Load data for each elevation
        for elev_key in sorted(f.keys()):
            if elev_key not in ['x_nztm', 'y_nztm']:
                grp = f[elev_key]
                if 'data' in grp:
                    data_table = grp['data'][:]
                    
                    # Convert structured array to dataframe
                    x_coords = coordinates['x_nztm'][data_table['x_idx']]
                    y_coords = coordinates['y_nztm'][data_table['y_idx']]
                    
                    df = pd.DataFrame({
                        'x_nztm': x_coords,
                        'y_nztm': y_coords,
                        'vp': data_table['vp'],
                        'vs': data_table['vs'],
                        'rho': data_table['rho']
                    })
                    
                    elev = float(elev_key)
                    data_by_elev[elev] = df
    
    return coordinates, data_by_elev


def create_grid_from_data(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    spacing_km: float,
    padding_km: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular NZTM grid from data extent."""
    spacing_m = spacing_km * 1000.0
    padding_m = padding_km * 1000.0
    
    x_min = x_coords.min() - padding_m
    x_max = x_coords.max() + padding_m
    y_min = y_coords.min() - padding_m
    y_max = y_coords.max() + padding_m
    
    x_nztm = np.arange(x_min, x_max + spacing_m, spacing_m)
    y_nztm = np.arange(y_min, y_max + spacing_m, spacing_m)
    
    print(f"   Grid bounds: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}] m")
    print(f"   Grid size: {len(y_nztm)} x {len(x_nztm)} = {len(x_nztm) * len(y_nztm):,} nodes")
    print(f"   Grid spacing: {spacing_km} km")
    print(f"   Grid extent: {(x_max - x_min)/1000:.1f} km (X) x {(y_max - y_min)/1000:.1f} km (Y)")
    
    return x_nztm, y_nztm


def interpolate_property(
    data_by_elev: dict,
    x_nztm: np.ndarray,
    y_nztm: np.ndarray,
    field_name: str,
    method: str = "linear"
) -> tuple[np.ndarray, list]:
    """
    Interpolate a property to grid for all elevations.
    
    Returns: (data_array, elevations_list)
      data_array: shape (n_elev, ny, nx)
      elevations_list: sorted elevations
    """
    elevations = sorted(data_by_elev.keys())
    ny, nx, nelev = len(y_nztm), len(x_nztm), len(elevations)
    out = np.full((nelev, ny, nx), np.nan, dtype=np.float32)
    
    print(f"   Interpolating {field_name} ({method}) to grid...")
    
    x_grid, y_grid = np.meshgrid(x_nztm, y_nztm)
    grid_points = (x_grid, y_grid)
    
    for iz, elev in enumerate(elevations):
        df_e = data_by_elev[elev]
        
        if len(df_e) == 0:
            print(f"     Elevation {elev:7.2f} km: No data")
            continue
        
        # Create point array for interpolation
        points = np.column_stack([df_e['x_nztm'].values, df_e['y_nztm'].values])
        values = df_e[field_name].values
        
        # Interpolate
        interp = griddata(points, values, grid_points, method=method, fill_value=np.nan)
        
        n_valid = np.sum(~np.isnan(interp))
        coverage = 100.0 * n_valid / (ny * nx)
        n_input = len(df_e)
        print(f"     Elevation {elev:7.2f} km: {n_input:6,} input points -> {coverage:5.1f}% coverage")
        
        out[iz] = interp
    
    return out, elevations


def write_hdf5_gridded(
    output_path: Path,
    x_nztm: np.ndarray,
    y_nztm: np.ndarray,
    elevations: list,
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    compression_level: int = 4
) -> None:
    """Write gridded data to HDF5."""
    
    print(f"\n   Writing gridded HDF5...")
    
    with h5py.File(output_path, "w") as hf:
        # Store coordinates at root
        hf.create_dataset("x_nztm", data=x_nztm, dtype=np.float32)
        hf.create_dataset("y_nztm", data=y_nztm, dtype=np.float32)
        
        # Store gridded data for each elevation
        for i, elev_val in enumerate(elevations):
            elev_key = f"{elev_val:.1f}"
            grp = hf.create_group(elev_key)
            
            # Replace NaN with -999.0 for storage
            vp_store = np.where(np.isnan(vp[i]), -999.0, vp[i]).astype(np.float32)
            vs_store = np.where(np.isnan(vs[i]), -999.0, vs[i]).astype(np.float32)
            rho_store = np.where(np.isnan(rho[i]), -999.0, rho[i]).astype(np.float32)
            
            grp.create_dataset("vp", data=vp_store, compression="gzip", 
                             compression_opts=compression_level)
            grp.create_dataset("vs", data=vs_store, compression="gzip", 
                             compression_opts=compression_level)
            grp.create_dataset("rho", data=rho_store, compression="gzip", 
                             compression_opts=compression_level)
    
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"   Output: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")


def main() -> None:
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description="Interpolate sparse HDF5 to gridded HDF5 for mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Takes sparse HDF5 from format_nztm_tomo_data.py and interpolates to a regular
grid suitable for visualization with tomo_nztm_map.py.

Examples:
  # Default interpolation (linear, 2 km spacing)
  python %(prog)s data_sparse.h5
  
  # Custom grid spacing
  python %(prog)s data_sparse.h5 --spacing 1.4
  
  # Nearest-neighbor interpolation (faster, less smooth)
  python %(prog)s data_sparse.h5 --spacing 2.0 --method nearest
  
  # Custom output name
  python %(prog)s data_sparse.h5 --output-name my_grid.h5
        """
    )
    parser.add_argument("sparse_file", type=Path, help="Input sparse HDF5 file")
    parser.add_argument("--spacing", type=float, default=2.0, metavar="KM",
                       help="Grid spacing in km (default: 2.0)")
    parser.add_argument("--method", type=str, default="linear", 
                       choices=["linear", "nearest", "cubic"],
                       help="Interpolation method (default: linear)")
    parser.add_argument("--padding", type=float, default=10.0, metavar="KM",
                       help="Padding around data extent in km (default: 10.0)")
    parser.add_argument("--output-name", type=Path, default=None, help="Output filename")
    parser.add_argument("--compression", type=int, default=4, choices=range(0, 10),
                       help="Gzip compression level (default: 4)")
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output_name is None:
        stem = args.sparse_file.stem.replace('_sparse', '')
        output_file = args.sparse_file.parent / f"{stem}_grid.h5"
    else:
        output_file = args.output_name
    
    print("=" * 70)
    print("TOMOGRAPHY DATA INTERPOLATOR (SPARSE -> GRID NZTM HDF5)")
    print("=" * 70)
    
    # Load sparse data
    print(f"\n   Reading sparse HDF5...")
    coords, data_by_elev = load_sparse_hdf5(args.sparse_file)
    
    x_unique = coords['x_nztm']
    y_unique = coords['y_nztm']
    elevations_input = sorted(data_by_elev.keys())
    
    print(f"   Loaded {len(x_unique):,} unique X coordinates")
    print(f"   Loaded {len(y_unique):,} unique Y coordinates")
    print(f"   Loaded {len(elevations_input)} elevation levels")
    
    total_points = sum(len(df) for df in data_by_elev.values())
    print(f"   Total data points: {total_points:,}")
    
    print(f"\n   Data range:")
    all_x = np.concatenate([df['x_nztm'].values for df in data_by_elev.values()])
    all_y = np.concatenate([df['y_nztm'].values for df in data_by_elev.values()])
    print(f"   NZTM X: {all_x.min():.0f} to {all_x.max():.0f} m")
    print(f"   NZTM Y: {all_y.min():.0f} to {all_y.max():.0f} m")
    print(f"   Elevations: {elevations_input[0]:.1f} to {elevations_input[-1]:.1f} km")
    
    # Create regular grid
    print(f"\n   Creating regular grid...")
    x_grid, y_grid = create_grid_from_data(
        all_x, all_y, 
        spacing_km=args.spacing,
        padding_km=args.padding
    )
    
    # Interpolate
    print(f"\n   Interpolating data...")
    vp, elevs = interpolate_property(data_by_elev, x_grid, y_grid, "vp", args.method)
    vs, _ = interpolate_property(data_by_elev, x_grid, y_grid, "vs", args.method)
    rho, _ = interpolate_property(data_by_elev, x_grid, y_grid, "rho", args.method)
    
    # Write
    write_hdf5_gridded(output_file, x_grid, y_grid, elevs, vp, vs, rho, args.compression)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Next: python tomo_nztm_map.py {output_file} --scalar vs")
    print("=" * 70)


if __name__ == "__main__":
    main()
