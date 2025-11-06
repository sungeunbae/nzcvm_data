"""
Process tomography data: load from various formats and save to NZTM HDF5.

Optionally interpolates scattered data to a regular NZTM grid.
Supports CSV, Parquet, and NetCDF input formats.

BUGFIX: No-interpolation mode now correctly populates grid cells with point data.
Uses fast direct array indexing (searchsorted) instead of nearest-neighbor matching.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def load_data(input_file: Path) -> pd.DataFrame:
    """Load data from CSV, Parquet, or NetCDF."""
    suffix = input_file.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(input_file)
    elif suffix == '.parquet':
        return pd.read_parquet(input_file)
    elif suffix in ['.nc', '.ncdf', '.netcdf']:
        import xarray as xr
        ds = xr.open_dataset(input_file)
        return ds.to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv, .parquet, or .nc")


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
    
    # Select columns by index
    df_selected = pd.DataFrame({
        'x_nztm': df.iloc[:, x_nztm_col_idx].values,
        'y_nztm': df.iloc[:, y_nztm_col_idx].values,
        'elevation': df.iloc[:, depth_col_idx].values,
        'vp': df.iloc[:, vp_col_idx].values,
        'vs': df.iloc[:, vs_col_idx].values,
        'rho': df.iloc[:, rho_col_idx].values
    })
    
    # Convert depth to elevation if not already elevation (depth positive down -> elevation positive up)
    if not already_elev:
        df_selected['elevation'] = -1 * df_selected['elevation']
    
    return df_selected


def create_grid_from_data(df: pd.DataFrame, spacing_km: float, padding_km: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular NZTM grid from data extent with padding."""
    spacing_m = spacing_km * 1000.0
    padding_m = padding_km * 1000.0
    
    x_min = df['x_nztm'].min() - padding_m
    x_max = df['x_nztm'].max() + padding_m
    y_min = df['y_nztm'].min() - padding_m
    y_max = df['y_nztm'].max() + padding_m
    
    x_nztm = np.arange(x_min, x_max + spacing_m, spacing_m)
    y_nztm = np.arange(y_min, y_max + spacing_m, spacing_m)
    
    print(f"   Grid bounds: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}] m")
    print(f"   Grid size: {len(y_nztm)} x {len(x_nztm)} = {len(x_nztm) * len(y_nztm):,} points")
    print(f"   Grid spacing: {spacing_km} km")
    print(f"   Grid extent: {(x_max - x_min)/1000:.1f} km (X) x {(y_max - y_min)/1000:.1f} km (Y)")
    
    return x_nztm, y_nztm


def interpolate_property(
    df: pd.DataFrame,
    x_nztm: np.ndarray,
    y_nztm: np.ndarray,
    elevations: np.ndarray,
    field_name: str,
    fill_value: float = np.nan
) -> np.ndarray:
    """Interpolate a property from scattered points to a regular NZTM grid."""
    ny, nx, nelev = len(y_nztm), len(x_nztm), len(elevations)
    out = np.full((nelev, ny, nx), np.nan, dtype=np.float32)

    print(f"   Interpolating {field_name}...")

    for iz, elev in enumerate(elevations):
        df_e = df[np.isclose(df["elevation"], elev, atol=1e-3)]
        if df_e.empty:
            continue

        points = np.column_stack([df_e["x_nztm"].values, df_e["y_nztm"].values])
        values = df_e[field_name].values
        
        x_grid, y_grid = np.meshgrid(x_nztm, y_nztm)
        interp = griddata(points, values, (x_grid, y_grid), method="linear", fill_value=fill_value)
        
        n_valid = np.sum(~np.isnan(interp))
        coverage = 100.0 * n_valid / (ny * nx)
        print(f"     Elevation {elev:7.2f} km: {coverage:5.1f}% coverage ({n_valid:,} points)")
        
        out[iz] = interp

    return out


def write_hdf5(
    output_path: str | Path,
    x_nztm: np.ndarray,
    y_nztm: np.ndarray,
    elevations: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    compression_level: int = 4
) -> None:
    """Write data to HDF5."""
    with h5py.File(output_path, "w") as hf:
        # Store coordinates at root level
        hf.create_dataset("x_nztm", data=x_nztm, dtype=np.float32)
        hf.create_dataset("y_nztm", data=y_nztm, dtype=np.float32)
        
        # Store data for each elevation
        for i, elev_val in enumerate(elevations):
            elev_key = f"{elev_val}"
            grp = hf.create_group(elev_key)
            
            # Replace NaN with -999.0 for storage
            vp_store = np.where(np.isnan(vp[i]), -999.0, vp[i]).astype(np.float32)
            vs_store = np.where(np.isnan(vs[i]), -999.0, vs[i]).astype(np.float32)
            rho_store = np.where(np.isnan(rho[i]), -999.0, rho[i]).astype(np.float32)
            
            grp.create_dataset("vp", data=vp_store, compression="gzip", compression_opts=compression_level)
            grp.create_dataset("vs", data=vs_store, compression="gzip", compression_opts=compression_level)
            grp.create_dataset("rho", data=rho_store, compression="gzip", compression_opts=compression_level)
    
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n   Output: {output_path}")
    print(f"   Size: {file_size_mb:.1f} MB")


def main() -> None:
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description="Process tomography data: load and optionally interpolate to NZTM grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just load and save (no interpolation) -> EP2020_NZTM_points.h5
  python %(prog)s input.csv
  
  # Interpolate to grid -> EP2020_NZTM_grid.h5
  python %(prog)s input.csv --interpolate --spacing 1.4
  
  # Custom column indices
  python %(prog)s input.csv --interpolate --spacing 1.4 \\
    --x-nztm-col 0 --y-nztm-col 1 --depth-col 2 --vp-col 3 --vs-col 4 --rho-col 5
  
  # Custom output name
  python %(prog)s input.csv --output-name custom.h5 --interpolate
        """
    )
    parser.add_argument("input_file", type=Path, help="Input file (CSV, Parquet, or NetCDF)")
    
    # Interpolation options
    parser.add_argument("--interpolate", action="store_true", help="Interpolate to regular grid")
    parser.add_argument("--spacing", type=float, default=2.0, metavar="KM", help="Grid spacing in km (default: 2.0)")
    
    # Column mapping by index
    parser.add_argument("--x-nztm-col", type=int, required=True, help="NZTM X coordinate column index")
    parser.add_argument("--y-nztm-col", type=int, required=True, help="NZTM Y coordinate column index")
    parser.add_argument("--depth-col", type=int, required=True, help="Depth column index (will be converted to elevation)")
    parser.add_argument("--vp-col", type=int, required=True, help="Vp column index")
    parser.add_argument("--vs-col", type=int, required=True, help="Vs column index")
    parser.add_argument("--rho-col", type=int, required=True, help="Density column index")
    parser.add_argument("--already-elev", action="store_true", help="Depth column is already elevation (no need to multiply by -1)")


    # Output options
    parser.add_argument("--output-name", type=Path, default=None, help="Output HDF5 filename (default: auto-generated)")
    parser.add_argument("--compression", type=int, default=4, choices=range(0, 10), help="Gzip compression level (default: 4)")
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output_name is None:
        stem = args.input_file.stem
        suffix = "_grid.h5" if args.interpolate else "_points.h5"
        output_file = args.input_file.parent / f"{stem}{suffix}"
    else:
        output_file = args.output_name
    
    print("=" * 70)
    print("TOMOGRAPHY DATA PROCESSOR (NZTM)")
    print("=" * 70)
    
    # Load input
    print(f"\n   Reading {args.input_file.suffix.lower()}...")
    df = load_data(args.input_file)
    print(f"   Loaded {len(df):,} points")
    
    # Print first few rows to help user verify columns
    print("\n   First few rows:")
    for col_idx in range(min(8, len(df.columns))):
        print(f"     Col {col_idx}: {df.iloc[0, col_idx]:.2f} ... {df.iloc[-1, col_idx]:.2f}")
    
    # Prepare data by column indices
    print("\n   Preparing data...")
    print(f"   Using columns: x_col={args.x_nztm_col}, y_col={args.y_nztm_col}, depth_col={args.depth_col}")
    print(f"                  vp_col={args.vp_col}, vs_col={args.vs_col}, rho_col={args.rho_col}")
    
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
    except IndexError as e:
        print(f"\n   ERROR: Column index out of range!")
        print(f"   File has {len(df.columns)} columns (0-{len(df.columns)-1})")
        print(f"   Please verify your --x-nztm-col, --y-nztm-col, --depth-col, --vp-col, --vs-col, --rho-col arguments")
        raise
    
    # Validate NZTM coordinates
    x_range = df['x_nztm'].max() - df['x_nztm'].min()
    y_range = df['y_nztm'].max() - df['y_nztm'].min()
    
    if x_range < 100 or y_range < 100:
        print(f"\n   WARNING: NZTM coordinates appear too small!")
        print(f"   NZTM X range: {df['x_nztm'].min():.0f} to {df['x_nztm'].max():.0f} m (range: {x_range:.0f} m)")
        print(f"   NZTM Y range: {df['y_nztm'].min():.0f} to {df['y_nztm'].max():.0f} m (range: {y_range:.0f} m)")
        print(f"   NZTM coordinates should be in hundreds of thousands.")
        print(f"   Please verify your --x-nztm-col and --y-nztm-col arguments!")
        raise ValueError("Invalid NZTM coordinate range - check column indices")
    
    elevations = sorted(set(df['elevation']))
    print(f"   Found {len(elevations)} elevation levels")
    print(f"   Elevation range: {df['elevation'].min():.1f} to {df['elevation'].max():.1f} km")
    print(f"   NZTM X range: {df['x_nztm'].min():.0f} to {df['x_nztm'].max():.0f} m")
    print(f"   NZTM Y range: {df['y_nztm'].min():.0f} to {df['y_nztm'].max():.0f} m")
    
    if args.interpolate:
        print("\n   Creating grid...")
        x_nztm, y_nztm = create_grid_from_data(df, args.spacing)
        
        print("\n   Interpolating...")
        vp = interpolate_property(df, x_nztm, y_nztm, elevations, "vp")
        vs = interpolate_property(df, x_nztm, y_nztm, elevations, "vs")
        rho = interpolate_property(df, x_nztm, y_nztm, elevations, "rho")
    else:
        # No interpolation - organize points by elevation into grid cells
        print("\n   Organizing data by elevation (no interpolation)...")
        x_nztm = np.array(sorted(df['x_nztm'].unique()))
        y_nztm = np.array(sorted(df['y_nztm'].unique()))
        ny, nx, nelev = len(y_nztm), len(x_nztm), len(elevations)
        
        vp = np.full((nelev, ny, nx), np.nan, dtype=np.float32)
        vs = np.full((nelev, ny, nx), np.nan, dtype=np.float32)
        rho = np.full((nelev, ny, nx), np.nan, dtype=np.float32)
        
        for iz, elev in enumerate(elevations):
            df_e = df[np.isclose(df["elevation"], elev, atol=1e-3)]
            n_points = len(df_e)
            print(f"     Elevation {elev:7.2f} km: {n_points:,} points", end="")
            
            if n_points > 0:
                # Fast direct indexing: use searchsorted to find grid cell indices
                x_indices = np.searchsorted(x_nztm, df_e["x_nztm"].values)
                y_indices = np.searchsorted(y_nztm, df_e["y_nztm"].values)
                
                # Assign values directly to grid cells
                vp[iz, y_indices, x_indices] = df_e["vp"].values
                vs[iz, y_indices, x_indices] = df_e["vs"].values
                rho[iz, y_indices, x_indices] = df_e["rho"].values
                
                print(f" -> {n_points:,} points assigned")
            else:
                print()
    
    print("\n   Writing HDF5...")
    write_hdf5(output_file, x_nztm, y_nztm, elevations, vp, vs, rho, args.compression)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
