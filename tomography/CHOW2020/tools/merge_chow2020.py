#!/usr/bin/env python3
"""
Merge shallow, crust, and mantle CHOW2020 models into a single HDF5 file.

Handles overlapping depths by preferring the shallowest model.
Internally uses depth convention (positive down) for correct merge logic.
Outputs elevation convention (positive up) in HDF5 groups.

Depth ranges (input):
  - shallow: -2.25 to 8 km depth (includes above-ground to shallow crust)
  - crust: 7 to 50 km depth
  - mantle: 44 to 400 km depth

Elevation groups (output):
  - shallow: 2.25 to -8 km elevation
  - crust: -7 to -50 km elevation
  - mantle: -44 to -400 km elevation

Overlaps (prefer shallowest):
  - shallow/crust: 7-8 km depth = -7 to -8 km elevation (prefer shallow)
  - crust/mantle: 44-50 km depth = -44 to -50 km elevation (prefer crust)
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import xarray as xr


def load_and_prepare_nc(nc_file: Path) -> Dict:
    """Load NetCDF and extract gridded data."""
    print(f"\n   Loading: {nc_file.name}")
    ds = xr.open_dataset(nc_file)
    
    # Identify dimensions
    dim_names = list(ds.sizes.keys())
    
    depth_dim = None
    for name in ['depth', 'z', 'elevation']:
        if name in dim_names:
            depth_dim = name
            break
    
    if depth_dim is None:
        raise ValueError(f"Cannot identify depth dimension. Available: {dim_names}")
    
    # Get depth values (keep as positive for consistency)
    # In CHOW data: positive values = depth below surface
    depth_vals = ds[depth_dim].values.astype(np.float32)
    # Don't negate - keep as positive depth values
    elevations = depth_vals
    
    # Get spatial dimensions
    spatial_dims = [d for d in dim_names if d != depth_dim]
    y_dim = spatial_dims[0] if len(spatial_dims) > 0 else 'y'
    x_dim = spatial_dims[1] if len(spatial_dims) > 1 else 'x'
    
    # Extract coordinates
    y_nztm = ds[y_dim].values.astype(np.float32)
    x_nztm = ds[x_dim].values.astype(np.float32)
    
    # Handle 2D coordinates
    if 'x_nztm' in ds.coords and ds['x_nztm'].ndim == 2:
        x_nztm_2d = ds['x_nztm'].values
        y_nztm_2d = ds['y_nztm'].values
        x_nztm = x_nztm_2d[0, :].astype(np.float32)
        y_nztm = y_nztm_2d[:, 0].astype(np.float32)
    
    # Extract data variables
    data_vars = {}
    for var_name in ['vp', 'vs', 'rho']:
        if var_name in ds.data_vars:
            data_vars[var_name] = ds[var_name].values.astype(np.float32)
    
    print(f"   Depth range: {elevations.min():.2f} to {elevations.max():.2f} km")
    print(f"   Grid: {len(y_nztm)} × {len(x_nztm)} (Y × X)")
    print(f"   Data vars: {list(data_vars.keys())}")
    
    ds.close()
    
    return {
        'elevations': elevations,
        'x_nztm': x_nztm,
        'y_nztm': y_nztm,
        'data': data_vars,
        'depth_dim': depth_dim,
        'y_dim': y_dim,
        'x_dim': x_dim,
        'ny': len(y_nztm),
        'nx': len(x_nztm),
    }


def merge_models(shallow_data: Dict, crust_data: Dict, mantle_data: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Merge three models into single dataset, handling overlaps.
    
    Priority for overlaps:
    - 7-8 km (shallow/crust): prefer shallow
    - 44-50 km (crust/mantle): prefer crust
    """
    
    print("\n   Merging models with overlap handling...")
    
    # Combine all unique depths (sorted, ascending)
    all_depths = np.concatenate([
        shallow_data['elevations'],
        crust_data['elevations'],
        mantle_data['elevations']
    ])
    unique_depths = np.unique(all_depths)
    print(f"   Combined depth levels: {len(unique_depths)}")
    print(f"   Range: {unique_depths.min():.2f} to {unique_depths.max():.2f} km")
    
    # Use shallow/crust grid as base (finer: 617 × 461)
    # Mantle is coarser (155 × 116) - we'll need to handle this
    x_base = shallow_data['x_nztm']
    y_base = shallow_data['y_nztm']
    nx_base = len(x_base)
    ny_base = len(y_base)
    
    print(f"   Using base grid: {ny_base} × {nx_base} (from shallow/crust)")
    
    # Create output arrays
    merged_data = {
        'vp': np.full((len(unique_depths), ny_base, nx_base), -999.0, dtype=np.float32),
        'vs': np.full((len(unique_depths), ny_base, nx_base), -999.0, dtype=np.float32),
        'rho': np.full((len(unique_depths), ny_base, nx_base), -999.0, dtype=np.float32),
    }
    
    # Process each depth level, with priority to shallowest model
    for elev_idx, elev in enumerate(unique_depths):
        source = None
        
        # Check priority (shallowest first)
        # Shallow: -2.25 to 8 km depth (use first, most detailed)
        if -2.25 <= elev <= 8.0:
            shallow_idx = np.where(np.isclose(shallow_data['elevations'], elev, atol=0.01))[0]
            if len(shallow_idx) > 0:
                source = 'shallow'
                data_idx = shallow_idx[0]
                data_src = shallow_data
        
        # Crust: 7 to 50 km depth (use if not found in shallow, overlap at 7-8 km)
        if source is None and 7.0 <= elev <= 50.0:
            crust_idx = np.where(np.isclose(crust_data['elevations'], elev, atol=0.01))[0]
            if len(crust_idx) > 0:
                source = 'crust'
                data_idx = crust_idx[0]
                data_src = crust_data
        
        # Mantle: 44 to 400 km depth (use if not found in crust, overlap at 44-50 km)
        if source is None and 44.0 <= elev <= 400.0:
            mantle_idx = np.where(np.isclose(mantle_data['elevations'], elev, atol=0.01))[0]
            if len(mantle_idx) > 0:
                source = 'mantle'
                data_idx = mantle_idx[0]
                data_src = mantle_data
        
        if source:
            elev_display = -elev  # Convert depth to elevation for display
            print(f"     Elev {elev_display:8.2f} km: {source:8s}", end="")
            
            # Handle grid mismatch (mantle is coarser)
            if data_src['nx'] != nx_base or data_src['ny'] != ny_base:
                print(f" (resampling {data_src['ny']}×{data_src['nx']} → {ny_base}×{nx_base})", end="")
                
                # For mantle: use nearest neighbor resampling
                # Create index mapping
                x_src = data_src['x_nztm']
                y_src = data_src['y_nztm']
                
                for iy, y_val in enumerate(y_base):
                    iy_src = np.argmin(np.abs(y_src - y_val))
                    for ix, x_val in enumerate(x_base):
                        ix_src = np.argmin(np.abs(x_src - x_val))
                        
                        for var in ['vp', 'vs', 'rho']:
                            val = data_src['data'][var][data_idx, iy_src, ix_src]
                            if not np.isnan(val) and val != -999:
                                merged_data[var][elev_idx, iy, ix] = val
            else:
                # Same grid size - direct copy
                for var in ['vp', 'vs', 'rho']:
                    val = data_src['data'][var][data_idx]
                    merged_data[var][elev_idx] = val
            
            print()
        else:
            print(f"     Elev {elev:8.2f} km: NOT FOUND")
    
    return unique_depths, (x_base, y_base), merged_data


def write_merged_h5(output_file: Path, depths: np.ndarray, 
                    coords: Tuple[np.ndarray, np.ndarray], 
                    data: Dict, compression_level: int = 4) -> None:
    """
    Write merged data to HDF5.
    
    Converts depth values (positive down) to elevation values (positive up)
    for the HDF5 group names.
    """
    
    print(f"\n   Writing HDF5: {output_file.name}")
    
    x_nztm, y_nztm = coords
    
    with h5py.File(output_file, 'w') as hf:
        # Store root coordinates
        hf.create_dataset('x_nztm', data=x_nztm, dtype=np.float32)
        hf.create_dataset('y_nztm', data=y_nztm, dtype=np.float32)
        
        # Store metadata about sources
        hf.attrs['source'] = 'Merged CHOW2020 (shallow/crust/mantle)'
        hf.attrs['reference'] = 'Chow et al. (2022), DOI:10.1029/2021JB022865'
        hf.attrs['coordinate_convention'] = 'elevation (positive up)'
        
        # Store each depth level as elevation groups
        for i, depth in enumerate(depths):
            # Convert depth to elevation (negate)
            elevation = -depth
            elev_key = f"{elevation:.1f}".rstrip('0').rstrip('.')
            grp = hf.create_group(elev_key)
            
            for var_name in ['vp', 'vs', 'rho']:
                grp.create_dataset(
                    var_name,
                    data=data[var_name][i],
                    compression='gzip',
                    compression_opts=compression_level
                )
            
            if (i + 1) % 20 == 0 or i == len(depths) - 1:
                print(f"     {i+1}/{len(depths)} elevation levels written")
    
    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n   Output: {output_file}")
    print(f"   Size: {file_size_mb:.1f} MB")
    print(f"   Elevation levels: {len(depths)} (from {-depths[-1]:.2f} to {-depths[0]:.2f} km)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge CHOW2020 shallow/crust/mantle models into single HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Combines three CHOW2020 velocity models with automatic overlap handling:
  - Shallow (-2.25 to 8 km depth): preferred at 7-8 km overlap
  - Crust (7 to 50 km depth): preferred at 44-50 km overlap  
  - Mantle (44 to 400 km depth): used below 50 km

Internally uses depth convention (positive down) for merge logic.
Outputs HDF5 groups with elevation convention (positive up).

HDF5 group names show elevation:
  - "2.25" to "-8" km for shallow
  - "-7" to "-50" km for crust
  - "-44" to "-400" km for mantle

Compatible with map_nztm_tomo.py and other mapping tools.

Example:
  python %(prog)s \\
    -s nz-atom-north-chow-etal-2021-vp+vs-shallow.r0.1-n4_NZTM.nc \\
    -c nz-atom-north-chow-etal-2021-vp+vs-crust.r0.1-n4_NZTM.nc \\
    -m nz-atom-north-chow-etal-2021-vp+vs-mantle.r0.1-n4_NZTM.nc \\
    --output chow_merged.h5
        """
    )
    
    parser.add_argument('-s', '--shallow', type=Path, required=True, help='Shallow model NetCDF')
    parser.add_argument('-c', '--crust', type=Path, required=True, help='Crust model NetCDF')
    parser.add_argument('-m', '--mantle', type=Path, required=True, help='Mantle model NetCDF')
    parser.add_argument('--output', type=Path, default=None, help='Output HDF5 file')
    parser.add_argument('--compression', type=int, default=4, choices=range(0, 10),
                       help='Gzip compression level')
    
    args = parser.parse_args()
    
    # Verify files exist
    for f in [args.shallow, args.crust, args.mantle]:
        if not f.exists():
            print(f"Error: File not found: {f}")
            return
    
    if args.output is None:
        args.output = Path('chow_merged.h5')
    
    print("=" * 70)
    print("MERGE CHOW2020 MODELS (SHALLOW/CRUST/MANTLE)")
    print("=" * 70)
    print("\nOutput coordinate convention: ELEVATION (positive up)")
    print("Groups named by elevation value (e.g., '-7.5' for 7.5 km depth)\n")
    
    try:
        # Load all three models
        print("\n   Loading models...")
        shallow_data = load_and_prepare_nc(args.shallow)
        crust_data = load_and_prepare_nc(args.crust)
        mantle_data = load_and_prepare_nc(args.mantle)
        
        # Merge with overlap handling
        elevations, coords, merged_data = merge_models(shallow_data, crust_data, mantle_data)
        
        # Write output
        write_merged_h5(args.output, elevations, coords, merged_data, args.compression)
        
        print("\n" + "=" * 70)
        print("COMPLETE")
        print(f"Next: python map_nztm_tomo.py {args.output}")
        print("=" * 70)
    
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    main()
