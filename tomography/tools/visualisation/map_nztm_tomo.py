#!/usr/bin/env python3
"""
2D Map Viewer for NZTM Tomography Data

Specifically designed for NZTM coordinate system HDF5 files.
- HDF5 files must have x_nztm and y_nztm at root level
- CSV files must provide x_nztm and y_nztm columns
- All coordinates automatically converted to WGS84 for plotting (0-360deg longitude)

Usage:
    map_nztm_tomo.py h5file1 [--compared h5file2] \
                     [--with-csv csvfile --x-nztm-col X --y-nztm-col Y --depth-col D --scalar-col S] \
                     [--csv-only csvfile --x-nztm-col X --y-nztm-col Y --depth-col D --scalar-col S] \
                     [--scalar {vp,vs,rho}] [--vmin VMIN] [--vmax VMAX] [--cmap CMAP] \
                     [--elevations ELEVATIONS ...] [--output-dir DIR] [--no-cartopy] [--dpi DPI]

Modes:
  1. Standard: Plot scalar data from h5file1
  2. Ratio: Provide --compared h5file2 to plot ln(h5file2 / h5file1)
  3. Overlay: Provide --with-csv to overlay CSV points on HDF5 data
  4. CSV-only: Provide --csv-only to plot only CSV data
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer
from pyproj import Transformer
from scipy.spatial import KDTree

from qcore import cli

warnings.filterwarnings("ignore", message="The behavior of DatetimeProperties.to_pydatetime is deprecated", 
                       category=FutureWarning)

# Optional Cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: Cartopy not found. Falling back to basic matplotlib plots.")


# ----------------------------
# NZTM Coordinate Conversion
# ----------------------------
def nztm_to_wgs84(x_nztm: np.ndarray, y_nztm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert NZTM (EPSG:2193) to WGS84 with longitude in 0-360deg range.

    Parameters
    ----------
    x_nztm : np.ndarray
        NZTM X coordinates (meters), 1D array
    y_nztm : np.ndarray
        NZTM Y coordinates (meters), 1D array

    Returns
    -------
    tuple
        (lat_grid, lon_grid) both as 2D arrays (ny, nx) where lon is in 0-360 range
        
    Note
    ----
    Returns 2D arrays to properly handle projection warping. NZTM grid is rectilinear
    in NZTM space but slightly warped in WGS84 space.
    """
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    
    # Create meshgrid for transformation
    x_grid, y_grid = np.meshgrid(x_nztm, y_nztm)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
    
    # Convert longitude to 0-360 range
    lon_grid = np.where(lon_grid < 0, lon_grid + 360, lon_grid)
    
    # Return 2D arrays to properly handle projection warping
    return lat_grid, lon_grid


def lons_centered_180(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes to [-180, 180) range for PlateCarree(lon_0=180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon - 180.0 + 180.0) % 360.0) - 180.0


def choose_projection_and_extent(lats: np.ndarray, lons: np.ndarray, 
                                  csv_lats: Optional[np.ndarray] = None,
                                  csv_lons: Optional[np.ndarray] = None) -> Tuple[object, object, Tuple[float, float, float, float], object]:
    """
    Determine projection and extent for plotting.
    Include CSV points in extent calculation if provided.
    
    Returns (ax_crs, data_crs, extent, extent_crs).
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    
    # Combine HDF5 and CSV coordinates for extent calculation
    if csv_lats is not None and csv_lons is not None and len(csv_lats) > 0:
        all_lats = np.concatenate([lats.ravel(), csv_lats.ravel()])
        all_lons = np.concatenate([lons.ravel(), csv_lons.ravel()])
    else:
        all_lats = lats
        all_lons = lons
    
    data_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
    
    pad_lon = 2.0
    pad_lat = 2.0
    minlat = max(-90.0, float(np.nanmin(all_lats)) - pad_lat)
    maxlat = min(90.0, float(np.nanmax(all_lats)) + pad_lat)
    
    lon_min = float(np.nanmin(all_lons))
    lon_max = float(np.nanmax(all_lons))
    
    # Check if crosses dateline
    has_over_180 = lon_max > 180.0
    has_negative = lon_min < 0.0
    crosses_seam = (lon_max - lon_min) > 180.0 or has_over_180 or has_negative
    
    eps = 1e-6
    
    if crosses_seam and HAS_CARTOPY:
        # Center on 180deg longitude
        ax_crs = ccrs.PlateCarree(central_longitude=180)
        lons_ax = lons_centered_180(all_lons)
        minlon_ax = float(np.nanmin(lons_ax))
        maxlon_ax = float(np.nanmax(lons_ax))
        minlon = max(-180.0 + eps, minlon_ax - pad_lon)
        maxlon = min(180.0 - eps, maxlon_ax + pad_lon)
        extent = (minlon, maxlon, minlat, maxlat)
        extent_crs = ax_crs
    else:
        # Standard PlateCarree
        ax_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
        minlon = max(-180.0 + eps, lon_min - pad_lon)
        maxlon = min(360.0 - eps, lon_max + pad_lon)
        extent = (minlon, maxlon, minlat, maxlat)
        extent_crs = data_crs
    
    return ax_crs, data_crs, extent, extent_crs


# ----------------------------
# Load HDF5 data
# ----------------------------
def load_h5_data(h5file: Path, elevation: str, scalar: str, 
                 mask_values: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NZTM tomography data from HDF5 file.
    
    Expected structure:
    - Root level: x_nztm (1D), y_nztm (1D)
    - Groups by elevation: vp, vs, rho (2D: ny x nx)
    
    Returns lat_grid, lon_grid (2D arrays converted to WGS84 0-360deg), and data.
    """
    with h5py.File(h5file, "r") as f:
        # Load NZTM coordinates from root
        if 'x_nztm' not in f or 'y_nztm' not in f:
            raise ValueError(f"File {h5file} missing x_nztm or y_nztm at root level. "
                           "This script requires NZTM coordinate format.")
        
        x_nztm = f['x_nztm'][:]
        y_nztm = f['y_nztm'][:]
        
        # Convert to WGS84 (returns 2D arrays)
        lat_grid, lon_grid = nztm_to_wgs84(x_nztm, y_nztm)
        
        # Load scalar data
        grp = f[elevation]
        data = grp[scalar][:].astype(float)
    
    # Apply masking
    mask_values_list = mask_values if mask_values else []
    data[data == -999.0] = np.nan
    for mask_val in mask_values_list:
        if not np.isnan(mask_val):
            data[data == mask_val] = np.nan
    
    return lat_grid, lon_grid, data


def load_csv_data_for_elevation(
    csvfile: Path,
    target_elevation: float,
    x_nztm_col: int,
    y_nztm_col: int,
    depth_col: int,
    scalar_col: int,
    depth_is_elevation: bool,
    depth_tolerance: float,
    skip_rows: int,
    sep: str
) -> pd.DataFrame:
    """
    Load CSV data for a specific elevation with NZTM coordinates.
    Converts NZTM to WGS84 (0-360deg longitude).

    Returns DataFrame with columns: lat, lon, scalar, depth
    """
    df = pd.read_csv(csvfile, sep=sep, skiprows=skip_rows, header=None)
    
    x_nztm = df.iloc[:, x_nztm_col].values
    y_nztm = df.iloc[:, y_nztm_col].values
    depth_vals = df.iloc[:, depth_col].values
    scalar_vals = df.iloc[:, scalar_col].values
    
    # Convert NZTM to WGS84
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    lon_vals, lat_vals = transformer.transform(x_nztm, y_nztm)
    lon_vals = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
    
    # Convert depth to elevation if needed
    elevation_vals = depth_vals if depth_is_elevation else -depth_vals
    
    # Filter by elevation tolerance
    mask = np.abs(elevation_vals - target_elevation) <= depth_tolerance
    
    return pd.DataFrame({
        'lat': lat_vals[mask],
        'lon': lon_vals[mask],
        'scalar': scalar_vals[mask],
        'depth': depth_vals[mask]
    })


# ----------------------------
# Create map plot
# ----------------------------
def create_map_plot(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    elevation: str,
    scalar: str,
    vmin: float,
    vmax: float,
    cmap: str,
    use_cartopy: bool,
    is_ratio: bool = False,
    csv_overlay_data: Optional[pd.DataFrame] = None,
    base_filename: str = "",
    compared_filename: str = "",
    csv_only: bool = False,
    no_outline_marker: bool = False,
    marker_size: float = 5.0,
    marker_zorder: int = 2
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a map plot with optional CSV overlay."""
    
    # Get CSV coordinates for extent calculation
    csv_lats = csv_overlay_data['lat'].values if csv_overlay_data is not None and not csv_overlay_data.empty else None
    csv_lons = csv_overlay_data['lon'].values if csv_overlay_data is not None and not csv_overlay_data.empty else None
    
    if use_cartopy and HAS_CARTOPY:
        ax_crs, data_crs, extent, extent_crs = choose_projection_and_extent(lat, lon, csv_lats, csv_lons)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ax_crs)
        ax.set_extent(extent, crs=extent_crs)
        
        # Add features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        data_crs = None
    
    # Plot HDF5 data if not CSV-only
    if not csv_only:
        # lon and lat are already 2D arrays from nztm_to_wgs84, use them directly
        if use_cartopy and HAS_CARTOPY:
            im = ax.pcolormesh(lon, lat, data, transform=data_crs,
                              cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        else:
            im = ax.pcolormesh(lon, lat, data,
                              cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        if is_ratio:
            cbar.set_label(f'ln({scalar}_2 / {scalar}_1)', fontsize=12)
        else:
            unit = 'km/s' if scalar in ['vp', 'vs'] else 'g/cm^3'
            cbar.set_label(f'{scalar.upper()} [{unit}]', fontsize=12)
    
    # Overlay CSV data
    if csv_overlay_data is not None and not csv_overlay_data.empty:
        csv_lons = csv_overlay_data['lon'].values
        csv_lats = csv_overlay_data['lat'].values
        csv_vals = csv_overlay_data['scalar'].values
        
        edge_color = 'none' if no_outline_marker else 'black'
        edge_width = 0 if no_outline_marker else 0.5
        
        if csv_only:
            if use_cartopy and HAS_CARTOPY:
                sc = ax.scatter(csv_lons, csv_lats, c=csv_vals, cmap=cmap,
                               vmin=vmin, vmax=vmax, s=marker_size,
                               edgecolors=edge_color, linewidths=edge_width,
                               transform=data_crs, zorder=marker_zorder)
            else:
                sc = ax.scatter(csv_lons, csv_lats, c=csv_vals, cmap=cmap,
                               vmin=vmin, vmax=vmax, s=marker_size,
                               edgecolors=edge_color, linewidths=edge_width, zorder=marker_zorder)
            
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label(f'CSV {scalar.upper()}', fontsize=12)
        else:
            if use_cartopy and HAS_CARTOPY:
                ax.scatter(csv_lons, csv_lats, c=csv_vals, cmap=cmap,
                          vmin=vmin, vmax=vmax, s=marker_size,
                          edgecolors=edge_color, linewidths=edge_width,
                          transform=data_crs, zorder=marker_zorder, alpha=0.8)
            else:
                ax.scatter(csv_lons, csv_lats, c=csv_vals, cmap=cmap,
                          vmin=vmin, vmax=vmax, s=marker_size,
                          edgecolors=edge_color, linewidths=edge_width,
                          zorder=marker_zorder, alpha=0.8)


    # Title
    elev_float = float(elevation.replace("_", "."))
    if is_ratio:
        title = f'ln({scalar.upper()}) Ratio: {compared_filename} / {base_filename}\nElevation: {elev_float:.1f} km'
    elif csv_only:
        title = f'CSV Data: {scalar.upper()}\nElevation: {elev_float:.1f} km'
    else:
        title = f'{scalar.upper()} from {base_filename}'
        if csv_overlay_data is not None and not csv_overlay_data.empty:
            title += f' + CSV ({len(csv_overlay_data)} points)'
        title += f'\nElevation: {elev_float:.1f} km'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if not (use_cartopy and HAS_CARTOPY):
        ax.set_xlabel('Longitude [deg]', fontsize=12)
        ax.set_ylabel('Latitude [deg]', fontsize=12)
        ax.set_aspect('equal')
    
    return fig, ax


# ----------------------------
# Main function
# ----------------------------
app = typer.Typer(add_completion=False)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    h5file1: Optional[Path] = typer.Argument(None, help="Primary HDF5 file (NZTM format)"),
    compared: Optional[Path] = typer.Option(None, "--compared", help="Second HDF5 for ratio comparison"),
    with_csv: Optional[Path] = typer.Option(None, "--with-csv", help="CSV file for overlay"),
    csv_only_file: Optional[Path] = typer.Option(None, "--csv-only", help="CSV file for CSV-only mode"),
    x_nztm_col: Optional[int] = typer.Option(None, "--x-nztm-col", help="X NZTM column index"),
    y_nztm_col: Optional[int] = typer.Option(None, "--y-nztm-col", help="Y NZTM column index"),
    depth_col: Optional[int] = typer.Option(None, "--depth-col", help="Depth column index"),
    depth_is_elevation: bool = typer.Option(False, "--depth-is-elevation", help="Depth is elevation (positive up)"),
    scalar_col: Optional[int] = typer.Option(None, "--scalar-col", help="Scalar column index"),
    depth_tolerance: float = typer.Option(0.1, "--depth-tolerance", help="Depth tolerance (km)"),
    elev_tolerance: float = typer.Option(0.01, "--elev-tolerance", help="Elevation tolerance (km) for matching --elevations"),
    lonlat_tolerance: float = typer.Option(1e-6, "--lonlat-tolerance", help="Lon/lat tolerance (degrees)"),
    skip_rows: int = typer.Option(0, "--skip-rows", help="Rows to skip in CSV"),
    sep: str = typer.Option(",", "--sep", help="CSV delimiter"),
    scalar: str = typer.Option("vs", "--scalar", help="Scalar field: vp, vs, or rho"),
    vmin: Optional[float] = typer.Option(None, "--vmin", help="Colorbar minimum"),
    vmax: Optional[float] = typer.Option(None, "--vmax", help="Colorbar maximum"),
    cmap: Optional[str] = typer.Option(None, "--cmap", help="Colormap name"),
    elevations: Optional[List[str]] = typer.Option(None, "--elevations", help="Elevations to plot"),
    auto_elevations: bool = typer.Option(False, "--auto-elevations", help="Auto-detect from CSV"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory"),
    no_cartopy: bool = typer.Option(False, "--no-cartopy", help="Disable cartopy"),
    no_outline_marker: bool = typer.Option(False, "--no-outline-marker", help="Remove CSV marker outlines"),
    marker_size: float = typer.Option(5.0, "--marker-size", help="CSV marker size"),
    marker_zorder: int = typer.Option(2, "--marker-zorder", help="CSV marker z-order"),
    dpi: int = typer.Option(150, "--dpi", help="Output DPI"),
    mask_value: Optional[List[float]] = typer.Option(None, "--mask-value", help="Additional mask values"),
    limits_mode: str = typer.Option("global", "--limits-mode", help="Color limits: global or local"),
    diff_tolerance: Optional[float] = typer.Option(None, "--diff-tolerance", help="Difference tolerance"),
):
    """Plot 2D tomography maps from NZTM coordinate HDF5 files."""
    
    # Validate modes
    csv_only_mode = csv_only_file is not None
    is_overlay_mode = with_csv is not None
    is_ratio_mode = compared is not None
    
    if csv_only_mode and (is_overlay_mode or is_ratio_mode or h5file1):
        cli.print_error_and_exit("--csv-only cannot be used with h5file1, --compared, or --with-csv")
    
    if is_ratio_mode and is_overlay_mode:
        cli.print_error_and_exit("Cannot use both --compared and --with-csv")
    
    if not csv_only_mode and h5file1 is None:
        cli.print_error_and_exit("Must provide h5file1 or use --csv-only mode")
    
    # Validate CSV arguments
    if is_overlay_mode or csv_only_mode:
        if x_nztm_col is None or y_nztm_col is None:
            cli.print_error_and_exit("CSV mode requires --x-nztm-col and --y-nztm-col")
        if depth_col is None or scalar_col is None:
            cli.print_error_and_exit("CSV mode requires --depth-col and --scalar-col")
    
    csv_file = csv_only_file if csv_only_mode else with_csv
    
    if cmap is None:
        cmap = "seismic" if is_ratio_mode else "RdYlBu_r"
    
    use_cartopy = not no_cartopy and HAS_CARTOPY
    use_fixed_limits = vmin is not None and vmax is not None
    mask_values_list = mask_value if mask_value else []
    
    # Output directory
    if output_dir is None:
        if h5file1:
            base_dir = h5file1.parent
            suffix = "_ln_ratio" if is_ratio_mode else ("_overlay" if is_overlay_mode else "")
            output_dir = base_dir / f"tomo_maps{suffix}"
        else:
            output_dir = Path("tomo_maps_csv_only")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("2D TOMOGRAPHY MAP VIEWER (NZTM)")
    print(f"{'='*70}\n")
    
    if is_ratio_mode:
        print(f"[*] Mode: Ratio (ln(file2/file1))")
        print(f"   Base: {h5file1.name}")
        print(f"   Compared: {compared.name}")
    elif csv_only_mode:
        print(f"[*] Mode: CSV-only")
        print(f"   CSV: {csv_file.name}")
    elif is_overlay_mode:
        print(f"[*] Mode: HDF5 + CSV overlay")
        print(f"   HDF5: {h5file1.name}")
        print(f"   CSV: {csv_file.name}")
    else:
        print(f"[*] Mode: Standard HDF5")
        print(f"   File: {h5file1.name}")
    
    print(f"   Scalar: {scalar.upper()}")
    print(f"   Output: {output_dir}")
    
    # Determine elevations
    if not csv_only_mode:
        with h5py.File(h5file1, "r") as f:
            available_elevs = sorted([k for k in f.keys() if k not in ['x_nztm', 'y_nztm']], 
                                   key=lambda x: float(x.replace("_", ".")))
        
        if elevations:
            user_elevs = [float(e) for e in elevations]
            elevs_to_plot = []
            for avail in available_elevs:
                avail_float = float(avail.replace("_", "."))
                if any(abs(avail_float - ue) <= elev_tolerance for ue in user_elevs):
                    elevs_to_plot.append(avail)
            if not elevs_to_plot:
                cli.print_error_and_exit(f"No elevations matched within tolerance {elev_tolerance}. Available: {available_elevs}")
        else:
            elevs_to_plot = available_elevs
        
        print(f"\n   Elevations: {len(elevs_to_plot)}")
    else:
        if auto_elevations:
            df_temp = pd.read_csv(csv_file, sep=sep, skiprows=skip_rows, header=None)
            depth_vals = df_temp.iloc[:, depth_col].values
            elev_vals = depth_vals if depth_is_elevation else -depth_vals
            unique_elevs = np.unique(elev_vals)
            elevs_to_plot = [f"{e:.0f}" if e == int(e) else f"{e:.1f}" for e in unique_elevs]
            print(f"\n   Auto-detected {len(elevs_to_plot)} elevations")
        elif elevations:
            elevs_to_plot = elevations
        else:
            cli.print_error_and_exit("CSV-only mode requires --elevations or --auto-elevations")
    
    # Global color limits
    global_vmin, global_vmax = None, None
    if limits_mode == "global" and not use_fixed_limits:
        print("\n[*] Computing global color limits...")
        all_vals = []
        
        for elev in elevs_to_plot:
            try:
                if not csv_only_mode:
                    _, _, data = load_h5_data(h5file1, elev, scalar, mask_values_list)
                    if is_ratio_mode:
                        _, _, data2 = load_h5_data(compared, elev, scalar, mask_values_list)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            valid_mask = (data > 0) & (data2 > 0) & ~np.isnan(data) & ~np.isnan(data2)
                            if np.any(valid_mask):
                                ratios = np.log(data2[valid_mask] / data[valid_mask])
                                all_vals.extend(ratios[np.isfinite(ratios)])
                    else:
                        all_vals.extend(data[~np.isnan(data)])
                
                if csv_file:
                    target_elev = float(elev.replace('_', '.'))
                    csv_data = load_csv_data_for_elevation(
                        csv_file, target_elev, x_nztm_col, y_nztm_col,
                        depth_col, scalar_col, depth_is_elevation,
                        depth_tolerance, skip_rows, sep
                    )
                    if not csv_data.empty:
                        all_vals.extend(csv_data['scalar'].values)
            except:
                continue
        
        if all_vals:
            all_vals = np.array(all_vals)
            if is_ratio_mode:
                max_abs = np.nanmax(np.abs(all_vals))
                global_vmin, global_vmax = -max_abs, max_abs
            else:
                global_vmin = np.percentile(all_vals, 2.0)
                global_vmax = np.percentile(all_vals, 98.0)
            print(f"   Range: [{global_vmin:.3f}, {global_vmax:.3f}]")
    
    # Plot each elevation
    all_diffs = []
    print(f"\n[*] Generating maps...\n")

    for elev in elevs_to_plot:
        elev_str = elev.replace("_", ".")
        print(f"   Processing {elev_str} km...")
        
        try:
            # Load HDF5 data
            if not csv_only_mode:
                lat, lon, data1 = load_h5_data(h5file1, elev, scalar, mask_values_list)
                
                if is_ratio_mode:
                    _, _, data2 = load_h5_data(compared, elev, scalar, mask_values_list)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.log(data2 / data1)
                        ratio[~np.isfinite(ratio)] = np.nan
                    plot_data = ratio
                else:
                    plot_data = data1
            
            # Load CSV data
            csv_data_slice = None
            if csv_file:
                target_elev = float(elev.replace('_', '.'))
                csv_data_slice = load_csv_data_for_elevation(
                    csv_file, target_elev, x_nztm_col, y_nztm_col,
                    depth_col, scalar_col, depth_is_elevation,
                    depth_tolerance, skip_rows, sep
                )
                if csv_data_slice is not None and not csv_data_slice.empty:
                    print(f"      CSV: {len(csv_data_slice)} points")
            
            # For CSV-only, create extent from CSV
            if csv_only_mode and csv_data_slice is not None and not csv_data_slice.empty:
                lat = np.array([csv_data_slice['lat'].min(), csv_data_slice['lat'].max()])
                lon = np.array([csv_data_slice['lon'].min(), csv_data_slice['lon'].max()])
                plot_data = np.zeros((2, 2))
            
            # Compute differences in overlay mode
            if is_overlay_mode and csv_data_slice is not None and not csv_data_slice.empty:
                # lon and lat are already 2D, create flattened point coordinates
                points = np.column_stack((lon.ravel(), lat.ravel()))
                tree = KDTree(points)
                query_points = csv_data_slice[['lon', 'lat']].values
                dists, idxs = tree.query(query_points)
                mask = dists < lonlat_tolerance
                num_common = np.sum(mask)
                
                if num_common > 0:
                    print(f"      Common: {num_common} points")
                    valid_h5 = plot_data.ravel()[idxs[mask]]
                    valid_csv = csv_data_slice['scalar'].values[mask]
                    diffs = valid_h5 - valid_csv
                    
                    for lon_p, lat_p, h5_p, csv_p, diff_p in zip(
                        csv_data_slice['lon'].values[mask],
                        csv_data_slice['lat'].values[mask],
                        valid_h5, valid_csv, diffs
                    ):
                        all_diffs.append({
                            'elevation': float(elev.replace('_', '.')),
                            'lon': lon_p, 'lat': lat_p,
                            'scalar_h5': h5_p, 'scalar_csv': csv_p,
                            'difference': diff_p
                        })
            
            # Determine color limits
            if use_fixed_limits:
                plot_vmin, plot_vmax = vmin, vmax
            elif limits_mode == "global":
                plot_vmin, plot_vmax = global_vmin, global_vmax
            else:  # local
                if csv_only_mode and csv_data_slice is not None:
                    vals = csv_data_slice['scalar'].values
                    plot_vmin = np.percentile(vals, 2.0)
                    plot_vmax = np.percentile(vals, 98.0)
                elif is_ratio_mode:
                    max_abs = np.nanmax(np.abs(plot_data[np.isfinite(plot_data)]))
                    plot_vmin, plot_vmax = -max_abs, max_abs
                else:
                    vals = plot_data[~np.isnan(plot_data)]
                    plot_vmin = np.percentile(vals, 2.0)
                    plot_vmax = np.percentile(vals, 98.0)
            
            # Create plot
            fig, ax = create_map_plot(
                lat, lon, plot_data, elevation=elev, scalar=scalar,
                vmin=plot_vmin, vmax=plot_vmax, cmap=cmap, use_cartopy=use_cartopy,
                is_ratio=is_ratio_mode, csv_overlay_data=csv_data_slice,
                base_filename=h5file1.name if h5file1 else "csv",
                compared_filename=compared.name if compared else "",
                csv_only=csv_only_mode, no_outline_marker=no_outline_marker,
                marker_size=marker_size, marker_zorder=marker_zorder
            )
            
            # Save
            fname_elev = elev.replace("_", ".")
            if is_ratio_mode:
                outfile = output_dir / f"ln_ratio_{scalar}_elev{fname_elev}.png"
            elif csv_only_mode:
                outfile = output_dir / f"csv_only_{scalar}_elev{fname_elev}.png"
            elif is_overlay_mode:
                outfile = output_dir / f"overlay_{scalar}_elev{fname_elev}.png"
            else:
                outfile = output_dir / f"{scalar}_elev{fname_elev}.png"
            
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"      [OK] {outfile.name}")

        except Exception as e:
            print(f"      [ERROR] Error: {e}")

    # Save differences
    if is_overlay_mode and all_diffs:
        diff_df = pd.DataFrame(all_diffs)
        diff_file = output_dir / 'differences.csv'
        diff_df.to_csv(diff_file, index=False)
        print(f"\n[OK] Differences: {diff_file}")

        if diff_tolerance is not None:
            exceeding = diff_df[abs(diff_df['difference']) > diff_tolerance]
            if not exceeding.empty:
                print(f"[WARNING] {len(exceeding)} points exceed tolerance {diff_tolerance}")

    print(f"\n{'='*70}")
    print("[*] Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    app()
