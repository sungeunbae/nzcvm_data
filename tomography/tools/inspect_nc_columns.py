#!/usr/bin/env python3
"""Quick inspection of NetCDF columns after flattening."""

import sys
from pathlib import Path
import xarray as xr

if len(sys.argv) < 2:
    print("Usage: python inspect_nc_columns.py file.nc")
    sys.exit(1)

filepath = Path(sys.argv[1])
ds = xr.open_dataset(filepath)
df = ds.to_dataframe().reset_index()

print(f"File: {filepath.name}")
print(f"Shape after flattening: {df.shape}\n")

print("COLUMNS (use these indices in format_nztm_tomo_data.py):\n")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    try:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  Col {i:2d}: {col:15s} {str(dtype):10s}  [{min_val:.3f}, {max_val:.3f}]")
    except:
        print(f"  Col {i:2d}: {col:15s} {str(dtype):10s}  (non-numeric)")

print("\nSample commands for your files:")
print("""
# For CHOW2020 (Chow et al. 2021):
python format_nztm_tomo_data.py chow_file.nc \\
  --x-nztm-col 10 --y-nztm-col 11 --depth-col 0 \\
  --vp-col 3 --vs-col 4 --rho-col 5
""")
