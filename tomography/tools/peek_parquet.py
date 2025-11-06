#!/usr/bin/env python3
"""
Simple peek at parquet file structure: show columns and sample data.
Usage: python peek_parquet.py <parquet_file>
"""

import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd


def peek_parquet(parquet_file):
    """Show parquet structure and sample data."""
    
    parquet_file = Path(parquet_file)
    
    if not parquet_file.exists():
        print(f"Error: File not found: {parquet_file}")
        return False
    
    print("="*80)
    print(f"PARQUET FILE: {parquet_file.name}")
    print("="*80)
    
    # Read metadata
    pq_file = pq.ParquetFile(parquet_file)
    
    print(f"\nFile size: {parquet_file.stat().st_size / 1e9:.3f} GB")
    print(f"Total rows: {pq_file.metadata.num_rows:,}")
    print(f"Row groups: {pq_file.num_row_groups}")
    
    # Get schema
    print(f"\n{'Index':<6} {'Column Name':<30} {'Data Type':<15}")
    print("-" * 80)
    
    arrow_schema = pq_file.schema_arrow
    for i, field in enumerate(arrow_schema):
        col_name = field.name
        col_type = str(field.type)
        print(f"{i:<6} {col_name:<30} {col_type:<15}")
    
    # Load sample
    print(f"\n{'='*80}")
    print("SAMPLE DATA (first row):")
    print("="*80)
    
    # Read first row group
    df = pq_file.read_row_groups([0]).to_pandas()
    
    print(f"\nLoaded {len(df):,} rows from first row group\n")
    
    # Show first row with all columns
    first_row = df.iloc[0]
    for i, (col_name, value) in enumerate(first_row.items()):
        # Truncate long values
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:60] + "..."
        print(f"{i:<6} {col_name:<30} {val_str}")
    
    # Show data ranges for numeric columns
    print(f"\n{'='*80}")
    print("DATA RANGES (numeric columns):")
    print("="*80 + "\n")
    
    for i, col in enumerate(df.columns):
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"{i:<6} {col:<30} min={min_val:12.2f}  max={max_val:12.2f}")
        except:
            pass
    
    # Find likely columns
    print(f"\n{'='*80}")
    print("LIKELY COLUMNS FOR PROCESSING:")
    print("="*80 + "\n")
    
    x_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if 'x' in col.lower()]
    y_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if 'y' in col.lower()]
    z_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if any(x in col.lower() for x in ['depth', 'elev', 'z'])]
    vp_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if any(x in col.lower() for x in ['vp', 'p_vel', 'p-wave'])]
    vs_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if any(x in col.lower() for x in ['vs', 's_vel', 's-wave'])]
    rho_cand = [f"{i}:{col}" for i, col in enumerate(df.columns) if any(x in col.lower() for x in ['rho', 'dens', 'density'])]
    
    print(f"X coordinate:    {', '.join(x_cand) if x_cand else 'Not found'}")
    print(f"Y coordinate:    {', '.join(y_cand) if y_cand else 'Not found'}")
    print(f"Depth/Elevation: {', '.join(z_cand) if z_cand else 'Not found'}")
    print(f"Vp velocity:     {', '.join(vp_cand) if vp_cand else 'Not found'}")
    print(f"Vs velocity:     {', '.join(vs_cand) if vs_cand else 'Not found'}")
    print(f"Density (Rho):   {', '.join(rho_cand) if rho_cand else 'Not found'}")
    
    print(f"\n{'='*80}")
    print("USAGE FOR PROCESSING:")
    print("="*80)
    print("""
For sparse points (keep measurement locations as-is):
  python process_nztm_tomo_data_sparse.py <file> \\
    --x-nztm-col X --y-nztm-col Y --depth-col D \\
    --vp-col VP --vs-col VS --rho-col RHO

For interpolated grid (create smooth continuous model):
  python process_nztm_tomo_data_parquet.py <file> \\
    --x-nztm-col X --y-nztm-col Y --depth-col D \\
    --vp-col VP --vs-col VS --rho-col RHO \\
    --interpolate --spacing 2.0

Replace X, Y, D, VP, VS, RHO with the column indices shown above.
    """)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek_parquet.py <parquet_file>")
        print("\nExample:")
        print("  python peek_parquet.py DB2025_NZTM.parquet")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    
    if not peek_parquet(parquet_file):
        sys.exit(1)
