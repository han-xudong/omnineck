#!/usr/bin/env python

"""
Export Parquet Data

This script exports the data to Parquet format for easier analysis.

Example usage:

```bash
python export_parquet.py --data_path <data_path> --chunk_size <chunk_size>
```

where <data_path> is the path to the data file, and <chunk_size> (default: 5000) is the number of samples per chunk.
"""

import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def export_parquet(data_path: str, chunk_size: int) -> None:
    """
    Export data to Parquet format.

    Args:
        data_path (str): The path to the data file.
    """

    # Load the data
    data = np.load(data_path)
    num_samples, num_features = data.shape

    # Create output directory if not exists
    output_dir = os.path.join(os.path.dirname(data_path), "parquet")
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    for i in range(0, num_samples, chunk_size):
        chunk_data = data[i : i + chunk_size]
        motion = chunk_data[:, :6]
        force = chunk_data[:, 6:12]
        nodes = chunk_data[:, 12:].reshape(-1, (num_features - 12) // 3, 3)

        motion_array = pa.array(motion.tolist(), type=pa.list_(pa.float64(), 6))
        force_array = pa.array(force.tolist(), type=pa.list_(pa.float64(), 6))
        nodes_array = pa.array(nodes.tolist(), type=pa.list_(pa.list_(pa.float64(), 3)))
        schema = pa.schema([
            ("motion", pa.list_(pa.float64(), 6)),
            ("force", pa.list_(pa.float64(), 6)),
            ("nodes", pa.list_(pa.list_(pa.float64(), 3))),
        ])
        table = pa.table({
            "motion": motion_array,
            "force": force_array,
            "nodes": nodes_array,
        }, schema=schema)
        
        output_path = os.path.join(output_dir, f"data_{i // chunk_size}.parquet")
        pq.write_table(table, output_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Exported to {output_path}, size: {file_size:.2f} MB, samples: {len(chunk_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/finger/sim/data.npy",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of samples per chunk.",
    )
    args = parser.parse_args()
    export_parquet(args.data_path, args.chunk_size)
