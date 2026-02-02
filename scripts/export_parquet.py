#!/usr/bin/env python

"""
Export HDF5 dataset to Parquet format.

HDF5 structure assumed:

root
├── train
│   ├── pose           (N, 6)           float32
│   ├── force          (N, 6)           float32
│   ├── bound_node     (N, O, 4)        float32
│   └── surface_node   (N, P, 4)        float32
└── test
    ├── poses
    ├── force
    ├── bound_node
    └── surface_node
"""

import argparse
import os
import h5py
import pyarrow as pa
import pyarrow.parquet as pq


def export_split(
    h5_path: str,
    split: str,
    output_dir: str,
    chunk_size: int,
) -> None:
    """
    Export one split (train / test) to Parquet files.
    """

    with h5py.File(h5_path, "r") as f:
        grp = f[split]

        pose_ds = grp["pose"]
        force_ds = grp["force"]
        surface_node_ds = grp["surface_node"]

        num_samples = pose_ds.shape[0]

        os.makedirs(output_dir, exist_ok=True)

        print(f"▶ Exporting split '{split}' ({num_samples} samples)")

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)

            pose = pose_ds[start:end]
            force = force_ds[start:end]
            surface_node = surface_node_ds[start:end]

            table = pa.table(
                {
                    "pose": pa.array(
                        pose.tolist(),
                        type=pa.list_(pa.float32(), 6),
                    ),
                    "force": pa.array(
                        force.tolist(),
                        type=pa.list_(pa.float32(), 6),
                    ),
                    "surface_node": pa.array(
                        surface_node.tolist(),
                        type=pa.list_(pa.list_(pa.float32(), 4)),
                    ),
                }
            )

            out_path = os.path.join(output_dir, f"data_{start // chunk_size:05d}.parquet")
            pq.write_table(table, out_path)

            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  ✓ {out_path} | samples: {end - start} | {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_path",
        type=str,
        default="data/omnineck/sim/data.h5",
        help="Path to the input HDF5 file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="parquet",
        help="Output directory",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help="Number of samples per Parquet file",
    )
    args = parser.parse_args()

    export_split(
        h5_path=args.h5_path,
        split="train",
        output_dir=os.path.join(os.path.dirname(args.h5_path), args.output_dir, "train"),
        chunk_size=args.chunk_size,
    )

    export_split(
        h5_path=args.h5_path,
        split="test",
        output_dir=os.path.join(os.path.dirname(args.h5_path), args.output_dir, "test"),
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
