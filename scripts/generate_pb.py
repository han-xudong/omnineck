# !/usr/bin/env python3

"""
Protobuf Generation Script

This script generates Python code from Protobuf (.proto) files located in the
`metaball/modules/protobuf` directory.

Usage:

```bash
python generate_pb.py
```

It uses the `protoc` compiler to generate the Python files and saves them in the
same directory as the .proto files.
"""

import os
import subprocess

def generate_protobuf_files(proto_dir: str) -> None:
    """
    Generate Python code from Protobuf files.

    Args:
        proto_dir (str): The directory containing the .proto files.
    """

    # Iterate over all .proto files in the specified directory
    for filename in os.listdir(proto_dir):
        if filename.endswith(".proto"):
            proto_path = os.path.join(proto_dir, filename)
            # Generate the Python file using protoc
            subprocess.run([
                "protoc",
                f"--python_out={proto_dir}",
                f"--proto_path={proto_dir}",
                proto_path
            ], check=True)
            print(f"Generated Python code for {filename}")
            
if __name__ == "__main__":
    proto_directory = os.path.join(
        "omnineck",
        "modules",
        "protobuf"
    )
    generate_protobuf_files(proto_directory)