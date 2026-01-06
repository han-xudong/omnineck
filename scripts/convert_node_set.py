#!/usr/bin/env python

"""
Convert Node Set

This script is to load a mesh file and convert the node set from .inp to .stl. 

Example usage:

```bash
python convert_node_set.py --stl_file <stl_file> --inp_file <inp_file> --node_set <node_set>
```

where <stl_file> is the path to the STL file, <inp_file> is the path to the INP file, 
<node_set> is the name of the target node set in the .inp file.
"""

import argparse
import os
from typing import Tuple
import numpy as np
import trimesh

def load_stl_file(stl_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh file and export the point and face data.

    Args:
        mesh_path: Path to the mesh file.
    """

    # Load the mesh
    mesh = trimesh.load_mesh(stl_file_path)

    # Export the node and face data
    nodes = mesh.vertices
    triangles = mesh.faces

    print("Nodes shape:", nodes.shape)
    print("Triangles shape:", triangles.shape)
    
    return nodes, triangles

def read_nodes_from_inp_file(inp_file_path: str, keyword: str) -> dict:
    
    """
    Read the node set from the INP file.

    Args:
        inp_file_path: Path to the INP file.
        keyword: Keyword to search for in the INP file.
    """

    # Read the INP file
    with open(inp_file_path, "r") as f:
        lines = f.readlines()

    # Read the node set
    nodes = {}
    for line in lines:
        if line.startswith(keyword):
            if keyword.startswith("*Node"):
                for node_line in lines[lines.index(line) + 1:]:
                    if node_line.startswith("*"):
                        break
                    node_data = node_line.split(",")
                    node_id = int(node_data[0].strip())
                    coordinates = np.array([float(coord.strip()) for coord in node_data[1:]])
                    nodes[node_id] = coordinates
                break
            elif keyword.startswith("*Nset"):
                for node_line in lines[lines.index(line) + 1:]:
                    if node_line.startswith("*"):
                        break
                    node_data = node_line.split(",")
                    for node in node_data:
                        nodes[int(node.strip())] = int(node.strip())
                break
    
    return nodes

def match_nodes(stl_nodes: np.ndarray, inp_nodes: dict) -> dict:
    """
    Match the nodes from the STL file with the nodes from the INP file.

    Args:
        stl_nodes (np.ndarray): Nodes from the STL file.
        inp_nodes (List[List[int, np.ndarray]]): Nodes from the INP file.

    Returns:
        inp_to_stl_node_map (dict): List of node IDs from the INP file that match the nodes from the STL file.
    """

    inp_to_stl_node_map = {}
    
    for node_id, coordinates in inp_nodes.items():
        
        distances = np.linalg.norm(stl_nodes - coordinates, axis=1)
        closest_node_index = np.argmin(distances)
        
        inp_to_stl_node_map[node_id] = closest_node_index
        
    return inp_to_stl_node_map


def convert_node_set(node_set: list, inp_to_stl_node_map: dict) -> list:
    """
    Convert the node set from the INP file to the STL file.

    Args:
        node_set (list): Node set from the INP file.
        inp_to_stl_node_map (dict): Mapping of node IDs from the INP file to the STL file.

    Returns:
        converted_node_set (list): Converted node set.
    """

    converted_node_set = []
    
    for node in node_set:
        
        if node in inp_to_stl_node_map:
            converted_node_set.append(inp_to_stl_node_map[node])
            
    return converted_node_set
    
def main(
    stl_file: str,
    inp_file: str,
    node_set: str,
) -> None:
    """
    Main function to load the mesh file and export the point and face data.

    Args:
        stl_file: Path to the STL file.
        inp_file_: Path to the INP file.
        node_set: Name of the target node set in the INP file.
    """

    # Load the mesh
    stl_nodes, triangles = load_stl_file(stl_file)
    
    # Read the nodes from the INP file
    inp_nodes = read_nodes_from_inp_file(inp_file, keyword="*Node")
    
    # Match the nodes
    inp_to_stl_node_map = match_nodes(stl_nodes, inp_nodes)
    
    # Read the node set from the INP file
    inp_node_set = read_nodes_from_inp_file(inp_file, keyword=f"*Nset, nset={node_set}")
    
    print("Node Set Name:", node_set)
    print("Node Set Size:", len(inp_node_set))
    
    # Convert the node set
    stl_node_set = convert_node_set(inp_node_set, inp_to_stl_node_map)
    
    # Export the converted node set to txt file in the same directory
    txt_path = os.path.join(os.path.dirname(inp_file), f"converted_{node_set.lower()}.txt")
    with open(txt_path, "w") as f:
        for node in stl_node_set:
            f.write(f"{node}\n")
            
    print(f"Converted node set saved to {txt_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Mesh")
    parser.add_argument(
        "--stl_file",
        type=str,
        default="./templates/finger.stl",
        help="Path to the STL file",
    )
    parser.add_argument(
        "--inp_file",
        type=str,
        default="./templates/finger.inp",
        help="Path to the INP file",
    )
    parser.add_argument(
        "--node_set",
        type=str,
        default="Set-bottom",
        help="Name of the target node set in the INP file",
    )
    args = parser.parse_args()

    main(
        stl_file=args.stl_file,
        inp_file=args.inp_file,
        node_set=args.node_set,
    )