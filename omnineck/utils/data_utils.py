#!/usr/bin/env python

import os
import numpy as np
import cv2


def save_data(data: list[tuple], data_dir: str) -> None:
    """
    Save the data to files.

    Args:
        data_dir (str): The directory to save the data to.
        data (list): The data to save.
    """
    # Create a directory to save the data
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    print(f"Saving data to {data_dir}...")

    pose_list = []
    force_list = []
    # Save the data to files
    for i, d in enumerate(data):
        # Unpack the data
        pose, force, img = d

        # Save the image
        img_path = os.path.join(data_dir, "images", f"{i}.jpg")
        cv2.imwrite(img_path, img)

        # Add the pose and force data to the lists
        pose_list.append(pose)
        force_list.append(force)

    # Save the pose and force data to files
    pose_list = np.array(pose_list)
    force_list = np.array(force_list)
    np.savetxt(os.path.join(data_dir, "pose.csv"), pose_list, fmt="%.6f", delimiter=",")
    np.savetxt(os.path.join(data_dir, "force.csv"), force_list, fmt="%.6f", delimiter=",")

    # Print the number of frames saved
    print(f"Saved {len(data)} frames to {data_dir}.")
