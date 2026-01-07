#!/usr/bin/env python

import subprocess
import re
from collections import defaultdict
import cv2
import numpy as np


# Set the jpeg parameters
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 50]


def img_encode(img: np.ndarray) -> bytes:
    """
    Encode the image to JPEG format.

    Args:
        img (np.ndarray): The image to encode.

    Returns:
        bytes: The encoded image in JPEG format.
    """

    # Encode the image to JPEG format
    _, img_encoded = cv2.imencode(".jpg", img, JPEG_PARAMS)

    return img_encoded.tobytes()


def calibrate_chessboard(images, chess_size=(7, 7), square_size=0.025):
    """
    Find chessboard corners in the images for camera calibration.

    Args:
        images (list): List of images for calibration.
        chess_size (tuple): Size of the chessboard pattern (rows, columns).
        square_size (float): Size of a square in the chessboard pattern (in mm).

    Returns:
        tuple: Camera matrix, distortion coefficients, rotation vectors, translation vectors.
    """

    # Prepare object points and image points for calibration
    pattern_points = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    pattern_points[:, :2] = np.mgrid[0 : chess_size[0], 0 : chess_size[1]].T.reshape(-1, 2)
    pattern_points *= square_size

    # Load the images for calibration
    obj_points = []
    img_points = []
    count_ret = 0
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        if ret:
            obj_points.append(pattern_points)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, chess_size, corners, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)
            count_ret += 1
    print(f"Found corners in {count_ret} images out of {len(images)}.")
    cv2.destroyAllWindows()

    # Perform camera calibration
    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist, rvecs, tvecs


def get_camera_capabilities(device="/dev/video0") -> dict:
    """
    Query camera supported pixel formats, resolutions and fps using v4l2-ctl.

    Returns:
        dict:
        {
            "MJPG": {
                (640, 480): [30, 60],
                (1280, 720): [30]
            },
            "YUYV": {
                (640, 480): [30]
            }
        }
    """
    cmd = ["v4l2-ctl", "--device", device, "--list-formats-ext"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"v4l2-ctl failed:\n{result.stderr}")

    output = result.stdout.splitlines()

    caps = defaultdict(lambda: defaultdict(list))

    current_fmt = None
    current_size = None

    for line in output:
        line = line.strip()

        # Pixel Format: 'MJPG'
        m = re.match(r"Pixel Format:\s+'(\w+)'", line)
        if m:
            current_fmt = m.group(1)
            continue

        # Size: Discrete 1280x720
        m = re.match(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if m and current_fmt:
            w, h = map(int, m.groups())
            current_size = (w, h)
            continue

        # Interval: Discrete 0.033s (30.000 fps)
        m = re.search(r"\(([\d.]+)\s+fps\)", line)
        if m and current_fmt and current_size:
            fps = int(float(m.group(1)))
            if fps not in caps[current_fmt][current_size]:
                caps[current_fmt][current_size].append(fps)

    # convert defaultdict → dict
    return {fmt: dict(res) for fmt, res in caps.items()}
