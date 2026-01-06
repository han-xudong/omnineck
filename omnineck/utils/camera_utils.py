#!/usr/bin/env python

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


def calibrate_camera(images, chess_size=(7, 7), square_size=0.025):
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
