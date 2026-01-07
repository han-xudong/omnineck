# !/usr/bin/env python3

"""
Camera Calibration Script

This script captures images from a camera (USB) and saves them for calibration.

```bash
python calibrate_camera.py --id <id> --width 640 --height 480
```

where `<id>` is the camera ID which is usually 0 for the first camera, and can also be found in `ls /dev/video*`.

The script opens a window to display the frame from the camera.
You can press 'c' to capture the image, and press 'ESC' to quit capturing.
The captured images are saved in the `data/camera_calibration` directory.

It also provides an option to calibrate the camera using OpenCV or MATLAB after capturing the images.
If you choose OpenCV, it will prompt you to enter the chessboard size and square size.
Then it will perform camera calibration and display the camera matrix and distortion coefficients.
If you choose MATLAB, it will exit the script and you can use MATLAB with the saved images.
A MATLAB script is also provided in the `scripts` directory for saving the camera matrix and
distortion coefficients into a `.yaml` file.

The calibration results should be copied to the `configs/camera` directory for further use.

For more information, please refer to https://github.com/han-xudong/omnineck.
"""

import argparse
import sys
import os
import cv2
import yaml
from omnineck.utils.camera_utils import calibrate_chessboard


def main(id: int, width: int, height: int) -> None:
    """
    Main function to calibrate the camera.

    Args:
        id (int): The ID of the USB camera.
        width (int): The width of the image.
        height (int): The height of the image.
    """

    # Initialize the USB camera
    try:
        camera = cv2.VideoCapture(id)
    except Exception as e:
        print(f"\033[31mError initializing camera: {e}\033[0m")
        print("\033[31mPlease check the camera ID.\033[0m")
        sys.exit()
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Print camera information
    print(f"Camera id: {id}")
    print(f"Camera width: {width}")
    print(f"Camera height: {height}")

    # Create a window to display the camera feed
    cv2.namedWindow(f"Camera {id}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Camera {id}", width, height)

    # Create a directory to save the images
    img_dir = os.path.join(
        "data",
        "camera_calibration",
        f"camera_{id}_{width}x{height}",
    )
    os.makedirs(img_dir, exist_ok=True)

    # Start capturing images
    print("Press 'c' to capture the image.")
    print("Press 'ESC' to quit capturing.")
    count = 0
    while True:
        # Read the frame from the camera
        ret, img = camera.read()
        # Display the frame
        cv2.imshow(f"Camera {id}", img)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == 27:
            # ESC key pressed, exit the loop
            break
        elif key == ord("c"):
            # 'c' key pressed, capture the image
            img_path = os.path.join(img_dir, f"{count}.jpg")
            cv2.imwrite(img_path, img)
            print(f"{count}.jpg saved.")
            count += 1

    camera.release()

    # Close the OpenCV window
    cv2.destroyAllWindows()
    print("Image collection completed.")
    print(f"Images saved to {img_dir}.")

    # Ask the user if they want to calibrate the camera using OpenCV or MATLAB
    print("To calibrate the camera, MATLAB or OpenCV can be used.")
    print("To use OpenCV, press 'y'. The script will continue to calibrate the camera.")
    print("To use MATLAB, press 'n'. The script will exit, and you can use MATLAB with the saved images.")
    user_input = input("Use OpenCV? (y/n): ")

    # If the user chooses OpenCV, perform camera calibration
    if user_input == "y":
        # Set up OpenCV camera calibration parameters
        print("OpenCV camera calibration started.")
        print("Please select the chessboard size.")
        chess_size = input("Chessboard size (e.g., 12x9): ")
        chess_size = tuple(map(int, chess_size.split("x")))
        print("Please select the square size.")
        square_size = float(input("Square size (mm): "))

        # Load the images for calibration
        print("Loading images for calibration...")
        images = []
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            images.append(img)

        # Find chessboard corners in the images
        print("Finding chessboard corners...")
        mtx, dist, rvecs, tvecs = calibrate_chessboard(images, chess_size=chess_size, square_size=square_size)

        # Print the calibration results
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
        print("Rotation vectors:")
        print(rvecs)
        print("Translation vectors:")
        print(tvecs)
        print("OpenCV camera calibration completed.")

        # Ask the user if they want to save the camera matrix and distortion coefficients
        user_input = input("Save the camera matrix and distortion coefficients? (y/n): ")
        if user_input == "y":
            camera_dist = {
                "mtx": mtx.tolist(),
                "dist": dist.tolist(),
            }
            # Save the camera matrix and distortion coefficients to a YAML file
            with open(
                f"{img_dir}/camera_{id}_{width}x{height}.yaml",
                "w",
            ) as f:
                yaml.dump(camera_dist, f)
            print(
                f"Calibration results are saved in {img_dir}/camera_{id}_{width}x{height}.yaml."
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate the camera.")
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="The ID of the USB camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="The width of the image.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="The height of the image.",
    )
    args = parser.parse_args()

    main(args.id, args.width, args.height)
    print("Camera calibration script finished.")
