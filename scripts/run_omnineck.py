#!/usr/bin/env python

"""
Run OmniNeck

This script is to run the OmniNeck, capturing omni-neck's deformation and
inferring the force and node displacement using the trained model.

Example usage:

```bash
python run_omnineck.py
```

Note that before running this script, please make sure to modify the configuration
file `configs/omnineck.yaml`.
"""

import time
import cv2
import yaml
from omnineck.devices import UsbCamera
from omnineck.modules import NeckPublisher
from omnineck.models import NeckNetRuntime


class OmniNeck:
    """
    OmniNeck class to run the omni-neck sensing and publishing.

    Attributes:
        camera (UsbCamera): The camera object to capture images and poses.
        neck_net (NeckNetRuntime): The NeckNet model for inference.
        neck_publisher (NeckPublisher): The publisher to send data.
    """

    def __init__(self) -> None:
        """
        Initialize the OmniNeck.
        """
        
        # Load the omni-neck parameters
        with open("./configs/omnineck.yaml", "r") as f:
            omnineck_params = yaml.load(f.read(), Loader=yaml.Loader)

        # Load the camera parameters
        with open(omnineck_params["camera_params_path"], "r") as f:
            camera_params = yaml.load(f.read(), Loader=yaml.Loader)

        # Load the detector parameters
        with open("./configs/detector.yaml", "r") as f:
            detector_params = yaml.load(f.read(), Loader=yaml.Loader)

        # Create a camera
        self.camera = UsbCamera(
            camera_params=camera_params,
            detector_params=detector_params,
        )

        # Create a NeckNet model
        self.neck_net = NeckNetRuntime(
            model_path=omnineck_params["model_path"],
        )

        # Create a neck publisher
        self.neck_publisher = NeckPublisher(
            host=omnineck_params["host"],
            port=omnineck_params["port"],
        )

    def release(self) -> None:
        """
        Release the camera and close the neck publisher.
        """

        self.camera.release()
        self.neck_publisher.close()

    def run(self) -> None:
        """
        Run the OmniNeck to capture images, infer force and node displacement,
        and publish the data.
        """

        # Set the jpeg parameters
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 50]

        # Initialize the variables
        start_time = time.time()
        frame_count = 0

        # Start publishing
        try:
            while True:
                # Get the image and pose
                pose, img = self.camera.readImageAndPose()
                # Convert the pose to the reference pose
                pose_ref = self.camera.poseToReferece(pose)
                # Convert the pose from the marker frame to the camera frame
                pose_global = self.camera.poseAxisTransfer(pose_ref)
                # Convert the pose to the euler angles
                pose_euler = self.camera.poseVectorToEuler(pose_global)

                # Predict the force and node
                force, node = self.neck_net.infer(pose_euler)

                # Publish the message
                self.neck_publisher.publishMessage(
                    cv2.imencode(".jpg", img, jpeg_params)[1].tobytes(),
                    pose_euler,
                    force,
                    node,
                )

                frame_count += 1

                # Print the FPS
                if frame_count == 60:
                    print(f"FPS: %.2f" % (frame_count / (time.time() - start_time)))
                    start_time = time.time()
                    frame_count = 0
        except KeyboardInterrupt:
            print("Stopping the camera...")
        finally:
            self.release()


if __name__ == "__main__":
    # Create a OmniNeck instance
    omnineck = OmniNeck()
    # Run the OmniNeck
    omnineck.run()
