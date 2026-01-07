#!/usr/bin/env python

"""
Deploy OmniNeck

This script is to run the OmniNeck, capturing omni-neck's deformation and
inferring the force and node displacement using the trained model.

Example usage:

```bash
uv run python deploy.py
```

Various configuration options are available:
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
| Options       | Description                                   | Type   | Default                          |
|---------------|-----------------------------------------------|--------|----------------------------------|
| --host        | Host address for the publisher.               | str    | 127.0.0.1                        |
| --port        | Port number for the publisher.                | int    | 6666                             |
| --camera-yaml | Path to the camera configuration YAML file.   | str    | ./configs/camera/camera_001.yaml |
| --onnx-path   | Path to the ONNX model file.                  | str    | ./models/NeckNet.onnx            |
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""

import time
import tyro
import cv2
import yaml
from omnineck.devices import UsbCamera
from omnineck.modules import NeckPublisher
from omnineck.models import NeckNetRuntime
from omnineck.configs.deploy import DeployConfig, CameraConfig


class OmniNeck:
    """
    OmniNeck class to run the omni-neck sensing and publishing.

    Attributes:
        camera (UsbCamera): The camera object to capture images and poses.
        neck_net (NeckNetRuntime): The NeckNet model for inference.
        neck_publisher (NeckPublisher): The publisher to send data.
    """

    def __init__(self, cfg: DeployConfig) -> None:
        """
        Initialize the OmniNeck.
        
        Args:
            cfg (DeployConfig): The deployment configuration.
        """
        
        with open(cfg.camera_yaml, "r") as f:
            camera_params_dict = yaml.safe_load(f)
            
        camera_cfg = CameraConfig(**camera_params_dict)

        # Create a camera
        self.camera = UsbCamera(camera_cfg)

        # Create a NeckNet model
        self.neck_net = NeckNetRuntime(cfg.onnx_path)

        # Create a neck publisher
        self.neck_publisher = NeckPublisher(cfg.host, cfg.port)

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
    cfg = tyro.cli(DeployConfig)

    omnineck = OmniNeck(cfg)
    omnineck.run()
