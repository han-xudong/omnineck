from dataclasses import dataclass


@dataclass
class DeployConfig:
    host: str = "127.0.0.1"
    """Host address for the publisher."""
    
    port: int = 6666
    """Port number for the publisher."""
    
    camera_yaml: str = "./configs/camera/camera_001.yaml"
    """Path to the camera configuration YAML file."""
    
    onnx_path: str = "./models/NeckNet.onnx"
    """Path to the ONNX model file."""