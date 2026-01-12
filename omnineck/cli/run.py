"""
OmniNeck CLI

Usage:

To run the OmniNeck, use the following command:

```
omnineck
```

Various configuration options are available:

╭───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
| Options       | Description                                   | Type   | Default                          |
|---------------|-----------------------------------------------|--------|----------------------------------|
| --host        | Host address for the publisher.               | str    | 127.0.0.1                        |
| --port        | Port number for the publisher.                | int    | 6666                             |
| --camera-yaml | Path to the camera configuration YAML file.   | str    | ./configs/camera_01.yaml         |
| --onnx-path   | Path to the ONNX model file.                  | str    | ./models/NeckNet.onnx            |
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""

import tyro
from omnineck import OmniNeck
from omnineck.configs.deploy import DeployConfig


def main():
    cfg = tyro.cli(DeployConfig)

    omnineck = OmniNeck(cfg=cfg)
    omnineck.run()