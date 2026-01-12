#!/usr/bin/env python

import zmq
import numpy as np
from typing import Tuple
from datetime import datetime
from omnineck.modules.protobuf import omnineck_msg_pb2


class OmniNeckPublisher:
    def __init__(
        self, host: str, port: int, hwm: int = 1, conflate: bool = True
    ) -> None:
        """Publisher initialization.

        Args:
            host (str): The host address of the publisher.
            port (int): The port number of the publisher.
            hwm (int): High water mark for the publisher. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
        """

        print("{:-^80}".format(" OmniNeck Publisher Initializing... "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ publisher
        self.publisher = self.context.socket(zmq.PUB)
        # Set high water mark
        self.publisher.set_hwm(hwm)
        # Set conflate
        self.publisher.setsockopt(zmq.CONFLATE, conflate)
        # Bind the address
        self.publisher.bind(f"tcp://{host}:{port}")

        print("OmniNeck Publisher Initialization Done.")

    def publishMessage(
        self,
        img_bytes: bytes = b"",
        pose: list = np.zeros(6, dtype=np.float32).tolist(),
        force: list = np.zeros(6, dtype=np.float32).tolist(),
        node: list = np.zeros(6, dtype=np.float32).tolist(),
    ) -> None:
        """Publish the message.

        Args:
            img: The image captured by the camera.
            pose: The pose of the marker.
            force: The force on the bottom surface of the omni-neck.
            node: The node displacement of the omni-neck.
        """

        # Set the message
        omnineck = omnineck_msg_pb2.OmniNeck()
        omnineck.timestamp = datetime.now().timestamp()
        omnineck.img = img_bytes
        omnineck.pose[:] = pose
        omnineck.force[:] = force
        omnineck.node[:] = node
        # Publish the message
        self.publisher.send(omnineck.SerializeToString())

    def close(self):
        """Close ZMQ socket and context to prevent memory leaks."""
        if hasattr(self, "publisher") and self.publisher:
            self.publisher.close()
        if hasattr(self, "context") and self.context:
            self.context.term()


class OmniNeckSubscriber:
    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 1,
        conflate: bool = True,
        timeout: int = 100,
    ) -> None:
        """Subscriber initialization.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
            hwm (int): High water mark for the subscriber. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
            timeout (int): Maximum time to wait for a message in milliseconds. Default is 100 ms.
        """

        print("{:-^80}".format(" OmniNeck Subscriber Initializing... "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{host}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout
        
        print("OmniNeck Subscriber Initialization Done.")

    def subscribeMessage(self) -> Tuple[bytes, list, list, list]:
        """Subscribe the message.

        Args:
            timeout: Maximum time to wait for a message in milliseconds. Default is 100ms.

        Returns:
            img: The image captured by the camera.
            pose: The pose of the marker.
            force: The force on the bottom surface of the omni-neck.
            node: The node displacement of the omni-neck.

        Raises:
            zmq.ZMQError: If no message is received within the timeout period.
        """

        # Receive the message

        if self.poller.poll(self.timeout):
            # Receive the message
            msg = self.subscriber.recv()
            
            # Parse the message
            omnineck = omnineck_msg_pb2.OmniNeck()
            omnineck.ParseFromString(msg)
        else:
            raise RuntimeError("No message received within the timeout period.")
        return (
            omnineck.img,
            omnineck.pose,
            omnineck.force,
            omnineck.node,
        )

    def close(self):
        """Close ZMQ socket and context to prevent memory leaks."""
        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()
