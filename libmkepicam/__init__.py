from __future__ import annotations

import numpy as np


class DummyMkEpiCam:
    """Simple dummy for libmkepicam camera functions."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        return True

    def close(self) -> None:
        self._opened = False

    def capture_frame(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        return np.zeros((self.height, self.width), dtype=np.uint16)


_camera = DummyMkEpiCam()


def open_camera() -> bool:
    return _camera.open()


def close_camera() -> None:
    _camera.close()


def grab_frame() -> np.ndarray:
    return _camera.capture_frame()
