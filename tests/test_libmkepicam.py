import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import libmkepicam as cam


def test_open_close_cycle():
    assert cam.open_camera()
    cam.close_camera()


def test_grab_frame_returns_array():
    cam.open_camera()
    frame = cam.grab_frame()
    cam.close_camera()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640)
    assert frame.dtype == np.uint16
