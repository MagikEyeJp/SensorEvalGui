#!/usr/bin/env python3
import sys
import types
import importlib

import pytest

pytest.importorskip('numpy')
pytest.importorskip('tifffile')

import numpy as np
import tifffile

# provide roifile stub before importing modules that require it
class _StubROI:
    left = 0
    top = 0
    width = 2
    height = 2


def _roiread(path):
    return [_StubROI()]

sys.modules['roifile'] = types.SimpleNamespace(roiread=_roiread)

from utils.config import load_config
import utils.roi as roi  # reload with stubbed roifile
importlib.reload(roi)
import core.analysis as analysis
importlib.reload(analysis)


def test_calculate_dark_noise_gain(tmp_path):
    project = tmp_path
    gain_dir = project / 'gain_0dB' / 'dark'
    gain_dir.mkdir(parents=True)

    for i in range(2):
        tifffile.imwrite(gain_dir / f'frame{i}.tiff', np.full((2, 2), i, dtype=np.uint16))

    roi_file = project / 'roi.roi'
    roi_file.write_text('dummy')

    cfg_data = {
        'measurement': {
            'gains': {0: {'folder': 'gain_0dB'}},
            'flat_roi_file': str(roi_file),
        }
    }
    cfg_file = project / 'config.yaml'
    with cfg_file.open('w') as fh:
        import yaml
        yaml.safe_dump(cfg_data, fh)

    cfg = load_config(cfg_file)

    dsnu, rn, dsnu_map, rn_map = analysis.calculate_dark_noise_gain(project, 0, cfg)

    assert pytest.approx(dsnu, abs=1e-6) == 0.0
    assert pytest.approx(rn, abs=1e-6) == 0.5
    assert dsnu_map.shape == (2, 2)
    assert rn_map.shape == (2, 2)
