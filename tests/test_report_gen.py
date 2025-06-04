import pytest

yaml = pytest.importorskip("yaml")

from utils.config import load_config
from core.report_gen import save_summary_txt, report_html, save_snr_signal_json
import numpy as np
import json


def test_save_summary_txt_full_scale(tmp_path):
    cfg_data = {
        "sensor": {"adc_bits": 10, "lsb_shift": 6},
        "measurement": {"gains": {0: {"folder": "g0"}}, "exposures": {}},
        "output": {"report_summary": True},
    }
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_data, fh)
    cfg = load_config(cfg_file)

    summary = {0.0: {"Metric": 1.0}}
    out_file = tmp_path / "summary.txt"
    save_summary_txt(summary, cfg, out_file)
    text = out_file.read_text(encoding="utf-8")
    assert "ADC Full Scale (DN):" in text
    assert str(((1 << 10) - 1) * (1 << 6)) in text


def test_report_html_summary_text(tmp_path):
    cfg_data = {"output": {"report_html": True}}
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_data, fh)
    cfg = load_config(cfg_file)

    summary_path = tmp_path / "summary.txt"
    summary_text = "Line1\nLine2"
    summary_path.write_text(summary_text, encoding="utf-8")

    html_file = tmp_path / "report.html"
    report_html({}, {}, cfg, html_file)
    html = html_file.read_text(encoding="utf-8")
    assert "Line1" in html and "Line2" in html


def test_save_snr_signal_json(tmp_path):
    cfg_data = {"output": {"snr_signal_data": True}, "sensor": {"adc_bits": 8}}
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_data, fh)
    cfg = load_config(cfg_file)

    data = {0.0: (np.array([1.0, 2.0]), np.array([2.0, 4.0]))}
    out_file = tmp_path / "snr.json"
    save_snr_signal_json(data, cfg, out_file, black_levels={0.0: 0.5})
    txt = json.loads(out_file.read_text(encoding="utf-8"))
    assert "0" in txt
    assert txt["0"]["signal"][0] == pytest.approx(1.0)
    assert len(txt["0"]["fit_signal"]) == 400
