import pytest

yaml = pytest.importorskip("yaml")

from utils.config import load_config
from core.report_gen import save_summary_txt


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
