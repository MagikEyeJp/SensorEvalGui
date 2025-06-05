# generated: 2025-05-18T11:05:00Z (auto)
# core/report_gen.py – Spec‑aligned, no legacy alias

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Mapping
from datetime import datetime

from utils import config as cfgutil
from utils.metrics import format_metric
import base64
import json
import csv

__all__ = [
    "save_summary_txt",
    "report_csv",
    "report_html",
    "save_snr_signal_json",
]

# ──────────────────────────────────────────────── helpers


def _write_if_enabled(flag: bool, path: Path, writer) -> None:
    if flag:
        writer(path)


def _b64(img: Path) -> str:
    with img.open("rb") as fh:
        return base64.b64encode(fh.read()).decode()


# ──────────────────────────────────────────────── public api


def _meta_lines(cfg: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    sensor_name = cfg.get("sensor", {}).get("name")
    if sensor_name:
        lines.append(f"Sensor: {sensor_name}")
    env_note = cfg.get("environment", {}).get("note")
    if env_note:
        lines.append("Environment:")
        for ln in str(env_note).splitlines():
            lines.append(f"  {ln}")
    power = cfg.get("illumination", {}).get("power_uW_cm2")
    exp_ms = cfg.get("illumination", {}).get("exposure_ms")
    if power is not None and exp_ms is not None:
        lines.append(f"Illumination: power_uW_cm2={power}, exposure_ms={exp_ms}")
    gains = ", ".join(f"{g:.0f} dB" for g, _ in cfgutil.gain_entries(cfg))
    if gains:
        lines.append(f"Gains: {gains}")
    exps = ", ".join(str(r) for r, _ in cfgutil.exposure_entries(cfg))
    if exps:
        lines.append(f"Exposures: {exps}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    return lines


def save_summary_txt(
    summary: Dict[float, Dict[str, float]], cfg: Dict[str, Any], path: Path
):
    flag = cfg.get("output", {}).get("report_summary", True)

    def writer(p: Path):
        lines = _meta_lines(cfg)
        adc_bits = int(cfg.get("sensor", {}).get("adc_bits", 0))
        lsb_shift = int(cfg.get("sensor", {}).get("lsb_shift", 0))
        lines.append(f"ADC Bits: {adc_bits}")
        lines.append(f"LSB Shift: {lsb_shift}")
        if adc_bits > 0:
            lines.append(f"ADC Full Scale (DN): {cfgutil.adc_full_scale(cfg)}")
        lines.append("")

        if summary:
            metrics = sorted({m for g in summary.values() for m in g})

            header = ["Metric"] + [f"{g:.0f} dB" for g in sorted(summary)]
            rows: list[list[str]] = []
            for key in metrics:
                row = [format_metric(key)]
                for gain in sorted(summary):
                    val = summary[gain].get(key, float("nan"))
                    if isinstance(val, (int, float)):
                        row.append(f"{val:.3f}")
                    else:
                        row.append(str(val))
                rows.append(row)

            table = [header] + rows
            col_widths = [
                max(len(str(col[i])) for col in table) for i in range(len(header))
            ]
            col_widths[0] = max(20, col_widths[0])

            def fmt(row: list[str]) -> str:
                return "  ".join(
                    text.ljust(col_widths[i]) for i, text in enumerate(row)
                )

            lines.append(fmt(header))
            for row in rows:
                lines.append(fmt(row))
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")

    _write_if_enabled(flag, path, writer)


def report_csv(stats: list[Dict[str, Any]], cfg: Dict[str, Any], path: Path):
    if not cfg.get("output", {}).get("report_csv", True):
        return
    if not stats:
        return
    fieldnames = stats[0].keys()
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(stats)


def report_html(
    summary: Dict[str, float], graphs: Dict[str, Path], cfg: Dict[str, Any], path: Path
):
    if not cfg.get("output", {}).get("report_html", True):
        return

    order = cfg.get("output", {}).get(
        "report_order",
        [
            "Dynamic Range",
            "SNR @ 50%",
            "DN @ 10 dB",
            "DN @ 0 dB",
            "Read Noise",
            "DSNU",
            "DN_sat",
            "Pseudo PRNU",
            "SensitivityDN",
        ],
    )

    summary_file = path.parent / "summary.txt"
    if summary_file.is_file():
        summary_text = summary_file.read_text(encoding="utf-8")
    else:
        lines = _meta_lines(cfg)
        for k in order:
            val = summary.get(k, "—")
            label = format_metric(k)
            if isinstance(val, (int, float)):
                lines.append(f"{label}: {val:.3f}")
            else:
                lines.append(f"{label}: {val}")
        summary_text = "\n".join(lines)

    html = [
        "<html><head><meta charset='utf-8'><title>Sensor Evaluation Report</title></head><body>"
    ]
    html.append("<h1>Sensor Evaluation Summary</h1>")
    html.append("<pre>")
    html.append(summary_text)
    html.append("</pre>")
    groups = [
        ("snr_signal", "noise_signal"),
        ("snr_exposure",),
        ("prnu_fit",),
        ("dsnu_map", "dsnu_map_scaled"),
        ("readnoise_map", "readnoise_map_scaled"),
        ("prnu_residual_map", "prnu_residual_map_scaled"),
        ("roi_area",),
    ]

    for keys in groups:
        valid = [k for k in keys if k in graphs and graphs[k].exists()]
        if not valid:
            continue
        title = keys[0].replace("_", " ").title()
        html.append(f"<h2>{title}</h2>")
        if len(valid) == 1:
            html.append(
                f"<img src='data:image/png;base64,{_b64(graphs[valid[0]])}' width='600'/>"
            )
        else:
            html.append("<table><tr>")
            for k in valid:
                html.append(
                    f"<td><img src='data:image/png;base64,{_b64(graphs[k])}' width='300'/></td>"
                )
            html.append("</tr></table>")
    html.append("</body></html>")
    html = "\n".join(html)
    path.write_text(html, encoding="utf-8")


def save_snr_signal_json(
    data: Dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    path: Path,
    *,
    black_levels: Dict[float, float] | None = None,
    dn_sat: float | Dict[float, float] | None = None,
) -> None:
    """Save SNR-Signal data and fitted curves as JSON.

    Parameters
    ----------
    dn_sat:
        Optional saturation level(s) to clip the SNR data before fitting.
    """

    import numpy as np
    from . import analysis

    proc_cfg = cfg.get("processing", {})
    snr_cfg = proc_cfg.get("snr_fit", {})
    flag = cfg.get("output", {}).get("snr_signal_data", False)

    def writer(p: Path) -> None:
        full_scale = cfgutil.adc_full_scale(cfg)
        out: Dict[str, Any] = {}
        for gain, (sig, snr) in sorted(data.items()):
            bl = 0.0 if black_levels is None else float(black_levels.get(gain, 0.0))
            limit = None
            if isinstance(dn_sat, dict):
                limit = dn_sat.get(gain)
            elif dn_sat is not None:
                limit = dn_sat

            if limit is not None and np.isfinite(limit):
                mask = sig <= limit
                sig_use = sig[mask]
                snr_use = snr[mask]
            else:
                sig_use = sig
                snr_use = snr

            xs, snr_fit = analysis.fit_snr_signal_model(
                sig_use,
                snr_use,
                full_scale,
                black_level=bl,
                deg=int(snr_cfg.get("deg", 3)),
                n_splines=snr_cfg.get("n_splines", "auto"),
                lam=snr_cfg.get("lam"),
                knot_density=snr_cfg.get("knot_density", "auto"),
                robust=snr_cfg.get("robust", "huber"),
                num_points=int(snr_cfg.get("num_points", 400)),
                max_signal=limit,
            )
            snr_fit = np.maximum(snr_fit, 0.0)
            out[f"{gain:g}"] = {
                "signal": sig_use.tolist(),
                "snr": snr_use.tolist(),
                "fit_signal": xs.tolist(),
                "fit_snr": snr_fit.tolist(),
            }
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    _write_if_enabled(flag, path, writer)
