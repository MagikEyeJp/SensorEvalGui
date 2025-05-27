# generated: 2025-05-18T11:05:00Z (auto)
# core/report_gen.py – Spec‑aligned, no legacy alias

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import base64
import json
import csv

__all__ = [
    "save_summary_txt",
    "report_csv",
    "report_html",
]

# ──────────────────────────────────────────────── helpers

def _write_if_enabled(flag: bool, path: Path, writer) -> None:
    if flag:
        writer(path)


def _b64(img: Path) -> str:
    with img.open("rb") as fh:
        return base64.b64encode(fh.read()).decode()

# ──────────────────────────────────────────────── public api

def save_summary_txt(summary: Dict[str, float], cfg: Dict[str, Any], path: Path):
    flag = cfg.get("output", {}).get("report_summary", True)
    _write_if_enabled(flag, path, lambda p: p.write_text("\n".join(f"{k}: {v:.3f}" for k, v in summary.items()), encoding="utf-8"))


def report_csv(stats: list[Dict[str, Any]], cfg: Dict[str, Any], path: Path):
    if not cfg.get("output", {}).get("report_csv", True):
        return
    if not stats:
        return
    fieldnames = stats[0].keys()
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(stats)


def report_html(summary: Dict[str, float], graphs: Dict[str, Path], cfg: Dict[str, Any], path: Path):
    if not cfg.get("output", {}).get("report_html", True):
        return

    order = cfg.get("output", {}).get("report_order", [
        "Dynamic Range (dB)",
        "SNR @ 50% (dB)",
        "DN @ 10 dB",
        "DN @ 0 dB",
        "Read Noise (DN)",
        "DSNU (DN)",
        "DN_sat",
        "PRNU (%)",
        "System Sensitivity",
    ])

    rows = "".join(
        f"<tr><td>{k}</td><td>{summary.get(k, '—'):.3f}</td></tr>" if isinstance(summary.get(k), (int, float)) else f"<tr><td>{k}</td><td>{summary.get(k, '—')}</td></tr>"
        for k in order
    )

    html = ["<html><head><meta charset='utf-8'><title>Sensor Evaluation Report</title></head><body>"]
    html.append("<h1>Sensor Evaluation Summary</h1>")
    html.append(f"<table border='1' cellpadding='4'>{rows}</table>")
    for key in [
        "snr_signal",
        "snr_exposure",
        "prnu_fit",
        "dsnu_map",
        "readnoise_map",
        "prnu_residual_map",
    ]:
        if key in graphs and graphs[key].exists():
            title = key.replace("_", " ").title()
            html.append(f"<h2>{title}</h2>")
            html.append(f"<img src='data:image/png;base64,{_b64(graphs[key])}' width='600'/>")
    html.append("</body></html>")
    html = "\n".join(html)
    path.write_text(html, encoding="utf-8")
