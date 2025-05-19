# generated: 2025-05-18T10:15:00Z (auto)
# core/report_gen.py – config-aware implementation (Spec keys)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import base64
import json

__all__ = [
    "save_summary_txt",
    "report_csv",
    "report_html",
]

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _write_if_enabled(flag: bool, path: Path, writer) -> None:
    """Call *writer* if flag True."""
    if flag:
        writer(path)


def _embed_b64(img_path: Path) -> str:
    with img_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode()

# -----------------------------------------------------------------------------
# Public API (Spec key names)
# -----------------------------------------------------------------------------

def save_summary_txt(stats: Dict[str, float], cfg: Dict[str, Any], path: Path) -> None:
    _write_if_enabled(
        cfg.get("output", {}).get("report_summary_txt", True),
        path,
        lambda p: p.write_text("\n".join(f"{k}: {v:.3f}" for k, v in stats.items()), encoding="utf-8"),
    )


def report_csv(stats_list: list[Dict[str, Any]], cfg: Dict[str, Any], path: Path) -> None:
    import csv

    if not cfg.get("output", {}).get("report_csv", True):
        return
    if not stats_list:
        return

    keys = stats_list[0].keys()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats_list)


def report_html(summary: Dict[str, float], graph_paths: Dict[str, Path], cfg: Dict[str, Any], path: Path) -> None:
    if not cfg.get("output", {}).get("report_html", True):
        return

    order = cfg.get("output", {}).get(
        "report_order",
        [
            "Dynamic Range (dB)",
            "SNR (max)",
            "Read Noise",
            "DN @ 20 dB",
            "PRNU (%)",
        ],
    )

    summary_html = "".join(
        f"<tr><td>{k}</td><td>{summary.get(k, '—'):.3f}</td></tr>" if isinstance(summary.get(k), (int, float)) else f"<tr><td>{k}</td><td>{summary.get(k, '—')}</td></tr>"
        for k in order
    )

    html = f"""
    <html><head><meta charset='utf-8'><title>Sensor Evaluation Report</title></head><body>
    <h1>Sensor Evaluation Summary</h1>
    <table border='1' cellpadding='4'>
    {summary_html}
    </table>
    <h2>SNR vs Signal</h2>
    <img src='data:image/png;base64,{_embed_b64(graph_paths.get("snr_signal"))}' width='600'/>
    <h2>SNR vs Exposure</h2>
    <img src='data:image/png;base64,{_embed_b64(graph_paths.get("snr_exposure"))}' width='600'/>
    </body></html>
    """
    path.write_text(html, encoding="utf-8")

# -----------------------------------------------------------------------------
# Backward compatibility aliases
# -----------------------------------------------------------------------------
save_stats_csv = report_csv
generate_html_report = report_html
