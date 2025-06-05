"""Metric unit definitions."""

from __future__ import annotations

METRIC_UNITS: dict[str, str] = {
    "Dynamic Range": "dB",
    "SNR @ 50%": "dB",
    "Read Noise": "DN",
    "Black level": "DN",
    "DSNU": "DN",
    "DN_sat": "",
    "Pseudo PRNU": "%",
    "System Sensitivity": "DN / μW·cm²·s",
}


def format_metric(name: str) -> str:
    """Return metric label with unit in parentheses."""

    unit = METRIC_UNITS.get(name, "")
    return f"{name} ({unit})" if unit else name
