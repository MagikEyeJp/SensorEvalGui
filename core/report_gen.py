# core/report_gen.py

from pathlib import Path
from typing import Dict
import json
import base64


def save_summary_txt(stats: Dict[str, float], output_path: Path) -> None:
    """
    Save sensor evaluation summary to a plain text file.
    """
    with open(output_path, 'w') as f:
        for key, val in stats.items():
            f.write(f"{key}: {val:.3f}\n")


def save_stats_csv(stats_list: list[Dict], output_path: Path) -> None:
    """
    Save a list of stats (dict per ROI or condition) to CSV.
    """
    import csv
    if not stats_list:
        return

    keys = stats_list[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats_list)


def embed_image_as_base64(path: Path) -> str:
    """
    Convert image to base64 for HTML embedding.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(summary: Dict[str, float], graph_paths: Dict[str, Path], output_path: Path) -> None:
    """
    Generate an HTML report embedding key graphs and summary metrics.
    """
    snr_sig_b64 = embed_image_as_base64(graph_paths["snr_signal"])
    snr_exp_b64 = embed_image_as_base64(graph_paths["snr_exposure"])

    html = f"""
    <html><head><title>Sensor Evaluation Report</title></head><body>
    <h1>Sensor Evaluation Summary</h1>
    <ul>
    {''.join(f'<li>{k}: {v:.3f}</li>' for k, v in summary.items())}
    </ul>
    <h2>SNR vs Signal</h2>
    <img src="data:image/png;base64,{snr_sig_b64}" width="600"/>
    <h2>SNR vs Exposure</h2>
    <img src="data:image/png;base64,{snr_exp_b64}" width="600"/>
    </body></html>
    """
    output_path.write_text(html)


def save_json_config(config: dict, path: Path) -> None:
    """
    Optional helper to save config for debugging.
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)