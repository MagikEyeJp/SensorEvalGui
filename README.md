# 📷 Sensor Evaluation GUI

Evaluate image sensor performance (Dynamic Range, SNR curves, Exposure-dependent SNR) from 16-bit TIFF stacks captured under 940 nm LED and gray chart. Visualize results in a PySide6-based GUI and generate automated reports.

---

## 🧭 Purpose

This project aims to:
- Analyze image sensor TIFF stacks (gray chart, dark, flat) under varying exposure and gain
- Compute key metrics: SNR, Dynamic Range, read noise, PRNU residual
- Provide real-time GUI visualization (PySide6 + matplotlib)
- Export results as PNG graphs, summary text, CSV, and HTML report
- Display the generated summary and graphs directly in the GUI

---

## 🏗 Directory Structure

	SensorEvalGui/
	├── main.py                  # Entry point for GUI
	├── gui/
	│   └── main_window.py           # Main PySide6 GUI logic
	├── core/
	│   ├── loader.py                # TIFF loader and folder scanner
	│   ├── analysis.py              # ROI-based stats, SNR, DR
	│   ├── plotting.py              # Graph generation with matplotlib
	│   └── report_gen.py            # summary.txt and HTML report
	├── config/
        │   └── default_config.yaml      # configurable behavior
	└── utils/
	└── logger.py                # (Planned) logging utilities

---

## 📦 Dependencies

Create an isolated Python environment and install the project by running the
included setup script:

```bash
scripts/setup_env.sh
```

This creates a `.venv` directory and installs the package in editable mode with
its development dependencies (including `pytest`).

Optional:
        •       pyqtgraph for ROI/pixel image interaction (future feature)

Tested on:
	•	Python 3.10+
	•	macOS (M3 Ultra), Windows, Linux
```

## 🚀 How to Run

Run the GUI using `python main.py`

	1.	Select a folder containing a stack of 16-bit TIFF images
	2.	The GUI will compute ROI stats, plot SNR graphs, and export results
	3.	Output saved to output/ folder in the selected directory

##📊 Outputs

	•	📈 snr_signal.png: SNR vs Signal (DN)
	•	📉 snr_exposure.png: SNR vs Exposure Time
        •       🟢 prnu_fit.png: PRNU regression (μ vs σ in DN; set `plot.prnu_squared: true` for μ² vs σ² in DN²)
        •       🗺 dsnu_map.png: DSNU map
        •       🗺 dsnu_map_scaled.png: DSNU map (scaled to 99th percentile)
        •       🗺 readnoise_map.png: Read noise map
        •       🗺 readnoise_map_scaled.png: Read noise map (scaled to 99th percentile)
        •       🗺 prnu_residual_map.png: PRNU residual map (mean frame minus ROI average)
        •       🗺 prnu_residual_map_scaled.png: PRNU residual map (scaled to 99th percentile)
        •       📋 summary.txt: Key evaluation metrics
        •       📄 report.html: Embedded HTML report
        •       📌 Metrics include SNR @ 50% and DN @ SNR=1 (0 dB)
        •       📑 roi_stats.csv: Per-condition stats

For a comprehensive description of each output file, refer to
[Sensor_Output_Spec.md](Sensor_Output_Spec.md).

##🔮 Planned Features

	•	ROI/pixel mode switching in GUI
	•	Configurable pipeline via default_config.yaml
	•	Support for multi-gain and multi-exposure batch evaluation
	•	PRNU and black-level correction logic
        •       `gain_map_mode` normalizes the gain map by its mean for relative correction
	•	Optional Excel/Markdown export
	•	CI tests (PyTest + GitHub Actions)

##🧪 Example Stack Structure

	project/
	├── graychart/
	│   ├── gain0dB_exp1x/
	│   ├── gain0dB_exp2x/
	│   └── ...
	├── flat/
	├── dark/

📎 Reference (Gist Legacy)

Archived early version:
	•	https://gist.github.com/gaolay/bd04d5cd9d1bda623bb3f883abdb90a1

🪪 License

MIT License © 2025

