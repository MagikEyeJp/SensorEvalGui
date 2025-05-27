# 📷 Sensor Evaluation GUI

Evaluate image sensor performance (Dynamic Range, SNR curves, Exposure-dependent SNR) from 16-bit TIFF stacks captured under 940 nm LED and gray chart. Visualize results in a PySide6-based GUI and generate automated reports.

---

## 🧭 Purpose

This project aims to:
- Analyze image sensor TIFF stacks (gray chart, dark, flat) under varying exposure and gain
- Compute key metrics: SNR, Dynamic Range, read noise, PRNU
- Provide real-time GUI visualization (PySide6 + matplotlib)
- Export results as PNG graphs, summary text, CSV, and HTML report

---

## 🏗 Directory Structure

	sensor_eval/
	├── main.py                  # Entry point for GUI
	├── gui/
	│   └── main_window.py           # Main PySide6 GUI logic
	├── core/
	│   ├── loader.py                # TIFF loader and folder scanner
	│   ├── analysis.py              # ROI-based stats, SNR, DR
	│   ├── plotting.py              # Graph generation with matplotlib
	│   └── report_gen.py            # summary.txt and HTML report
	├── config/
	│   └── default_config.json      # (Planned) configurable behavior
	└── utils/
	└── logger.py                # (Planned) logging utilities

---

## 📦 Dependencies

```bash
pip install numpy pandas tifffile matplotlib PySide6

Optional:
	•	pyqtgraph for ROI/pixel image interaction (future feature)

Tested on:
	•	Python 3.10+
	•	macOS (M3 Ultra), Windows, Linux
```

## 🚀 How to Run

python -m sensor_eval

	1.	Select a folder containing a stack of 16-bit TIFF images
	2.	The GUI will compute ROI stats, plot SNR graphs, and export results
	3.	Output saved to output/ folder in the selected directory

##📊 Outputs

	•	📈 snr_signal.png: SNR vs Signal (DN)
	•	📉 snr_exposure.png: SNR vs Exposure Ratio
        •       🟢 prnu_fit.png: PRNU regression
        •       🗺 dsnu_map.png: DSNU map
        •       🗺 readnoise_map.png: Read noise map
        •       🗺 prnu_residual_map.png: PRNU residual map
        •       📋 summary.txt: Key evaluation metrics
        •       📄 report.html: Embedded HTML report
        •       📌 Metrics include SNR @ 50% and DN @ SNR=1 (0 dB)
        •       📑 roi_stats.csv: (Planned) Per-condition stats

##🔮 Planned Features

	•	ROI/pixel mode switching in GUI
	•	Configurable pipeline via default_config.json
	•	Support for multi-gain and multi-exposure batch evaluation
	•	PRNU and black-level correction logic
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

