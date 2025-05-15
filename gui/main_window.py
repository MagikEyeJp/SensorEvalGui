# gui/main_window.py

from PySide6.QtWidgets import QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QTextEdit
from pathlib import Path
from core.loader import load_image_stack
from core.analysis import extract_roi_stats
from core.plotting import plot_snr_vs_signal, plot_snr_vs_exposure
from core.report_gen import save_summary_txt, generate_html_report
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Evaluation GUI")

        self.folder_label = QLabel("No folder selected")
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)

        self.select_button = QPushButton("Select Folder")
        self.select_button.clicked.connect(self.select_folder)

        layout = QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.output_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.folder_label.setText(str(folder_path))

        # Load TIFF stack
        try:
            stack = load_image_stack(folder_path)
        except Exception as e:
            self.output_display.setPlainText(f"Error loading images: {e}")
            return

        # Analyze
        stats = extract_roi_stats(stack)

        # Plot SNR vs Signal (mock data for now)
        signal = np.linspace(100, 10000, 11)
        snr = signal / np.sqrt(signal)  # dummy SNR

        output_dir = folder_path / "output"
        output_dir.mkdir(exist_ok=True)

        snr_signal_path = output_dir / "snr_signal.png"
        snr_exposure_path = output_dir / "snr_exposure.png"

        plot_snr_vs_signal(signal, snr, snr_signal_path)
        plot_snr_vs_exposure(np.array([1/16, 1/8, 1/4, 1/2, 1, 2, 4]), snr, snr_exposure_path)

        save_summary_txt(stats, output_dir / "summary.txt")
        generate_html_report(stats, {
            "snr_signal": snr_signal_path,
            "snr_exposure": snr_exposure_path
        }, output_dir / "report.html")

        self.output_display.setPlainText("Evaluation complete. Output saved to: " + str(output_dir))