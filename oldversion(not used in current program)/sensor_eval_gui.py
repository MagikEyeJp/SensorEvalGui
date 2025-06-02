"""
Sensor evaluator GUI – DSNU-subtracted, multi-exposure
=====================================================

機能
* プロジェクトフォルダ選択 → Run で全露光セットを自動解析
* DSNU (dark-average) を全チャートから減算
* ROI 毎 Mean/Std → SNR を算出
* 読出しノイズ σ_read は dark スタックの StdDev 平均
* 2 つのプロットを表示／PNG 保存
    1) SNR vs Signal (ログ X、全露光重ね描き)
    2) SNR vs Exposure (mid-gray、Measured + Ideal √k)
* summary.txt、roi_stats.csv、report.html を自動生成
依存: PySide6 matplotlib numpy pandas imageio roifile jinja2
"""

from __future__ import annotations
import math, glob, pathlib, base64, sys, traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import imageio.v3 as iio
import roifile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from jinja2 import Template
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QTextEdit,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSplitter,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
)
from PySide6 import QtCore
import yaml
from datetime import datetime

# ------------ パラメータ ------------
FULL_WELL_DN = 65535
# EXPOSURE_FOLDERS: Dict[float, str] = {
#     2.00: "chart_2x",
#     1.00: "chart_1x",
#     0.50: "chart_0.5x",
#     0.25: "chart_0.25x",
#     0.125: "chart_0.125x",
#     0.0625: "chart_0.0625x",
# }
CONFIG_YAML = "config.yaml"
ROI_ZIP_NAME = "roi.zip"
ROI_MID_INDEX = 2  # mid-gray ROI index
IMG_PATTERNS = ("*.tif", "*.tiff")
# ---- thresholds for domain colouring ----
RN_FACTOR = 5  # < σ_read × 5  → read-noise domain
IDEAL_RATIO = 0.90  # < 90 % of ideal √k  → dark-current / headroom limit
SLOPE_TH = 0.30  # 勾配 < 0.3 → dark-current/headroom


# ------------ ヘルパ ------------
def load_stack(folder: pathlib.Path) -> np.ndarray:
    files: List[str] = []
    for pat in IMG_PATTERNS:
        files += glob.glob(str(folder / pat))
    if not files:
        raise FileNotFoundError(f"No TIFF images in {folder}")
    return np.stack([iio.imread(f) for f in sorted(files)]).astype(np.int32)


def load_rois(zip_path: pathlib.Path):
    rects = []
    for r in roifile.roiread(zip_path):
        l, t = int(r.left), int(r.top)
        w = int(getattr(r, "width", r.right - r.left))
        h = int(getattr(r, "height", r.bottom - r.top))
        rects.append((l, t, w, h))
    if not rects:
        raise ValueError("roi.zip has no rectangles")
    return rects


def roi_stats(stack: np.ndarray, roi):
    x, y, w, h = roi
    region = stack[:, y : y + h, x : x + w]
    m = region.mean((1, 2))
    s = region.std((1, 2))
    return float(m.mean()), float(np.sqrt((s**2).mean()))


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ------------ ワーカー ------------
class EvalWorker(QtCore.QThread):
    finished = QtCore.Signal(str, object, object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)

    def __init__(self, project: pathlib.Path):
        super().__init__()
        self.project = project

    def run(self):
        try:
            txt, f1, f2 = self._evaluate()
            self.finished.emit(txt, f1, f2)
        except Exception:
            self.error.emit(traceback.format_exc())

    def _evaluate(self):
        # 解析用関数
        def load_corrected(folder: pathlib.Path, dark_avg: np.ndarray) -> np.ndarray:
            """
            DSNU を減算して 0–FullWell にクリップしたスタックを返す
            """
            st = load_stack(folder) - dark_avg  # DSNU 補正
            st = np.clip(st, 0, FULL_WELL_DN)
            return st.astype(np.int32)

        def roi_stats_batch(stack: np.ndarray, rects):
            # stack: (N, H, W)
            means, stds = [], []
            for x, y, w, h in rects:
                reg = stack[:, y : y + h, x : x + w]
                means.append(reg.mean())
                stds.append(reg.std())
            return np.array(means), np.array(stds)

        # レポート関数
        def _write_report(root: Path, gain_defs, figs_saved):
            """
            root        : プロジェクトフォルダ Path
            gain_defs   : config.yaml の gains リスト（dict の list）
            figs_saved  : {gain_name: (fig1_fname, fig2_fname)} dict
            """
            html = [
                "<html><head><meta charset='utf-8'><title>Sensor Report</title>",
                "<style>body{font-family:sans-serif} h2{margin-top:40px}</style>",
                "</head><body>",
                f"<h1>Sensor evaluation — {datetime.now():%Y-%m-%d %H:%M}</h1>",
            ]

            for g in gain_defs:
                gname = g["name"]
                html.append(f"<h2>{gname}</h2>")
                for fname in figs_saved.get(gname, ()):
                    if (root / fname).exists():
                        html.append(f"<img src='{fname}' width='600'><br>")
                    else:
                        html.append(f"<p style='color:red'>⚠ {fname} not found</p>")
            html += ["</body></html>"]
            (root / "report.html").write_text("\n".join(html), encoding="utf-8")

        # evaluate
        root = self.project
        self.progress.emit(0)
        rois = load_rois(root / ROI_ZIP_NAME)
        self.progress.emit(5)
        cfg_path = root / CONFIG_YAML
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}  # ← 無い場合は空 dict

        # 既定値
        default_gains = [{"name": "gain_0dB", "db": 0}]
        default_exps = {
            2.0: "chart_2x",
            1.0: "chart_1x",
            0.5: "chart_0.5x",
            0.25: "chart_0.25x",
            0.125: "chart_0.125x",
            0.0625: "chart_0.0625x",
        }

        gain_defs = cfg.get("gains", default_gains)
        exp_folders = cfg.get("exposures", default_exps)

        dark_avg = {}
        stack_cache = {}
        read_noise = {}
        figs_saved = {}
        summary = ""

        for g in gain_defs:
            gpath = root / g["name"]
            if not gpath.exists():
                print(f"[warn] skip missing folder {gpath}")
                continue

            # ダーク平均 → DSNU
            dark_stack = load_stack(gpath / "dark")
            dark_avg[gpath.name] = dark_stack.mean(0)
            _, read_noise[gpath.name] = roi_stats(dark_stack, rois[ROI_MID_INDEX])
            self.progress.emit(15)  # rn = σ_read (DN rms)

            stack_cache = {}
            exist_fac = []
            for fac, folder in exp_folders.items():
                fpath = gpath / folder
                if not fpath.exists():
                    print(f"[warn] skip missing folder {fpath}")
                    continue
                exist_fac.append(fac)
                key = (gpath.name, fac)
                stack_cache[key] = load_corrected(gpath / folder, dark_avg[gpath.name])

            stats_cache: dict[float, dict[str, np.ndarray]] = (
                {}
            )  # キー = 露光倍率 (fac)     値 = {"mean":…, "std":…, "lin":…, "snr":…}
            for fac in exist_fac:
                key = (gpath.name, fac)
                # DSNU 減算済スタック (R, N)
                st = stack_cache[key]  # DSNU 減算済スタック (R, N)
                meansR, stdsR = roi_stats_batch(st, rois)  # 形状 (R, N)
                print(f"roi_stats_batch({fac}): {meansR.shape}, {stdsR.shape}")
                linR = meansR / stdsR
                snrR = 20 * np.log10(linR, where=stdsR > 0, out=np.zeros_like(linR))
                stats_cache[key] = dict(mean=meansR, std=stdsR, lin=linR, snr=snrR)

            # ROI 統計 (Exp1x)
            rows = []
            sig_all = []
            snr_all = []
            for i, r in enumerate(rois):
                data = stats_cache[(gpath.name, 1.00)]
                rows.append(
                    dict(
                        ROI=i,
                        Mean=data["mean"][i],
                        Std=data["std"][i],
                        SNR_dB=data["snr"][i],
                    )
                )
            df = pd.DataFrame(rows)
            df.to_csv(root / f"roi_stats_{gpath.name}.csv", index=False)
            self.progress.emit(35)

            # Full-Well
            flat_mean, _ = roi_stats(
                load_corrected(gpath / "flat", dark_avg[gpath.name]),
                rois[ROI_MID_INDEX],
            )
            dn_sat = flat_mean / 0.95
            dr_fw = 20 * math.log10(dn_sat / read_noise[gpath.name])

            s_txt = (
                f"Gain: {gpath.name} ({g['db']} dB) --------------------\n"
                f"Read noise : {read_noise[gpath.name]:.2f} DN rms\n"
                f"DN_sat     : {dn_sat:.0f} DN (≈{dn_sat/FULL_WELL_DN*100:.1f}% of 16-bit)\n"
                f"DR_FW      : {dr_fw:.1f} dB\n"
            )
            summary += s_txt

        # ===== グラフ 1: SNR-Signal （各露光を色分け＋領域着色） =====
        fig1, ax1 = plt.subplots()
        ls_map = {"gain_0dB": "-", "gain_6dB": "--", "gain_12dB": ":"}
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for g in gain_defs:
            for idx, fac in enumerate(sorted(exist_fac, reverse=True)):
                data = stats_cache[(g["name"], fac)]
                sig = data["mean"]
                snr_lin = data["lin"]
                snr_db = data["snr"]

                # ---- log-log 勾配（中央差分） ---
                slope = np.gradient(np.log10(snr_lin), np.log10(sig))

                # ---- ドメイン分類ごとのマーカー色／形 ----
                clr, mkr = [], []
                for mu, lin, sl in zip(sig, snr_lin, slope):
                    if mu < RN_FACTOR * read_noise[g["name"]]:  # read-noise
                        clr.append("tab:cyan")
                        mkr.append("v")  # read-noise
                    elif sl < SLOPE_TH:
                        clr.append("tab:red")
                        mkr.append("^")  # dark-current
                    else:
                        clr.append(colors[idx % len(colors)])
                        mkr.append("o")  # shot-noise

                # プロット（点ごと形状）
                for x, y, c, mk in zip(sig, snr_db, clr, mkr):
                    ax1.scatter(
                        x,
                        y,
                        c=c,
                        marker=mk,
                        s=36,
                        edgecolors="k" if mk != "o" else "none",
                    )

                # シリーズライン（薄色）
                ax1.semilogx(
                    sig,
                    snr_db,
                    "-",
                    color=colors[idx % len(colors)],
                    alpha=0.3,
                    label=f"{g['name']} ×{fac}",
                )

        # 書式
        ax1.grid(which="both", ls="--", alpha=0.6)
        ax1.set_xlabel("Signal [DN]")
        ax1.set_ylabel("SNR [dB]")
        ax1.set_title("SNR vs Signal (DSNU subtracted)")

        # 凡例追加
        ax1.scatter([], [], c="tab:cyan", marker="v", label="read-noise domain", s=40)
        ax1.scatter([], [], c="gray", marker="o", label="shot-noise (ideal)", s=30)
        ax1.scatter(
            [],
            [],
            c="tab:red",
            marker="^",
            label="dark-current limit",
            s=40,
            edgecolors="k",
        )
        ax1.legend(fontsize=7)
        fig1.tight_layout()
        fname1 = f"snr_signal_{g['name']}.png"
        fig1.savefig(root / fname1, dpi=150)

        # ===== グラフ 2: SNR-Exposure (mid-gray) =================================
        fig2, ax2 = plt.subplots()

        gain_color = {
            "gain_0dB": "tab:blue",
            "gain_6dB": "tab:green",
            "gain_12dB": "tab:red",
        }
        for g in gain_defs:
            facs, snr_lin_mid, colors_exp = [], [], []
            for fac in sorted(exist_fac):
                facs.append(fac)
                snr_lin_mid.append(stats_cache[(g["name"], fac)]["lin"][ROI_MID_INDEX])

            base = snr_lin_mid[facs.index(1.0)]
            ideal = [base * math.sqrt(f) for f in facs]

            # ① 全 ROI を灰色ドットで散布
            for i, r in enumerate(rois):  # 全パッチでループ
                lin = []
                for fac in sorted(exist_fac):
                    lin.append(stats_cache[(g["name"], fac)]["lin"][i])
                ax2.loglog(facs, lin, ".", color="lightgray", alpha=0.4, zorder=1)

            # ② mid-gray の測定線を上に重ねる
            # --- 判定しきい値（90 %）で色分け ---
            for x, y, y_ideal in zip(facs, snr_lin_mid, ideal):
                if y < 0.9 * y_ideal:  # 頭打ち（dark-current / 他ノイズ）
                    colors_exp.append("tab:red")
                else:  # 理想域
                    colors_exp.append("tab:blue")
            # --- プロット ---
            ax2.loglog(
                facs,
                snr_lin_mid,
                "o-",
                color=gain_color[g["name"]],
                zorder=3,
                label=g["name"],
            )
            ax2.scatter(facs, snr_lin_mid, c=colors_exp, s=40, zorder=3)
            ax2.loglog(
                facs, ideal, "k--", lw=1, label=f'{g["name"]} Ideal √k', zorder=2
            )

            # ③ 既存：SNR=1 横線
            ax2.axhline(
                1.0, ls="--", color="cyan", label="SNR = 1 (read-noise)", zorder=0
            )

        # 書式
        ax2.grid(which="both", ls="--", alpha=0.6)
        ax2.set_xlabel("Rel. exposure")
        ax2.set_ylabel("Linear SNR")
        ax2.set_title("SNR vs Exposure (mid-gray)")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fname2 = f"snr_exposure_{g['name']}.png"
        fig2.savefig(root / fname2, dpi=150)
        self.progress.emit(90)

        figs_saved.setdefault(g["name"], []).extend([fname1, fname2])

        (root / "summary.txt").write_text(summary)
        _write_report(root, gain_defs, figs_saved)

        return summary, fig1, fig2


# ------------ GUI ------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Evaluator Qt")
        self.resize(1200, 720)
        self.lbl = QLabel("Project: not selected")
        self.btn_sel = QPushButton("Select…")
        self.btn_sel.setShortcut("Ctrl+O")
        self.btn_run = QPushButton("Run")
        self.btn_run.setEnabled(False)
        self.btn_run.setShortcut("Ctrl+R")
        self.btn_run.setDefault(True)
        self.btn_run.setEnabled(False)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        hl = QHBoxLayout()
        [hl.addWidget(w) for w in (self.lbl, self.btn_sel, self.btn_run, self.pbar)]
        hl.addStretch()
        top = QWidget()
        top.setLayout(hl)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.canvas1 = FigureCanvas(plt.figure())
        self.canvas2 = FigureCanvas(plt.figure())
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas1.draw()
        self.canvas2.draw()
        pl = QHBoxLayout()
        pl.addWidget(self.canvas1)
        pl.addWidget(self.canvas2)
        plots = QWidget()
        plots.setLayout(pl)
        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.text)
        splitter.addWidget(plots)
        splitter.setSizes([200, 500])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(top)
        v.addWidget(splitter)
        self.setCentralWidget(central)
        self.btn_sel.clicked.connect(self.sel)
        self.btn_run.clicked.connect(self.run_eval)
        self.project: pathlib.Path | None = None
        self.worker = None

    def sel(self):
        p = QFileDialog.getExistingDirectory(self, "Select project")
        if p:
            self.project = pathlib.Path(p)
            self.lbl.setText(f"Project: {p}")
            self.btn_run.setEnabled((self.project / ROI_ZIP_NAME).exists())

    def run_eval(self):
        if not self.project:
            return
        self.btn_run.setEnabled(False)
        self.text.clear()
        self.pbar.setValue(0)
        self.worker = EvalWorker(self.project)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.finished.connect(self.done)
        self.worker.error.connect(self.err)
        self.worker.start()

    def done(self, summary, fig1, fig2):
        self.text.setPlainText(summary)
        self.canvas1.figure = fig1
        self.canvas1.draw()
        self.canvas2.figure = fig2
        self.canvas2.draw()
        self.btn_run.setEnabled(True)
        self.worker = None
        self.pbar.setValue(100)
        QtCore.QTimer.singleShot(0, self._refresh_canvas_geometry)

    def _refresh_canvas_geometry(self):
        # FigureCanvas の sizeHint を再評価させ、Splitter を広げる
        for canvas in (self.canvas1, self.canvas2):
            dpi = canvas.figure.dpi
            w = canvas.width() / dpi
            h = canvas.height() / dpi
            canvas.figure.set_size_inches(w, h, forward=True)
        self.canvas1.updateGeometry()
        self.canvas2.updateGeometry()

    def err(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.btn_run.setEnabled(True)
        self.worker = None
        self.pbar.setValue(0)

    def showEvent(self, event):
        print("showEvent")
        super().showEvent(event)
        # 画面が描画された直後 (次のイベントキュー) に再レイアウト
        QtCore.QTimer.singleShot(0, self._force_update)

    def _force_update(self):
        print("force_update")
        # splitter / 子キャンバスの sizePolicy が Expanding なら
        # ここで updateGeometry() するだけでフルサイズに広がる
        self.canvas1.updateGeometry()
        self.canvas2.updateGeometry()


# ------------ entry ------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
