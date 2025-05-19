# Sensor Evaluation – default_config.yaml (Spec-complete)
# *Sensor Output Spec* に準拠したデフォルト設定。プロジェクト固有の値は適宜書き換えてください。

sensor:
  name: Sony IMX273
  adc_bits: 10          # ADC 分解能 [bit]
  lsb_shift: 6          # デジタル LSB 右シフト量

environment:
  note: |
    LED : 940 nm CW
    Lens: LH2601A (IR-pass)
    Temp: 25 °C

measurement:
  gains:                # {folder: フォルダ名, db: ゲイン[dB]}
    0:   { folder: gain_0dB }
    6:   { folder: gain_6dB }
    12:  { folder: gain_12dB }
    24:  { folder: gain_24dB }
  exposures:            # {folder: サブフォルダ名, ratio: 露光倍率}
    4.0:    { folder: chart_4x }
    2.0:    { folder: chart_2x }
    1.0:    { folder: chart_1x }
    0.5:    { folder: chart_0.5x }
    0.25:   { folder: chart_0.25x }
    0.125:  { folder: chart_0.125x }
    0.0625: { folder: chart_0.0625x }
  flat_lens_folder: LensFlat
  flat_nolens_folder: flat
  dark_folder: dark
  flat_roi_file:  roi/flat.roi
  chart_roi_file: roi/chart_roi.zip
  roi_mid_index: 5               # Graychart 中間階調 (0–10)

illumination:
  power_uW_cm2: 67.92            # 入射光パワー
  exposure_ms: 960               # 基準露光時間 [ms]
  sat_factor: 0.95               # 飽和検出しきい値 (percentage)

processing:
  roi_default: [100, 100, 200, 200]   # (x, y, w, h)
  roi_zip_file: roi/chart_roi.zip      # ROI ZIP
  mask_upper_margin: 0.85        # 飽和マスク閾値 (DN_sat × 0.85)
  stat_mode: rms                 # rms / mean / mad
  snr_threshold_dB: 10           # 視認限界 SNR
  min_sig_factor: 3              # σ_read の n倍以上を有効信号
  apply_gain_map: false          # ゲインマップ補正
  plane_fit_order: 2             # 傾斜補正次数
  read_noise_mode: 0             # 0: std、1: 差分法
  prnu_fit: LS                   # LS or WLS

plot:
  color_by_exposure: false
  exposure_labels: [1×, 0.25×, 0.5×, 1×, 2×, 4×]

output:
  output_dir: output
  report_html: true              # HTML レポート出力
  report_csv: true               # CSV 出力
  debug_stacks: false            # デバッグ用スタック保存
  report_order:
    - Dynamic Range (dB)
    - SNR (max)
    - Read Noise
    - DN @ 20 dB
    - PRNU (%)
