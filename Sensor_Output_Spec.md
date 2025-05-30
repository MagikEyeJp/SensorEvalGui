# センサ評価プログラム仕様（2025年版・暫定）

## 入力項目仕様

### 📝 設定、センサ共通メタ情報(config.yaml)

* Sensor Name（型番やID）
* 使用レンズ（任意）
* 評価日、評価環境
* ゲインリスト：\[0dB, 6dB, 12dB, 24dB] # <- 6dBおきに取った方望ましいため18dBも追加
* 露光倍率リスト：\[1/16, 1/8, 1/4, 1/2, 1, 2, 4]
* ADCのbit数,シフト数,フルスケール: ADC_bits, ADC_shift, ADC_FullScaleDN = ((2^ADC_bits)-1) * 2^ADC_shift
* 各条件における画像枚数（例：10）
* 各種設定 (後述)

### 📝 画像データ
Gainごとに下記画像セットを入力とする。それぞれ10枚以上。Gainと露出倍率の組み合わせはconfig.yamlにて定義(後述)
露光時間は露出95%(configで設定可)を基準(exposure_ms)とする。Gain0dB以外は、exposure_ms / Gain倍率の露光時間で撮ること
* フラット画像: レンズ付きフラット画像(露出:フルスケール95%など)
* ダーク画像: 完全遮光画像
* グレースケール画像:  11階調グレースケールチャート。レンズ付きフラット画像を撮った時の露光時間をx1倍とし、各種倍率の画像を用意
* フラット/ダーク画像: ROIファイル(ImageJ形式 .zip/.roi)
* グレースケール画像: ROIファイル(ImageJ形式 .zip), 階調数分のROIがある


## 出力項目仕様

### 📄 出力ファイル一覧（センサごと）

※ 各指標の算出方法と関連config項目は以下に明記します

※ 各指標の算出方法は以下に明記します（関連configキー対応）

#### 1. `summary.txt'
センサ共通メタ情報、測定条件を先頭に出力

Gainごとに下記項目を出力

* **System Sensitivity (DN / μW·cm⁻²·s)**：フラット画像(各露光条件) のフラットROIの平均DNを、照度 power_uW_cm2（μW/cm²）×露光時間 exposure_ms（ms）x 露光倍率 で割って計算する。
※ conversion gainが不明のため、DN感度 `System Sensitivity (DN / μW·cm²·s)` を出力し、レンズ込みの実効感度とする。
  * 使用項目：
    * power_uW_cm2 フラット画像のパワーメータ測定値
    * exposure_ms フラット画像の露出時間
    * フラット画像ROI平均DN
  * ※ 感度算出には DN\_sat は使用せず、Exp95%(sat_factorにて指定)で取得したフラット画像のフラットROIの平均値を用いる。DN\_sat 近傍は非線形性やノイズが増大するため不適。

* **Pseudo PRNU (%)**：フラット画像スタックを平均化し、ROI全体の平均を引いた残差マップを作成する。apply\_gain\_map が true の場合は plane\_fit\_order に従いゲインマップ補正を行った後に平均を取る。残差マップを config.processing.stat\_mode に従って統計化し、ROI 平均信号値で正規化して百分率表示する。
  ゲインマップは画像の照度ムラを補正するものであり、画像の一番明るい値で正規化して作成する
  * ゲインマップ補正の方法: gain_map_mode ("self_fit"or"flat_fit"or"flat_frame"or"none")
    * self_fitはそれぞれの画像スタック平均値を平面フィットさせてマップを作成
    * flat_fitはそのGainのフラット画像スタック平均値を平面フィットさせてマップを作成
    * flat_frameはそのGainのフラット画像スタックの平均値を正規化してマップを作成
    * noneはゲインマップ補正なし
  * フィット手法選択: gain_fit_method (poly|rbf)。rbfは計算時間が長くなるため注意
  * フィッティング法：config.processing.prnu\_fit（"LS" or "WLS"）※ μ-σ回帰を行う場合に適用
  * 使用回帰：config.processing.prnu\_fit（"LS" or "WLS"）
    ※ 平均フレームから ROI 平均を引いた残差の空間ばらつきを DSNU と同様の方法で統計化する。ただし PRNU は出力を ROI 平均信号値で正規化する（残差/μ × 100 \[%]）。
* **DSNU (DN)**：遮光画像10枚を平均 → 各画素値の空間方向標準偏差（ROI内）を計算し、config.processing.stat\_mode に従って代表値を算出。
* **Read Noise (DN)**：遮光画像10枚の時間方向標準偏差（各画素の時系列におけるstd）を計算し、空間方向にまとめる際に config.processing.stat\_mode に従って代表値（rms/mean/mad）を算出。
* **Black level (DN)**：ダーク画像スタックから ROI 平均値を求めたもの。ダーク補正に使用するスカラ値。

  * 使用モード：config.processing.read\_noise\_mode
    * 0：スタック全体の標準偏差から計算（デフォルト）
    * 1：フレーム間差分の標準偏差を √2 で割って計算
* **DN\_sat (飽和DN)**：以下3方式の最大を採用：

  1. フラット画像ROIの最大輝度ドットの 99.9 パーセンタイル値
  2. ADC_FullScaleDN * config.reference.sat\_factor
* **Dynamic Range (dB)**：最大信号値として DN\_sat を用い、Read Noise (DN) との比から 20\*log10(DN\_sat / Noise) を算出。
* **SNR @ 50%**：グレースケールチャートまたはフラット画像で、Full-Wellの50%（例：32768 DN）に最も近いμとSNRの系列から、補間または回帰により推定して算出。
  * DN\_satの基準：config.reference.sat\_factor
* **DN @ SNR=10dB**：SNRカーブから、SNRが10dB（config.processing.snr\_threshold\_dB）を超える最小DN値を補間または回帰で推定。
* **DN @ SNR=1 (0 dB)**：SNRが1となる最小信号レベル（ノイズと等価）を補間または回帰で推定。

#### 2. `roi_stats.csv`

各露光条件での統計値をcsvとして出力
| ROI Type| ROI No. |Gain (dB)| Exposure | Mean  | Std  | SNR (dB)| 
| --------|---------|---------|----------|-------|------|---------|
| dark    | -       |  0      | x1       | 32.3  | 15.9 | –       |
| flat    | -       |  0      | x1       |65300.1| 932  | 35.7    |
| grayscale| 0      |  0      | x1       | 205.9 | 68.3 |  5.6    |
| grayscale| 1      |  0      | x1       | 312.5 | 78.3 | 10.2    |
| grayscale| 2      | 0       | x1       | 455.2 | 88.3 | 15.3    |

---
### GUIレイアウト
```
GUIコンポーネント配置について
[Select Project Folderボタン][RUNボタン]
進捗ステータス行
--------- プログレスバー ----------------
---------------------------------------
| Summary.txt表示エリア                 |
| 必要に応じてスクロールバーあり           |
|                                     |
---------------------------------------  
-[タブ切り替え]--------------------------  <-ここの境界は上下にドラッグ可能
| グラフ表示エリア                       |
| グループ項目ごとにタブでページ切り替え     |
| 必要に応じてスクロールバーあり           |
|                                     |
---------------------------------------  

```


---

### 📊 グラフ・可視化
GUI上、および出力画像として出力

グループ SNR-Signal
* `snr_signal.png`：SNR vs Mean Signal（log-log軸）
  * 横軸：ROI平均信号（DN）
  * 対象：Flat画像と グレーチャート画像の両方のROIを使用してμ-SNR系列を構成
  * 縦軸：SNR（dB）
  * 系列：Gainごと色分け
  * 補助線：理想曲線（√μ）、SNR=10dBライン

グループ SNR-Exposure
* `snr_exposure.png`：SNR vs Exposure Time（中間階調）
  * 横軸：露光時間（ms）。Graychart画像の他階調ROIも、中央階調に対するμの比から「擬似露光倍率」として換算し、クラウド状にマッピング可能。
  * 縦軸：SNR（dB）
  * 対象：グレースケール ROIの中央階調（config.reference.roi\_mid\_index）
  * 系列：ゲインごとに色分け

グループ PRNU Fit
* `prnu_fit.png`：μまたはμ² に対する標準偏差の回帰グラフ
  * 横軸：平均輝度（μ または μ²）
  * 縦軸：標準偏差 σ または σ²
  * 単位はいずれも DN。`plot.prnu_squared: true` の場合は DN² 表示となる
  * 系列：ゲインごとに色分け（露光倍率も含めた全条件を使用）
  * 回帰法：OLS または WLS（config.processing.prnu\_fit）

グループ DNSU MAP
* `dsnu_map.png`：DSNU（遮光画像の画素間オフセット）マップ
  * 表示：ROI領域の空間分布
  * カラーマップ：ヒートマップ（logスケール／標準化あり）
* `dsnu_map_scaled.png`：DSNUマップ（0〜99パーセンタイルスケール）

グループ Readnoise MAP
* `readnoise_map.png`：読み出しノイズの空間分布（Dark画像差分）
  * 表示：ROI内での時間方向標準偏差（各画素ごと）
  * 合成方法：10枚からのstd、または差分法
  * カラーマップ：ヒートマップ
* `readnoise_map_scaled.png`：読み出しノイズマップ（0〜99パーセンタイルスケール）

グループ  PRNU Residual
* `prnu_residual_map.png`：ゲイン補正後の固定パターンノイズ（空間ばらつき）
  * 表示：Flat画像スタックの残差の空間分布（std or RMS）
  * 補正前後で比較可能にしてもよい
* `prnu_residual_map_scaled.png`：PRNU残差マップ（0〜99パーセンタイルスケール）
* `gain_map.png`：ゲインマップ（最大値で正規化）

グループ  ROI area
* `roi_area.png`：画像のROIの可視化
  * グレーチャート画像、フラット画像、ダーク画像を横に並べ、それぞれのROIを矩形線でオーバーレイ描画する。それぞれの画像は露出時間倍率x1、Gain0dBのものを使用 (なければ近いもの)

---

### 🧾 HTMLレポート構成（`report.html`）

 * 出力ファイルの内容を全て網羅する
    * Summary.txtの内容
    * 各グラフの内容

※ 画像PNGは base64 埋め込みに対応

---

###

## 📂 config.yaml の配置ルールと GUI フロー

* **配置場所**: 評価対象 *project フォルダ* のトップレベルに `config.yaml` を置く。
* **Select ボタン**: GUI で *project フォルダ* を選択すると、`config.yaml` を自動ロードし直ちに計算を実行(RUNボタンを押すのと等価)する。
* **RUN ボタン**: 設定を変更したり同じ project で再計算したい場合に手動でトリガーする。

```
sensor:
  name: Sony IMX273
  adc_bits:    10
  lsb_shift:    6

environment:
  note: |
    LED : 940 nm CW
    Lens: LH2601A (IR-pass)
    Temp: 25 °C

measurement:
  gains:
    0:   { folder: gain_0dB }
    6:   { folder: gain_6dB }
    12:  { folder: gain_12dB }
    24:  { folder: gain_24dB }

  exposures:
    4.0:    { folder: chart_x4 }
    2.0:    { folder: chart_x2 }
    1.0:    { folder: chart_x1 }
    0.5:    { folder: chart_x0.5 }
    0.25:   { folder: chart_x0.25 }
    0.125: { folder: chart_x0.125 }
    0.0625: { folder: chart_x0.0625 }

  flat_lens_folder: LensFlat
  flat_nolens_folder: flat
  dark_folder: dark
  flat_roi_file: flat.roi
  chart_roi_file: chart_roi.zip
  roi_mid_index: 5
   
illumination:
  power_uW_cm2: 67.92     
  sat_factor: 0.95         # フラット照度 ≒ 飽和 95 %（DN_sat 検出に使用）
  exposure_ms: 960         # 95%時の露出時間

processing:
  roi_zip_file: chart_roi.zip  # 既定ROI ZIPファイル
  stat_mode: rms              # 'mean', 'rms', 'mad' から選択可能（分散合成・代表値算出に使用）
  snr_threshold_dB: 10       # SNR評価での可視限界（10dB など）
  min_sig_factor: 3           # σ_read の n倍以上を有効信号とみなす
  mask_upper_margin: 0.85     # 飽和 DN_sat の 90 % 未満を回帰に使う
  mask_lower_margin: 0.0      # 飽和 DN_sat の一定割合以上を回帰に使う
  gain_map_mode : none        # self_fit | flat_fit | flat_frame | none  PRNUの算出時gain_map補正方法
  plane_fit_order: 2          # ROI内傾斜補正次数
  gain_fit_method: poly       # poly | rbf  フィッティング手法
  read_noise_mode: 0          # 0:スタックstd, 1:差分std/√2
  prnu_fit: LS                # LS:最小二乗法 WLS:加重最小二乗法
  exclude_abnormal_snr: true  # SNRが極端に低いROIを除外

plot:
  exposures: [1.0, 0.0625]    # 図1に描画する露光倍率
  color_by_exposure: false   # 露光倍率ごとに色分け
  exposure_labels: [3840,1920,960,480,240,120,60]  # 露光時間(ms)ラベル
  prnu_squared: false        # μ² vs σ² 表示
  dummy: 0.0

output:
  output_dir: output         # 出力ルートディレクトリ
  report_html: true           # レポートhtmlを自動的に書き出す
  report_csv: true            # レポートcsvを自動的に書き出す
  report_summary: true        # サマリーテキストを生成
  debug_stacks: true          # (デバッグ用)stack_cache, dark_cache, flat_cacheの内容を画像ファイルとして保存する
  report_order:
    - Dynamic Range (dB)
    - SNR @ 50% (dB)
    - DN @ 10 dB
    - DN @ 0 dB
    - Read Noise (DN)
    - Black level (DN)
    - DSNU (DN)
    - DN_sat
    - PRNU (%)
    - System Sensitivity

```

### 📥 フォルダ構成

典型的な project ディレクトリレイアウト例

（project\_〈SensorName〉/ を 測定条件のルートとする）

```
project_IMX273/                ← ここを “プロジェクトフォルダ” と呼ぶ
│
├─ config.yaml                 ← 共通設定（sat_factor など）
├─ roi/                        ← ROI 定義をまとめるフォルダ
│   ├─ chart.roi.zip           (グレーチャート用・矩形 11 個など)
│   └─ flat.roi.zip            (フラット用 1 枚・端を避けた矩形)
│
├─ gain_0dB/                   ← ゲイン別サブフォルダ（名前は自由）
│   ├─ dark/                   ← 完全暗状態（N 枚 TIFF）
│   │    ├─ dark_0001.tif
│   │    └─ …
│   ├─ lensflat/               ← レンズありフラット（Exp95 %）
│   │    ├─ flat_0001.tif
│   │    └─ …
│   └─ chart_x1/               ← グレースケールチャート（×1.0）
│        ├─ chart_0001.tif
│        └─ …
│
├─ gain_6dB/
│   ├─ dark/
│   ├─ lensflat/
│   ├─ chart_x1/               ← x1.0（＝“基準”露光）
│   ├─ chart_x2/               ← x2
│   ├─ chart_x0.5/             ← x1/2
│   ├─ chart_x0.25/            ← x1/4
│   └─ …
│
├─ gain_12dB/
│   └─ …（構成は同じ）
│
└─ gain_24dB/
    └─ …

```

---
### 📌 プログラムの細かい動作
 * Select Project Folderは、計算中はDisableにする
 * RUNボタンは、デフォルトDisableで、config.yamlが存在するprojectフォルダを指定されたら、Enableにする
 * 計算開始時、Summary表示エリアとグラフ表示エリアの中身を空にする
 * Summaryとグラフを表示したあと、ウィンドウの横幅をグラフの大きさに合わせて自動リサイズ。その時Summary表示エリアとグラフ表示エリアの間のSplitterは、位置の初期値を高さの比率が1:3くらいにする
 * 出力フォルダの中身がすでにあった場合、各ファイルは上書きする
 * コマンドラインでprojectフォルダを指定すると、起動してすぐそのフォルダを読み込んで計算を始める


---
### 📌 特記事項

* "PRNU" は厳密には計測不可のため、ゲイン補正後の空間残差として "pseudo PRNU" を表示
* グレーチャートは視認可能階調を評価する目的で使用（SNR > 10dBを視認可能と定義）
* DSNU/ReadNoiseは完全遮光下で測定されるため、実信号と混じらない
* 全画像処理後、各種レポートと可視化を自動生成
