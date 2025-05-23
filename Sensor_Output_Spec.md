## センサ評価レポート出力項目仕様（2025年版・暫定）

### 📝 センサ共通メタ情報

* Sensor Name（型番やID）
* 使用レンズ（任意）
* 評価日、評価環境
* ゲインリスト：\[0dB, 6dB, 12dB, 24dB] # <- 6dBおきに取った方望ましいため18dBも追加
* 露光倍率リスト：\[1/16, 1/8, 1/4, 1/2, 1, 2, 4]
* 各条件における画像枚数（例：10）

## ※ `graychart_snr.png` は補助的可視化として残しているが、グレーチャートROIを `snr_signal.png` に統合することで一本化可能。

### 📄 出力ファイル一覧（センサごと）

※ この一覧は現在の実装・評価仕様に対応しているようですが、必要なファイルや処理がすべてここに網羅されているか最終確認を検討してもよいかもしれません。

※ 各指標の算出方法と関連config項目は以下に明記します

※ 各指標の算出方法は以下に明記します（関連configキー対応）

#### 1. `sensor_summary.csv`

※ `Sensitivity (e⁻/μW·cm²·s)` は conversion gain（e⁻/DN）が未定のため出力対象外とする。 代替として `System Sensitivity (DN / μW·cm²·s)` を出力し、レンズ込みの実効感度とする。

* **System Sensitivity (DN / μW·cm⁻²·s)**：Gain 0dB・Exp95% 時のフラット画像ROIの平均DNを、照度（μW/cm²）×露光時間（秒）で割って算出。

  * conversion gain が不明なため DN 単位の感度として扱う。
  * レンズによる透過率や光学減衰も含めた「システム実効感度」となる。
  * 使用項目：

    * illumination.power\_uW\_cm2
    * reference.exposure\_ms
    * フラット画像（gain\_0dB/chart\_1x）ROI平均DN
  * ※ 感度算出には DN\_sat は使用せず、Exp95%画像の平均値を用いる。DN\_sat 近傍は非線形性やノイズが増大するため不適。
* **Pseudo PRNU (%)**：フラット画像（各露光条件）において、ROI内の各画素について時間方向（10枚）の標準偏差（σ）を算出し、それを空間方向に統計化（config.processing.stat\_mode に従い rms/mean/mad）する。さらに、ROI平均信号値で割って百分率表示（σ/μ × 100 \[%]）。

  * ゲイン補正の有無：config.processing.apply\_gain\_map
  * フィッティング法：config.processing.prnu\_fit（"LS" or "WLS"）※ μ-σ回帰を行う場合に適用
  * 使用回帰：config.processing.prnu\_fit（"LS" or "WLS"）
    ※ Pseudo PRNU は処理構造として DSNU とほぼ同様であり、画像スタックに対して時間方向の統計を取り、空間方向に代表値（RMSなど）を算出する点で一致する。ただし PRNU は出力をROI内平均信号値で正規化する（σ/μ × 100 \[%]）。
* **DSNU (DN)**：遮光画像10枚を平均 → 各画素値の空間方向標準偏差（ROI内）を計算し、config.processing.stat\_mode に従って代表値を算出。
* **Read Noise (DN)**：遮光画像10枚の時間方向標準偏差（各画素の時系列におけるstd）を計算し、空間方向にまとめる際に config.processing.stat\_mode に従って代表値（rms/mean/mad）を算出。

  * 使用モード：config.processing.read\_noise\_mode
  * 差分法との切替（将来対応）：config.processing.read\_noise\_mode == 2 などで分岐予定
* **Dynamic Range (dB)**：最大信号値として DN\_sat を用い、Read Noise (DN) との比から 20\*log10(DN\_sat / Noise) を算出。

  * DN\_satの基準：config.reference.sat\_factor
* **DN\_sat (飽和DN)**：以下3方式の最大を採用：

  1. フラット画像スタックの 99.9 パーセンタイル値
  2. 最大DN値 / config.reference.sat\_factor
  3. ADC最大値 × 0.90（将来config化）
* **SNR @ 50%**：グレーチャートまたはフラット画像で、Full-Wellの50%（例：32768 DN）に最も近いμとSNRの系列から、補間または回帰により推定して算出。
* **DN @ SNR=10dB**：SNRカーブから、SNRが10dB（config.processing.snr\_threshold\_dB）を超える最小DN値を補間または回帰で推定。
* **DN @ SNR=1 (0 dB)**：SNRが1となる最小信号レベル（ノイズと等価）を補間または回帰で推定。

#### 2. `roi_stats.csv`

| ROI Type Gain (dB) Exposure Level Mean Std SNR (dB) Type |    |    |   |      |      |      |      |
| -------------------------------------------------------- | -- | -- | - | ---- | ---- | ---- | ---- |
| flat                                                     | 0  | x1 | – | 932  | 15.3 | 35.7 | Flat |
| gray                                                     | 12 | x1 | 2 | 68.3 | 10.1 | 13.6 | Gray |
| dark                                                     | 0  | x1 | – | 2.9  | 1.1  | –    | Dark |

---

### 📊 グラフ・可視化

* `snr_signal.png`：SNR vs Mean Signal（log-log軸）

  * 横軸：ROI平均信号（DN）
  * 対象：Flat画像と Graychart画像の両方のROIを使用してμ-SNR系列を構成
  * 縦軸：SNR（dB）
  * 系列：露光倍率で色分け、ゲインで重ね描き
  * 補助線：理想曲線（√μ）、SNR=10dBライン
* `snr_exposure.png`：SNR vs Exposure Time（中間階調）

  * 横軸：露光時間（ms）。Graychart画像の他階調ROIも、中間階調に対するμの比から「擬似露光倍率」として換算し、クラウド状にマッピング可能。
  * 縦軸：SNR（dB）
  * 対象：gray ROIの中央階調（config.reference.roi\_mid\_index）
  * 系列：ゲインごとに色分け
* `graychart_snr.png`：※将来的に `snr_signal.png` に統合可能（下記注釈参照）
* `prnu_fit.png`：μまたはμ² に対する標準偏差の回帰グラフ

  * 横軸：平均輝度（μ または μ²）
  * 縦軸：標準偏差 σ または σ²
  * 系列：ゲインごとに色分け（露光倍率も含めた全条件を使用）
  * 回帰法：OLS または WLS（config.processing.prnu\_fit）
* `dsnu_map.png`：DSNU（遮光画像の画素間オフセット）マップ

  * 表示：ROI領域の空間分布
  * カラーマップ：ヒートマップ（logスケール／標準化あり）
* `prnu_residual_map.png`：ゲイン補正後の固定パターンノイズ（空間ばらつき）

  * 表示：Flat画像スタックの残差の空間分布（std or RMS）
  * 補正前後で比較可能にしてもよい
* `readnoise_map.png`：読み出しノイズの空間分布（Dark画像差分）

  * 表示：ROI内での時間方向標準偏差（各画素ごと）
  * 合成方法：10枚からのstd、または差分法
  * カラーマップ：ヒートマップ

---

### 🧾 HTMLレポート構成（`report.html`）

1. Summary 表
2. SNR vs Signal グラフ
3. SNR vs Exposure グラフ
4. Graychart 可視階調分析　　　　　#-> 削除(以降の番号ずれる)
5. 残差ノイズ傾向（pseudo PRNU）
6. Noise Maps（DSNU / Residual / ReadNoise）
7. 備考・注釈

※ PNGは base64 埋め込みに対応

---

###

## 📂 config.yaml の配置ルールと GUI フロー

* **配置場所**: 評価対象 *project フォルダ* のトップレベルに `config.yaml` を置く。
* **Select ボタン**: GUI で *project フォルダ* を選択すると、`config.yaml` を自動ロードし直ちに計算を実行する。
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
    0.0125: { folder: chart_x0.125 }
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
  mask_upper_margin: 0.85     # 飽和 DN_sat の 90 % 未満を回帰に使う
  stat_mode: rms              # 'mean', 'rms', 'mad' から選択可能（分散合成・代表値算出に使用）
  snr_threshold_dB: 10       # SNR評価での可視限界（10dB など）
  min_sig_factor: 3           # σ_read の n倍以上を有効信号とみなす
  apply_gain_map : false      # PRNUの算出時gain_map補正をするか
  plane_fit_order: 2          # ROI内傾斜補正次数
  read_noise_mode: 0          # 差分法との切り替え
  prnu_fit: LS                # LS:最小二乗法 WLS:加重最小二乗法

plot:
  exposures: [1.0, 0.0625]    # 図1に描画する露光倍率
  dummy: 0.0

output:
  report_html: true           # レポートhtmlを自動的に書き出す
  report_csv: true            # レポートcsvを自動的に書き出す
  debug_stacks: true          # (デバッグ用)stack_cache, dark_cache, flat_cacheの内容を画像ファイルとして保存する

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

|   |
| - |

---

###

###

### 📥 入力前提

* フラット画像: フラットチャート中央の均一領域（200x200px程度）
* グレーチャート: 11階調チャート（階調別ROI）
* ダーク画像: 完全遮光環境で撮影
* 各露光倍率ごとに10枚（撮影済）
* ROI指定はImageJ形式（.zip）

---

### 📌 特記事項

* "PRNU" は厳密には計測不可のため、ゲイン補正後の空間残差として "pseudo PRNU" を表示
* グレーチャートは視認可能階調を評価する目的で使用（SNR > 10dBを視認可能と定義）
* DSNU/ReadNoiseは完全遮光下で測定されるため、実信号と混じらない
* 全画像処理後、各種レポートと可視化を自動生成
