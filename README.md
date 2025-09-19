# NIDS-CDD

コンセプトドリフトを考慮した自己適応型侵入検知システム（NIDS-CDD）．

### Setup

実行環境はLinuxを想定。パッケージ群は`requirements.txt`に記載．以下コマンドでインストール可能．

```
pip install -r requirement.txt
```
Debian系ディストリビューションなどpip経由のインストールが制限されている場合は仮想環境の使用を推奨．
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r .devcontainer/requirements.txt
```

### Getting started

* 侵入検知システムの起動エントリーポイントは`src/main/Run.py`
    * 各種設定は`src/main/settings.json`に記述
    * トラヒックデータをリアルタイム学習→それが攻撃？or通常のどちらかを検知モデルが予測
    * 読み込むトラヒックデータは`data/`に保存
    * 実行結果は`exp/`に保存
* テストは`src/test`．ただ内容はほとんど無用
* `src/util`には資料用のグラフ等を描画するプログラムが入っている．
* トラヒックデータは`pcap`ファイルとして保存されており、[zeek](https://github.com/zeek/zeek)を使用してフロー特徴量を抽出



### Tree
```
.
├── README.md
├── requirements.txt
├── src
│   ├── main
│   │   ├── DriftDetection.py
│   │   ├── Evaluator.py
│   │   ├── ModelFactory.py
│   │   ├── Run.py
│   │   ├── SessionController.py
│   │   ├── SessionDefiner.py
│   │   ├── settings.json
│   │   ├── SettingsLoader.py
│   │   └── Trainer.py
│   ├── test
│   │   ├── 3d_matrix.py
│   │   ├── environment_variables.json
│   │   ├── flowtest.py
│   │   ├── gpuTest.ipynb
│   │   ├── gpuTest.py
│   │   ├── qtest.py
│   │   ├── results.json
│   │   ├── sumTest.py
│   │   ├── test_environ.py
│   │   ├── test.py
│   │   └── wireshark.csv
│   └── util
│       ├── 2025ieice_abs
│       │   ├── dd_method.py
│       │   └── fin.py
│       ├── 2025_mtg
│       │   └── output_eval.ipynb
│       ├── 2025_ns
│       │   ├── pst_eval.ipynb
│       │   ├── pst_test_data_drift.ipynb
│       │   ├── pst_wdcw.ipynb
│       │   ├── pst_wdk.ipynb
│       │   ├── pst_wdpw.ipynb
│       │   └── pst_whole_data_drift.ipynb
│       ├── b_thesis
│       │   ├── DDResult.py
│       │   ├── gplot.py
│       │   └── ntst.py
│       ├── b_thesis_abs
│       │   ├── CwPwDistance.py
│       │   ├── DataDrift.py
│       │   ├── dd_method1.py
│       │   ├── dd_method2.py
│       │   ├── dd_method3.py
│       │   └── fin.py
│       ├── dataset
│       │   ├── combiner.py
│       │   ├── deleteAttack.py
│       │   ├── injector.py
│       │   ├── timeOverride.py
│       │   └── TimeRangeFilter.py
│       ├── exp1
│       │   ├── distCompare.py
│       │   ├── driftPlotter_meanAll.py
│       │   ├── driftPlotter.py
│       │   ├── featureImportance.py
│       │   ├── heatmap.py
│       │   ├── histgram.py
│       │   ├── tsa2.py
│       │   ├── tsa3.py
│       │   ├── tsa4.py
│       │   └── tsa.py
│       ├── exp5
│       │   ├── combinedPlotter.py
│       │   ├── fin_nt_delete.py
│       │   └── metrixPlotter.py
│       ├── FE.py
│       ├── FE_settings.json
│       ├── graph
│       │   ├── comparePlotter.py
│       │   ├── CsvBasicPlotterWithDaytime.py
│       │   └── histPlotter.py
│       └── result
│           ├── DDResult.py
│           ├── evalComparator.py
│           ├── EvalMetrixByThreshold.py
│           ├── EvalMetrixByWindowSize.py
│           ├── resultCombiner.py
│           ├── TrainingCostByThreshold.py
│           └── TrainingCostByWindowSize.py
└── uml.asta

15 directories, 71 files
```