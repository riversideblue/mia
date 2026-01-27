# adids

軽量な実行手順（新しいマシンでも再現しやすい形）です。

## 前提
- OS: Linux想定
- Python: **3.11 系**（`/usr/bin/python3.11` が使えること）

Ubuntu / Debian の例:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

## セットアップ
リポジトリ直下で実行します。

```bash
make bootstrap
```

これは次を自動で行います。
- `.venv` の作成
- `requirements.txt` のインストール

## 実行

```bash
make run
```

`Makefile` が `.venv/bin/python` を優先して使うため、通常は `activate` 不要です。

## プロジェクト構成
主要なディレクトリだけを抜粋しています。

```text
adids/
├─ Makefile                # bootstrap / run / log-to-csv
├─ requirements.txt        # 依存の固定
├─ src/
│  ├─ main/                # 実行本体（make run の入口）
│  │  ├─ Run.py
│  │  ├─ settings.json
│  │  ├─ SessionController.py
│  │  └─ SessionDefiner.py
│  ├─ util/                # 前処理・変換などのユーティリティ
│  │  └─ LogToCsvExtractor.py
│  └─ test/                # テスト・検証用
├─ data/
│  ├─ csv/                 # 学習・評価用CSVの置き場
│  ├─ logs/                # Zeekログなど
│  └─ pcap/                # PCAPの置き場
└─ exp/                    # 実行結果の出力先
```

## データ配置（重要）
学習・評価に使う CSV は `data/csv` 配下に置いてください。

- 例: `data/csv/your_dataset.csv`

現状の設定ファイルは次を参照します。
- `src/main/settings.json:45`

CSV が空でも `make run` 自体は終了しますが、学習は実質行われません。

## 追加コマンド

### ZeekログからCSVを作る
ZeekのJSONログがある場合:

```bash
make log-to-csv \
  LOG_DIR=data/logs/<log_dir> \
  OUT_CSV=data/csv/<name>/conn.csv \
  NETWORK_KEY=202304 \
  PATTERN=conn.log
```

注意:
- `src/util/LogToCsvExtractor.py` は **JSON行形式** のZeekログを想定しています。
- CSVなど別形式のログにはそのまま使えません。

### 仮想環境を手で触りたいとき

activate派:

```bash
source .venv/bin/activate
python -m pip install <package>
```

activateしない派:

```bash
.venv/bin/python -m pip install <package>
```

## よくある詰まりどころ
- `ModuleNotFoundError` が出たら:
  - まず `make bootstrap` を実行
- `/mnt/c` などのパスで失敗したら:
  - `src/main/settings.json` の `USER_DIR` / `DATASETS_DIR_PATH` をローカルのパスに直す

## 変更したファイル（この手順に関係）
- `Makefile:1`
- `requirements.txt:1`
- `src/main/settings.json:7`
- `src/main/SettingsLoader.py:28`
