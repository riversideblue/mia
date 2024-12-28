import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# データセットの分布をすべての特徴量についてヒストグラム化
# -------------------------------------------------------------------------#
directory_path = "/mnt/nas0/g005/murasemaru/data/csv/modif/2202LabAll"  # CSVファイルが保存されているディレクトリ
output_dir = f"/mnt/nas0/g005/murasemaru/exp/other/data_obs/{os.path.basename(directory_path)}/histgram"  # 保存先ディレクトリ
columns_to_plot = [
    "rcv_packet_count", "snd_packet_count", "tcp_count", "udp_count", "most_port",
    "port_count", "rcv_max_interval", "rcv_min_interval", "rcv_max_length",
    "rcv_min_length", "snd_max_interval", "snd_min_interval", "snd_max_length",
    "snd_min_length", "label"
]  # ヒストグラムを作成したい列の名前
# -------------------------------------------------------------------------#

# ディレクトリ内のすべてのCSVファイルを取得
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
all_data = pd.DataFrame()  # 空のデータフレームを作成

# 各CSVファイルを読み込み、統合
for file in csv_files:
    try:
        df = pd.read_csv(file)
        # 空のデータフレームをスキップ
        if df.empty:
            print(f"File {file} is empty. Skipping...")
            continue
        # 全てがNaNの列を除外
        df = df.dropna(how='all', axis=1)
        all_data = pd.concat([all_data, df], ignore_index=True)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 保存ディレクトリを作成
os.makedirs(output_dir, exist_ok=True)

# 各特徴量についてヒストグラムを作成
for column_name in columns_to_plot:
    if column_name not in all_data.columns:
        print(f"Column {column_name} not found in the data. Skipping...")
        continue

    # データが数値型であることを確認
    if not pd.api.types.is_numeric_dtype(all_data[column_name]):
        print(f"Column {column_name} is not numeric. Skipping...")
        continue

    try:
        plt.figure(figsize=(10, 6))
        plt.hist(all_data[column_name].dropna(), bins=50, edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{column_name}_histogram.png"))
        plt.close()
        print(f"End processing: {column_name}")
    except Exception as e:
        print(f"Error processing column {column_name}: {e}")
