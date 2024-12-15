import os
import pandas as pd
import matplotlib.pyplot as plt

# --- データセットの各特徴量が遷移する様子を一定期間プロットするスクリプト --------------------------------------------------------------------- #
beginning_dtime = "2022-01-01 09:03:08"
end_dtime = "2022-01-01 09:03:08"
target_dir_path = "src/main/traffic_data/csv/unprocessed/Lab01"
"""
metrix overview
"ex_address","in_address","daytime","rcv_packet_count","snd_packet_count","tcp_count","udp_count","most_port","port_count","rcv_max_interval","rcv_min_interval","rcv_max_length","rcv_min_length","snd_max_interval","snd_min_interval","snd_max_length","snd_min_length","label"
"""
metrix = ["ex_address","in_address","daytime","rcv_packet_count","snd_packet_count","tcp_count","udp_count","most_port","port_count","rcv_max_interval","rcv_min_interval","rcv_max_length","rcv_min_length","snd_max_interval","snd_min_interval","snd_max_length","snd_min_length","label"]
output_path = "output/traffic_data_plot.png"  # グラフ保存先
# ---------------------------------------------------------------------------------------------------------------------------------- #

# ディレクトリ内のすべてのCSVファイルをリスト化
csv_files = [os.path.join(target_dir_path, file) for file in sorted(os.listdir(target_dir_path)) if file.endswith('.csv')]

beginning_dtime = pd.to_datetime(beginning_dtime)
end_dtime = pd.to_datetime(end_dtime)

# 条件に合うデータを格納するリスト
filtered_data = []

# 各CSVファイルを処理
for file in csv_files:
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(file)
        
        # タイムスタンプ列をdatetime形式に変換
        df["daytime"] = pd.to_datetime(df["daytime"])
        
        # 条件に合うデータをフィルタリング
        filtered_df = df[(df["daytime"] >= beginning_dtime) & (df["daytime"] <= end_dtime)]
        
        # 条件に合うデータをリストに追加
        if not filtered_df.empty:
            filtered_data.append(filtered_df)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# すべての条件に合うデータを1つのDataFrameに結合
if filtered_data:
    result_df = pd.concat(filtered_data, ignore_index=True)
else:
    print("No data matched the given time range.")
    exit()

# グラフ保存先ディレクトリを作成
os.makedirs(os.path.dirname(output_path), exist_ok=True)

label_size = 22
ticks_size = 16
legend_size = 22

# グラフを作成
plt.figure(figsize=(12, 8))  # グラフ全体を少し大きく設定

for y in metrix:
    plt.plot(result_df['daytime'], result_df[y], label=y, linewidth=2)

# 軸ラベルとタイトルの設定（文字サイズを調整）
plt.xlabel('daytime', fontsize=label_size)
plt.ylabel(",".join(metrix), fontsize=label_size)

# 軸目盛りの文字サイズを調整
plt.xticks(fontsize=ticks_size, rotation=45)  # 横軸ラベルを回転
plt.yticks(fontsize=ticks_size)

# 凡例とグリッドを追加（凡例の文字サイズを調整）
plt.legend(fontsize=legend_size)
plt.grid(True)

# グラフを表示と保存
plt.tight_layout()  # レイアウト調整
plt.savefig(output_path, dpi=300)  # 高解像度で保存
plt.show()

