import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pytz
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance

# JSTの現在時刻を基にした保存ディレクトリ名
jst = pytz.timezone("Asia/Tokyo")
init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")
directory_path = "data/csv/wt2022"  # データディレクトリ
output_directory = f"exp/{init_time}"  # 保存先ディレクトリ
csv_output_path = f"{output_directory}/results.csv"
os.makedirs(output_directory, exist_ok=True)

# 時間範囲を指定
before_start_time = "2022-08-03 16:50:00"
before_end_time = "2022-08-03 17:50:00"
after_start_time = "2022-08-03 17:50:00"
after_end_time = "2022-08-03 18:50:00"

# ヒストグラム作成対象の特徴量リスト
features = [
    "rcv_packet_count", "snd_packet_count", "tcp_count", "udp_count",
    "most_port", "port_count", "rcv_max_interval", "rcv_min_interval",
    "rcv_max_length", "rcv_min_length", "snd_max_interval", "snd_min_interval",
    "snd_max_length", "snd_min_length"
]

# 空のリストを作成してデータを格納
before_data = []
after_data = []

# スケーラーの初期化
scaler = MinMaxScaler()

# ディレクトリ内のCSVファイルを処理
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".csv"):  # CSVファイルのみ対象
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)  # CSV読み込み
        df['daytime'] = pd.to_datetime(df['daytime'])  # datetime型に変換

        # 時間範囲でフィルタリング
        before_filtered = df[(df['daytime'] >= before_start_time) & (df['daytime'] <= before_end_time)]
        after_filtered = df[(df['daytime'] >= after_start_time) & (df['daytime'] <= after_end_time)]

        before_data.append(before_filtered)
        after_data.append(after_filtered)

# データの結合
if before_data:
    before_combined_data = pd.concat(before_data, ignore_index=True)
else:
    print("前時間範囲のデータが見つかりません．")
    exit()

if after_data:
    after_combined_data = pd.concat(after_data, ignore_index=True)
else:
    print("後時間範囲のデータが見つかりません．")
    exit()

# データセットを結合してスケーリング
combined_data = pd.concat([before_combined_data, after_combined_data], ignore_index=True)
wasserstein_results = []

# 各特徴量に対して処理を実行
for feature in features:
    if feature not in combined_data.columns:
        print(f"Feature {feature} is not in the dataset, skipping.")
        continue

    # スケーリング
    combined_data[feature] = scaler.fit_transform(combined_data[[feature]])

    # 時間範囲ごとにデータを分割
    before_combined_data = combined_data[
        (combined_data['daytime'] >= before_start_time) & (combined_data['daytime'] <= before_end_time)]
    after_combined_data = combined_data[
        (combined_data['daytime'] >= after_start_time) & (combined_data['daytime'] <= after_end_time)]

    # ヒストグラムの最大値を計算してylimを統一
    before_hist, _ = np.histogram(before_combined_data[feature], bins=100)
    after_hist, _ = np.histogram(after_combined_data[feature], bins=100)
    max_y = max(before_hist.max(), after_hist.max())  # 最大頻度を取得
    ylim = (0, max_y + 10)  # 少し余裕を持たせる

    # 前時間範囲のヒストグラムを作成
    plt.figure(figsize=(10, 6))
    plt.hist(before_combined_data[feature], bins=100, color='blue', edgecolor='black')
    plt.ylim(ylim)  # 縦幅を設定
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Before Time Range Histogram for {feature}')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, f"before_{feature}.png"))
    plt.close()

    # 後時間範囲のヒストグラムを作成
    plt.figure(figsize=(10, 6))
    plt.hist(after_combined_data[feature], bins=100, color='red', edgecolor='black')
    plt.ylim(ylim)  # 縦幅を設定
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'After Time Range Histogram for {feature}')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, f"after_{feature}.png"))
    plt.close()

    # Wasserstein距離を計算
    wasserstein_dist = wasserstein_distance(
        before_combined_data[feature].values,
        after_combined_data[feature].values
    )
    wasserstein_results.append({"Feature": feature, "Wasserstein Distance": wasserstein_dist})

# 結果をCSVに出力
wasserstein_df = pd.DataFrame(wasserstein_results)
wasserstein_df.to_csv(csv_output_path, index=False)
print(f"Wasserstein distances saved to {csv_output_path}")
