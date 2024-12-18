import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance

# ディレクトリのパスと時間範囲を指定
directory_path = "data/csv/wt2022"  # CSVファイルが入ったディレクトリのパス
before_start_time = "2022-08-03 16:50:00"  # 前時間範囲の開始
before_end_time = "2022-08-03 17:50:00"  # 前時間範囲の終了
after_start_time = "2022-08-03 17:50:00"  # 後時間範囲の開始
after_end_time = "2022-08-03 18:50:00"  # 後時間範囲の終了
feature = "most_port"  # ヒストグラムを作成する特徴量
x_lim = 0, 1
y_lim = 0, 300

# 空のリストを作成してデータを格納
before_data = []
after_data = []

# スケーラーの初期化
scaler = MinMaxScaler()

# ディレクトリ内のすべてのCSVファイルを一度だけ処理
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".csv"):  # CSVファイルのみ対象
        file_path = os.path.join(directory_path, filename)
        # CSVを読み込む
        df = pd.read_csv(file_path)
        # daytime列をdatetime型に変換
        df['daytime'] = pd.to_datetime(df['daytime'])

        # 時間範囲でフィルタリング
        before_filtered = df[(df['daytime'] >= before_start_time) & (df['daytime'] <= before_end_time)]
        after_filtered = df[(df['daytime'] >= after_start_time) & (df['daytime'] <= after_end_time)]

        # 各時間範囲のデータをリストに追加
        before_data.append(before_filtered)
        after_data.append(after_filtered)

# 各データセットを1つのDataFrameに結合
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

# 前後のデータを結合してスケーリング
combined_data = pd.concat([before_combined_data, after_combined_data], ignore_index=True)
combined_data[feature] = scaler.fit_transform(combined_data[[feature]])

# スケーリング後に時間範囲ごとにデータをフィルタリング
before_combined_data = combined_data[
    (combined_data['daytime'] >= before_start_time) & (combined_data['daytime'] <= before_end_time)]
after_combined_data = combined_data[
    (combined_data['daytime'] >= after_start_time) & (combined_data['daytime'] <= after_end_time)]

# 前時間範囲のデータでヒストグラムを作成（色: 青）
plt.figure(figsize=(10, 6))
plt.hist(before_combined_data[feature], bins=100, color='blue', edgecolor='black')
plt.xlim(x_lim)
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.ylim(y_lim)
plt.title('Before Time Range Histogram')
plt.grid(True)
plt.show()

# 後時間範囲のデータでヒストグラムを作成（色: 赤）
plt.figure(figsize=(10, 6))
plt.hist(after_combined_data[feature], bins=100, color='red', edgecolor='black')
plt.xlim(x_lim)
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.ylim(y_lim)
plt.title('After Time Range Histogram')
plt.grid(True)
plt.show()

# Wasserstein距離を計算
wasserstein_dist = wasserstein_distance(
    before_combined_data[feature].values,
    after_combined_data[feature].values
)

print(f"Wasserstein Distance between distributions: {wasserstein_dist}")
