import os

import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読みこむ
file_path = "/mnt/nas0/g005/murasemaru/exp/other/obs/2201AusEast/drift_obs.csv"  # CSVファイルのパスを指定
data = pd.read_csv(file_path)
metrix = ["f1_score"]
output_dir = f"{os.path.dirname(file_path)}/img"
output_path = f"{output_dir}/{','.join(metrix)}.png"
os.makedirs(output_dir, exist_ok=True)

label_size = 22
ticks_size = 16
legend_size = 22

# datetime列をdatetime型に変換（必要に応じて）
data['daytime'] = pd.to_datetime(data['daytime'])

# グラフを作成
plt.figure(figsize=(12, 8))  # グラフ全体を少し大きく設定

for y in metrix:
    plt.plot(data['daytime'], data[y], label=y, linewidth=2)

# 軸ラベルとタイトルの設定（文字サイズを調整）
plt.xlabel('daytime', fontsize=label_size)
plt.ylabel(",".join(metrix), fontsize=label_size)

# 軸目盛りの文字サイズを調整
plt.xticks(fontsize=ticks_size, rotation=45)  # 横軸ラベルを回転
plt.yticks(fontsize=ticks_size)

# 凡例とグリッドを追加（凡例の文字サイズを調整）
plt.legend(fontsize=legend_size)
plt.grid(True)

# グラフを表示
plt.tight_layout()  # レイアウト調整
plt.savefig(output_path, dpi=300)  # 高解像度で保存
plt.show()
