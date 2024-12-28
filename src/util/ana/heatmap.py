import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CSVファイルを読み込む
# ファイル名を適宜変更すること
file_path = '/mnt/nas0/g005/murasemaru/exp/other/2201H/averaged_w.csv'  # ここにCSVファイルのパスを指定
output_path = f'{os.path.dirname(file_path)}/{os.path.basename(file_path)}_heatmap.png'  # 保存する画像ファイルのパス
df = pd.read_csv(file_path, index_col=0)

# ヒートマップの作成
plt.figure(figsize=(12, 8))
sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.2f', cbar=True)

# グラフのタイトルとラベル
plt.title('Heatmap of CSV Data', fontsize=16)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Rows', fontsize=12)

# グラフの保存
plt.tight_layout()
plt.savefig(output_path, dpi=300)  # DPIを指定して高解像度で保存
plt.close()  # メモリを解放

print(f"ヒートマップを {output_path} に保存しました．")