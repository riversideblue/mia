import os
import pandas as pd
import matplotlib.pyplot as plt

# Drift Detection (mode obs)の結果を比較
all_dir_path = "/mnt/nas0/g005/murasemaru/exp/4_DriftDetection/cos/target"
metrix = "math"

output_dir = f"{all_dir_path}/res_img"
os.makedirs(output_dir, exist_ok=True)

label_size = 22
ticks_size = 16
legend_size = 22

dfs = pd.DataFrame()

# データの読み込みと統合
for di in sorted(os.listdir(all_dir_path)):
    file_path = os.path.join(all_dir_path, di, 'dd_res.csv')
    if not os.path.exists(file_path):  # ファイルが存在しない場合はスキップ
        print(f"File not found: {file_path}. Skipping.")
        continue

    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={metrix: di})  # カラム名をリネーム

        if dfs.empty:
            dfs = df[['date', di]]  # 初回はデータフレームを初期化
        else:
            dfs = pd.merge(dfs, df[['date', di]], on='date', how='outer')  # 日付を基準にマージ
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        continue

if not dfs.empty:
    print(dfs)

    # 各メトリクスごとにグラフを出力
    for y in dfs.columns[1:]:  # date以外の列をそれぞれプロット
        plt.figure(figsize=(12, 8))
        plt.plot(dfs['date'], dfs[y], label=y, linewidth=1.5, alpha=0.8)
        plt.title(f'Drift Detection Method: {y}', fontsize=22)
        plt.xlabel('Date', fontsize=label_size)
        plt.ylabel('Value', fontsize=label_size)
        plt.xticks(fontsize=ticks_size, rotation=45)
        plt.yticks(fontsize=ticks_size)
        plt.legend(fontsize=legend_size)
        plt.grid(True)
        plt.tight_layout()

        # ファイルに保存
        output_path = os.path.join(output_dir, f"{y}_plot.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
else:
    print("No data available for plotting.")
