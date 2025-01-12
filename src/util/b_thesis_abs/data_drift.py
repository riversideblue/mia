import os
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 入力パスと設定
drift_dir_path = "/mnt/nas0/g005/murasemaru/exp/1_DataAnalytics/drift/filtered_22020110-22020114/modif"
file_name = "data_drift"
metrix = "mean_dis"
label_size = 26
ticks_size = 20
legend_size = 26

# ハイライト窓の設定
highlight_start_1 = pd.Timestamp("2022-01-10 15:00:00")
highlight_end_1 = pd.Timestamp("2022-01-11 03:00:00")
highlight_start_2 = pd.Timestamp("2022-01-11 03:00:00")
highlight_end_2 = pd.Timestamp("2022-01-12 21:00:00")

# グラフの描画範囲設定
plot_start = pd.Timestamp("2022-01-10 12:00:00")
plot_end = pd.Timestamp("2022-01-13 00:00:00")

for di in os.listdir(drift_dir_path):
    di_path = f"{drift_dir_path}/{di}"
    output_dir_path = os.path.join(di_path, "res_img")
    os.makedirs(output_dir_path, exist_ok=True)

    for fi in os.listdir(di_path):
        if fi.endswith('.csv'):
            fi_path = os.path.join(di_path, fi)
            print(f"Processing file: {fi_path}")

            data = pd.read_csv(fi_path)
            ex_val = fi.replace("tsa_", "").replace(".csv", "")
            output_path = f"{output_dir_path}/{file_name}_{ex_val}.png"

            # 日付列をdatetime型に変換し、経過時間を計算
            data['date'] = pd.to_datetime(data['date'])
            data['elapsed_time'] = (data['date'] - pd.Timestamp("2022-01-10 15:00:00")).dt.total_seconds() / 3600

            # グラフ描画範囲でフィルタリング
            filtered_data = data[(data['date'] >= plot_start) & (data['date'] <= plot_end)]

            plt.figure(figsize=(18, 6))
            plt.plot(filtered_data['elapsed_time'], filtered_data[metrix], label="母集団との分布間ワッサースタイン距離", linewidth=2)

            # 落ち着いた色のハイライト窓を追加
            ax = plt.gca()  # 現在の軸を取得
            ylim = ax.get_ylim()  # Y軸の範囲を取得
            ax.add_patch(Rectangle(
                (0, ylim[0]),  # 開始位置 (経過時間=0, Y軸最小値)
                (highlight_end_1 - highlight_start_1).total_seconds() / 3600,  # 幅 (時間で計算)
                ylim[1] - ylim[0],  # 高さ
                color='#D3D3D3', alpha=0.5, label="初期モデルの学習範囲", edgecolor='#A9A9A9'
            ))

            # 別の落ち着いた色のハイライト窓を追加
            ax.add_patch(Rectangle(
                ((highlight_end_1 - highlight_start_1).total_seconds() / 3600, ylim[0]),  # 開始位置
                (highlight_end_2 - highlight_start_2).total_seconds() / 3600,  # 幅
                ylim[1] - ylim[0],  # 高さ
                color='#B0C4DE', alpha=0.5, label="評価範囲", edgecolor='#778899'
            ))

            # グラフのラベル設定
            plt.xlabel('Elapsed Time [h]', fontsize=label_size)
            plt.xticks(fontsize=ticks_size)
            plt.yticks(fontsize=ticks_size)

            plt.legend(loc='upper left', fontsize=legend_size)
            plt.grid(True)

            # 出力
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, format='png')
            plt.close()
