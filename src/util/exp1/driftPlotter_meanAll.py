import os

import pandas as pd
import matplotlib.pyplot as plt

# --- 指定した親ディレクトリに含まれるサブディレクトリのmean_disをサブディレクトリごとにグラフにまとめて出力
# -------------------------------------------------------------------------------#
all_dir_path = "/mnt/nas0/g005/murasemaru/exp/1_DataAnalytics/drift/filtered_22020110-22020114/2201HP"
metrix = "mean_dis"
label_size = 22
ticks_size = 16
legend_size = 22
file_name = f"{metrix}_8H"

# 出力ディレクトリを設定
output_dir_path = os.path.join(all_dir_path, "res_img")
os.makedirs(output_dir_path, exist_ok=True)

# 全てのグラフを1つの図にまとめる
plt.figure(figsize=(12, 8))
plt.title(f"{os.path.basename(all_dir_path)}:{metrix}")

# 各ディレクトリのデータを同じプロットに追加
for di in os.listdir(all_dir_path):
    di_path = f"{all_dir_path}/{di}"
    fi = os.path.join(di_path, "tsa_8H.csv")
    fi_path = os.path.join(di_path, fi)

    if os.path.exists(fi_path):  # eval_res.csv が存在する場合のみ処理
        print(f"Processing file: {fi_path}")
        
        data = pd.read_csv(fi_path)
        data['date'] = pd.to_datetime(data['date'])

        # 各ディレクトリごとに異なる色でプロット
        plt.plot(data['date'], data[metrix], label=f"{di}", linewidth=2)

# グラフの装飾
plt.xlabel('date', fontsize=label_size)
plt.ylabel(f"{metrix}", fontsize=label_size)
plt.xticks(fontsize=ticks_size, rotation=45)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size)
plt.grid(True)

# 出力
plt.tight_layout()
output_path = f"{output_dir_path}/{file_name}.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Combined graph saved at: {output_path}")

