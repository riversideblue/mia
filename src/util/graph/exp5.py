import os
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt


# staticとdynamicの精度と学習コストを比較
# 学習コストは学習回数×一つの学習に使用するデータの数で表現
# staticとdynamicの評価時間がすべて一致していることが条件
#-------------------------------------------------------------------#
metrix = "f1_score"
nt_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/terminated/nt"
st_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/terminated/st"
st_epochs = 30
dy_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/terminated/dy_th100_cw600_pw1200"
dy_epochs = 30
#-------------------------------------------------------------------#
common_path = os.path.commonpath([nt_dir_path, st_dir_path, dy_dir_path])
output_dir = f"{os.path.dirname(common_path)}/img_{os.path.basename(st_dir_path)}_{os.path.basename(dy_dir_path)}"
output_path = f"{output_dir}/STvsDY_{metrix}.png"

# データの読み込み
nt_eval_data = pd.read_csv(f"{nt_dir_path}/results_evaluate.csv")
st_eval_data = pd.read_csv(f"{st_dir_path}/results_evaluate.csv")
st_tr_data = pd.read_csv(f"{st_dir_path}/results_training.csv")
dy_eval_data = pd.read_csv(f"{dy_dir_path}/results_evaluate.csv")
dy_tr_data = pd.read_csv(f"{dy_dir_path}/results_training.csv")

# 列名の変更
nt_eval_data = nt_eval_data.rename(columns={metrix: "nt"})
st_eval_data = st_eval_data.rename(columns={metrix: "static"})
dy_eval_data = dy_eval_data.rename(columns={metrix: "dynamic"})

# ラベルサイズ
label_size = 24
ticks_size = 18
legend_size = 24

# Metrix比較
eval_data = pd.concat(
    [nt_eval_data.loc[:, ["daytime","nt"]], 
     st_eval_data.loc[:, "static", ], 
     dy_eval_data.loc[:, "dynamic"]],
    axis=1
)

eval_data['daytime'] = pd.to_datetime(eval_data['daytime'])

# 学習コスト比較
st_tr_data["daytime"] = pd.to_datetime(st_tr_data["daytime"])
dy_tr_data["daytime"] = pd.to_datetime(dy_tr_data["daytime"])

# 学習コストを計算する関数を定義
def calculate_training_cost(tr_data, eval_times, epochs, window_minutes=30):
    costs = []
    for eval_time in eval_times:
        start_time = eval_time - pd.Timedelta(minutes=window_minutes)
        end_time = eval_time + pd.Timedelta(minutes=window_minutes)
        window_data = tr_data[(tr_data["daytime"] >= start_time) & (tr_data["daytime"] < end_time)]
        tr_cost = window_data["flow_num"]*epochs
        costs.append(tr_cost.sum())
    return pd.Series(costs, index=eval_times)

# 評価時刻のリスト
eval_times = eval_data['daytime']

# 各データセットの学習コストを計算
st_tr_cost = calculate_training_cost(st_tr_data, eval_times,st_epochs)
dy_tr_cost = calculate_training_cost(dy_tr_data, eval_times,dy_epochs)
print(st_tr_cost)
print(dy_tr_cost)

# st_tr_list, dy_tr_list のインデックスを datetime64 に変換（安全のため再確認）
st_tr_cost.index = pd.to_datetime(st_tr_cost.index)
dy_tr_cost.index = pd.to_datetime(dy_tr_cost.index)

# eval_data['daytime'] を datetime64 に変換（再確認）
eval_data['daytime'] = pd.to_datetime(eval_data['daytime'])

# インデックスを用いた結合
eval_data = eval_data.set_index('daytime')  # 'daytime' をインデックスに設定
eval_data['st_tr_cost'] = st_tr_cost  # st_tr_list を結合
eval_data['dy_tr_cost'] = dy_tr_cost  # dy_tr_list を結合

# 結合後にインデックスをリセット（必要なら）
eval_data = eval_data.reset_index()
print(eval_data)

# 精度と学習コストを一枚のグラフにまとめて可視化
fig, ax1 = plt.subplots(figsize=(14, 10))

# 左側の軸に精度をプロット
line1, = ax1.plot(
    eval_data['daytime'],
    eval_data["nt"], 
    label="再学習なし", linewidth=2, color="#7F7F7F"
)
line2, = ax1.plot(
    eval_data['daytime'],
    eval_data["static"], 
    label="従来手法", linewidth=2, color="blue"
)
line3, = ax1.plot(
    eval_data['daytime'], 
    eval_data["dynamic"], 
    label="提案手法", linewidth=2, color='red'
)
ax1.set_ylabel(f'{metrix}', fontsize=label_size, color='black')
ax1.tick_params(axis='y', labelsize=ticks_size)
ax1.tick_params(axis='x', labelsize=ticks_size, rotation=45)
ax1.grid(True)

# 右側の軸に学習コストを追加
ax2 = ax1.twinx()
line4, = ax2.plot(
    eval_data['daytime'],
    eval_data['st_tr_cost'],
    label='従来手法による学習コスト', linewidth=2, linestyle='--', color='#1F77B4', marker='o'
)
line5, = ax2.plot(
    eval_data['daytime'],
    eval_data['dy_tr_cost'],
    label='提案手法による学習コスト', linewidth=2, linestyle='--', color='#FF7F0E', marker='s'
)
ax2.set_ylabel('Training Cost', fontsize=label_size, color='black')
ax2.tick_params(axis='y', labelsize=ticks_size)

# 両方の軸のラインを統一して1つの凡例にまとめる
lines = [line1, line2, line3, line4, line5]
labels = [line.get_label() for line in lines]
fig.legend(
    lines, labels, loc='center left', fontsize=legend_size, bbox_to_anchor=(0.08, 0.65), ncol=1
)

# レイアウト調整と保存
os.makedirs(output_dir, exist_ok=True)
fig.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()
