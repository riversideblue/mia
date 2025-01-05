import os
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt

#-------------------------------------------------------------------#
# 定数定義
metrix = "accuracy"
nt_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/1/nt"
st_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/1/st"
st_epochs = 30
dy_dir_path = "/mnt/nas0/g005/murasemaru/exp/5_Eval/2201AusEast/1/dy_th100_cw600_pw1200"
dy_epochs = 30
#-------------------------------------------------------------------#

# 出力先ディレクトリの設定
output_dir = f"{dy_dir_path}/res_img"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/STvsDY_{metrix}.png"

# データの読み込みと前処理
nt_eval_data = pd.read_csv(f"{nt_dir_path}/results_evaluate.csv").rename(columns={metrix: "nt"})
st_eval_data = pd.read_csv(f"{st_dir_path}/results_evaluate.csv").rename(columns={metrix: "static"})
dy_eval_data = pd.read_csv(f"{dy_dir_path}/results_evaluate.csv").rename(columns={metrix: "dynamic"})

eval_data = pd.concat(
    [nt_eval_data.loc[:, ["daytime", "nt"]],
     st_eval_data.loc[:, ["static"]],
     dy_eval_data.loc[:, ["dynamic"]]],
    axis=1
)
eval_data['daytime'] = pd.to_datetime(eval_data['daytime'])

st_tr_data = pd.read_csv(f"{st_dir_path}/results_training.csv")
st_tr_data["daytime"] = pd.to_datetime(st_tr_data["daytime"])
dy_tr_data = pd.read_csv(f"{dy_dir_path}/results_training.csv")
dy_tr_data["daytime"] = pd.to_datetime(dy_tr_data["daytime"])

# 学習コストを計算する関数を定義
def calculate_training_cost(tr_data, eval_times, epochs, window_minutes=30):
    costs = []
    for eval_time in eval_times:
        start_time = eval_time - pd.Timedelta(minutes=window_minutes)
        end_time = eval_time + pd.Timedelta(minutes=window_minutes)
        window_data = tr_data[(tr_data["daytime"] >= start_time) & (tr_data["daytime"] < end_time)]
        if window_data.empty:
            costs.append(0)
        else:
            tr_cost = window_data["flow_num"] * epochs
            costs.append(tr_cost.sum())
    return pd.Series(costs, index=eval_times)

# 評価時刻のリスト
eval_times = eval_data['daytime']

# 各データセットの学習コストを計算
st_tr_cost = calculate_training_cost(st_tr_data, eval_times, st_epochs)
dy_tr_cost = calculate_training_cost(dy_tr_data, eval_times, dy_epochs)

# データの結合
eval_data = eval_data.set_index('daytime')
eval_data['st_tr_cost'] = st_tr_cost
eval_data['dy_tr_cost'] = dy_tr_cost
eval_data = eval_data.reset_index()

# 精度と学習コストを可視化する関数
def plot_accuracy_and_cost(eval_data, output_path, label_size, ticks_size, legend_size):
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # 左側の軸に精度をプロット
    line1, = ax1.plot(eval_data['daytime'], eval_data["nt"], label="再学習なし", linewidth=2, color="#7F7F7F")
    line2, = ax1.plot(eval_data['daytime'], eval_data["static"], label="従来手法", linewidth=2, color="#377eb8")
    line3, = ax1.plot(eval_data['daytime'], eval_data["dynamic"], label="提案手法", linewidth=2, color='#e41a1c')
    ax1.set_ylabel(f'Accuracy', fontsize=label_size, color='black')
    ax1.tick_params(axis='y', labelsize=ticks_size)
    ax1.tick_params(axis='x', labelsize=ticks_size, rotation=45)
    ax1.grid(True)

    # 右側の軸に学習コストをプロット
    ax2 = ax1.twinx()
    line4, = ax2.plot(eval_data['daytime'], eval_data['st_tr_cost'], label='従来手法による学習コスト',
                      linewidth=2, linestyle='--', color='#984ea3', marker='o')
    line5, = ax2.plot(eval_data['daytime'], eval_data['dy_tr_cost'], label='提案手法による学習コスト',
                      linewidth=2, linestyle='--', color='#ff7f00', marker='s')
    ax2.set_ylabel(r'Training cost [$10^{7}$]', fontsize=label_size, color='black')
    ax2.tick_params(axis='y', labelsize=ticks_size)

    # 凡例
    lines = [line1, line2, line3, line4, line5]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='center left', fontsize=legend_size, bbox_to_anchor=(0.08, 0.40), ncol=1)

    # 保存と表示
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# プロット実行
plot_accuracy_and_cost(eval_data, output_path, label_size=24, ticks_size=18, legend_size=24)
print(output_path)
