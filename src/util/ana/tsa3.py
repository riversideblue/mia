import csv
import os
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler

# --- データセットの各特徴量の分布がどのように遷移しているかを調査 Time Series Analysis --------------------------------------------------- #
dir_path = "/mnt/nas0/g005/murasemaru/data/csv/unproc/2201AusEast"
data_sec_size = 1  # データ区間の長さ(hours)
unit_time = 100  # 評価単位時間(hours)
output_dir = f"/mnt/nas0/g005/murasemaru/exp/1_DataAnalytics/drift/{os.path.basename(dir_path)}"  # グラフ保存先

# ---------------------------------------------------------------------------------------------------------------------------------- #

features = ["rcv_packet_count", "snd_packet_count", "tcp_count", "udp_count", "most_port", "port_count",
            "rcv_max_interval", "rcv_min_interval", "rcv_max_length", "rcv_min_length", "snd_max_interval",
            "snd_min_interval", "snd_max_length", "snd_min_length", "label"]

class Window:
    def __init__(self):
        self.cw = deque()
        self.cw_date_q = deque()  # datetime型を持つDeque
        self.cw_end_date = None

    def update(self, row, c_time, data_sec_size):
        self.cw.append(np.array(row, dtype=float))

        if not self.cw_date_q:
            self.cw_end_date = c_time - timedelta(hours=data_sec_size)
        else:
            delta = c_time - self.cw_date_q[-1]
            self.cw_end_date += delta
            while self.cw_date_q and self.cw_date_q[0] < self.cw_end_date:
                self.cw.popleft()
                self.cw_date_q.popleft()
        self.cw_date_q.append(c_time)

# --- データセット全体をロードし母集団の分布を準備 --- #
population_data = []
for d_file in sorted(os.listdir(dir_path)):
    d_file_path: str = f"{dir_path}/{d_file}"
    with open(d_file_path, mode='r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            population_data.append([float(row[headers.index(feature)]) for feature in features])

# numpy配列に変換
population_data = np.array(population_data).T  # 列ごとのデータ
print("Population data loaded and prepared.")
print(population_data)

# --- メイン処理 --- #
w = Window()
first_row_flag = True
start_date = None
next_eval_date = None
eval_res_col = ["daytime","w_rcv_pkt_ct", "ks_rcv_pkt_ct", "w_snd_pkt_ct", "ks_snd_pkt_ct", "w_tcp_ct", "ks_tcp_ct",
                    "w_udp_ct", "ks_udp_ct", "w_most_port", "ks_most_port", "w_port_ct", "ks_port_ct", 
                    "w_rcv_max_int", "ks_rcv_max_int", "w_rcv_min_int", "ks_rcv_min_int", "w_rcv_max_len", "ks_rcv_max_len", 
                    "w_rcv_min_len", "ks_rcv_min_len", "w_snd_max_int", "ks_snd_max_int", 
                    "w_snd_min_int", "ks_snd_min_int", "w_snd_max_len", "ks_snd_max_len", "w_snd_min_len", "ks_snd_min_len", 
                    "w_label", "ks_label", "w_mean_dis", "ks_mean_dist", "mean_dist"]

eval_res_li = np.empty((0, len(eval_res_col)), dtype=object)

for d_file in sorted(os.listdir(dir_path)):
    d_file_path: str = f"{dir_path}/{d_file}"
    print(f"- {d_file} set now")
    f = open(d_file_path, mode='r')
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        c_time = datetime.strptime(row[headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        if first_row_flag:
            start_date = c_time
            first_row_flag = False

        if c_time < start_date + timedelta(hours=unit_time):
            next_eval_date = c_time
            w.update(row[3:], c_time, data_sec_size)
            continue

        # --- 現在ウィンドウと母集団のデータ分布の違いを計算
        if c_time > next_eval_date:
            cw_arr = np.array(w.cw)
            if cw_arr.ndim == 1:
                cw_arr = cw_arr.reshape(1, -1)
            try:
                ex_cw = cw_arr.T
            except IndexError:
                ex_cw = np.zeros((4, cw_arr.shape[0]))

            # 距離計算
            w_dists = Parallel(n_jobs=-1)(
                delayed(wasserstein_distance)(c, p) for c, p in zip(ex_cw, population_data)
            )
            # 距離の不足分をゼロ埋め
            while len(w_dists) < len(features):  # "label"を除いた特徴量数
                w_dists.append(0)

            # KS距離計算
            ks_dists = Parallel(n_jobs=-1)(
                delayed(lambda c, p: ks_2samp(c, p).statistic)(c, p) for c, p in zip(ex_cw, population_data)
            )
            
            dists = []
            for i in range(len(features)):
                dists.append(w_dists[i])
                dists.append(ks_dists[i])

            # 空の場合にデフォルト値を設定
            w_mean_dist = np.mean(w_dists) if len(w_dists) > 0 else 0
            ks_mean_dist = np.mean(ks_dists) if len(ks_dists) > 0 else 0
            mean_dist = (w_mean_dist+ks_mean_dist)/2 if w_mean_dist!=0 and ks_mean_dist!=0 else 0

            eval_daytime = next_eval_date
            eval_arr = [eval_daytime] + dists + [w_mean_dist, ks_mean_dist, mean_dist]
            eval_res_li = np.vstack([eval_res_li, eval_arr])
            next_eval_date += timedelta(hours=unit_time)
            print(eval_arr)
        w.update(row[3:], c_time, data_sec_size)
    f.close()

# ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

add_res_col = ["nmr_w_mean_dist","nmr_ks_mean_dist","nmr_mean_dist"]
add_res_li = []

scaler = StandardScaler()

# Wのスケーリング
w_mean_dist_li = eval_res_li[:, -2]
reshaped_w = w_mean_dist_li.reshape(-1, 1)
scaled_w = scaler.fit_transform(reshaped_w).flatten() 
add_res_li.append(scaled_w)

# KSのスケーリング
ks_mean_dist_li = eval_res_li[:, -1]
reshaped_ks = ks_mean_dist_li.reshape(-1, 1)
scaled_ks = scaler.fit_transform(reshaped_ks).flatten() 
add_res_li.append(scaled_ks)

# NMRの計算
nmr_mean_dist_li = (scaled_w + scaled_ks) / 2  # 1次元配列
add_res_li.append(nmr_mean_dist_li)

# 2次元配列に変換
add_res_li = np.array(add_res_li).T  # 転置して列形式に変換

eval_res = pd.DataFrame(eval_res_li, columns=eval_res_col)
add_res = pd.DataFrame(add_res_li, columns=add_res_col)

# Combine evaluate_results with additional_results
eval_res = pd.concat([eval_res, add_res], axis=1)
eval_res.to_csv(os.path.join(output_dir, "drift_obs_with_population.csv"), index=False)
print(f"output: {output_dir}/drift_obs_with_population.csv")
