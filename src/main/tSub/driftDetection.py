import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import pingouin as pg
import faiss

from collections import deque
from datetime import datetime, timedelta
import csv
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity

class Window:
    def __init__(self):
        self.cw = deque()
        self.cw_date_q = deque()  # datetime型を持つDeque
        self.cw_end_date=None
        self.pw = deque()
        self.pw_date_q = deque()
        self.pw_end_date=None
        self.cum_statics: float = 0.0

    def update(self, row, c_time, cw_size, pw_size):
        self.cw.append(np.array(row, dtype=float))

        if not self.cw_date_q:
            self.cw_end_date = c_time-timedelta(seconds=cw_size)
            self.pw_end_date = self.cw_end_date-timedelta(seconds=pw_size)
        else:
            delta = c_time - self.cw_date_q[-1]
            self.cw_end_date+=delta
            self.pw_end_date+=delta
            while self.cw_date_q and self.cw_date_q[0] < self.cw_end_date:
                self.pw.append(self.cw.popleft())
                self.pw_date_q.append(self.cw_date_q.popleft())
            while self.pw_date_q and self.pw_date_q[0] < self.pw_end_date:
                self.pw.popleft()
                self.pw_date_q.popleft()
        self.cw_date_q.append(c_time)

    def ex_cw(self):
        cw_arr = np.array(self.cw)
        if cw_arr.ndim == 1:
            cw_arr = cw_arr.reshape(1, -1)
        try:
            ex_cw = cw_arr[:, [4, 6, 7, 9, 10]].T
        except IndexError:
            ex_cw = np.zeros((5, cw_arr.shape[0]))
        return ex_cw

    def ex_pw(self):
        pw_arr = np.array(self.pw)
        if pw_arr.ndim == 1:
            pw_arr = pw_arr.reshape(1, -1)
        try:
            ex_pw = pw_arr[:, [4, 6, 7, 9, 10]].T
        except IndexError:
            ex_pw = np.zeros((5, pw_arr.shape[0]))
        return ex_pw

    def ex_cw_v(self):
        cw_arr = np.array(self.cw)
        if cw_arr.ndim == 1:
            cw_arr = cw_arr.reshape(1, -1)
        try:
            ex_cw = cw_arr[:, 0:-1]
        except IndexError:
            ex_cw = np.zeros((cw_arr.shape[0], 14))
        return ex_cw

    def ex_pw_v(self):
        pw_arr = np.array(self.pw)
        if pw_arr.ndim == 1:
            pw_arr = pw_arr.reshape(1, -1)
        try:
            ex_pw = pw_arr[:, 0:-1]
        except IndexError:
            ex_pw = np.zeros((pw_arr.shape[0], 14))
        return ex_pw

def w_scaler(w: np.ndarray):
    
    # rcv_pkt_ct, snd_pkt_ct, tcp_ct, udp_ct, port_ct
    w[:,0:4] = w[:,0:4]/1000 #0から1000の間で最大最小値正規化
    w[:,0:4] = np.log1p(w[:,0:4]) #対数正規化
    w[:,5] = w[:,5]/1000 #0から1000の間で最大最小値正規化
    w[:,5] = np.log1p(w[:,5]) #対数正規化

    # most_port（ポート番号のスケーリング: ウェルノウン、登録済み、動的の範囲ごとに分類）
    w[:,4] = np.where((w[:,4] >= 0) & (w[:,4] <= 1023), 0,  # ウェルノウンポート
                    np.where((w[:,4] >= 1024) & (w[:,4] <= 49151), 0.5,  # 登録済みポート
                             1))  # 動的ポート
    
    # rcv_max_int, rcv_min_int, snd_max_int, snd_min_int
    w[:,[6,7,10,11]] = w[:,[6,7,10,11]]/60 # 0から60の間で正規化

    # rcv_max_len, rcv_min_len, snd_max_len, snd_min_len
    w[:,[8,9,12,13]] = w[:,[8,9,12,13]]/10000 # 0から10000の間で正規化

    return w

def call(method_code:int,c_window,p_window, threshold,k):
    method_dict = {
        0: cos_similarity,
        1: euc_distance
    }
    method = method_dict.get(method_code)
    if method is None:
        raise ValueError(f"Invalid method_code: {method_code}")
    return method(c_window, p_window, threshold, k)[1]

def call_obs(method_code:int, c_window: np.ndarray, p_window: np.ndarray, threshold: float, c_time, o_dir_path, k : int) -> bool:
    output_csv_path = f'{o_dir_path}/dd_res.csv'
    method_dict = {
        0: cos_similarity,
        1: euc_distance
    }
    method = method_dict.get(method_code)
    if method is None:
        raise ValueError(f"Invalid method_code: {method_code}")
    dd = method(c_window, p_window, threshold, k)
    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["date", "math", "res"])
        writer.writerow([c_time, dd[0], dd[1]])
    return False

def cos_similarity(c_window: np.ndarray, p_window: np.ndarray, threshold: float, k: int) -> bool:
    """
    コサイン類似度行列をFAISSを使用して計算し，行ごとの上位k個の平均類似度を求める．
    平均類似度が閾値以下であるかを判定する．

    Args:
        c_window (np.ndarray): 比較するウィンドウ（2次元配列）．
        p_window (np.ndarray): 比較対象のウィンドウ（2次元配列）．
        threshold (float): 閾値．
        k (int): 上位k個の類似度を考慮する．

    Returns:
        tuple: 平均類似度 (`mean_similarity`) と閾値を超えているかの判定結果 (`True`/`False`)．
    """
    c_window = w_scaler(c_window)
    p_window = w_scaler(p_window)

    # L2正規化
    c_window = c_window / np.linalg.norm(c_window, axis=1, keepdims=True)
    p_window = p_window / np.linalg.norm(p_window, axis=1, keepdims=True)

    d = p_window.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(p_window.astype(np.float32))
    similarities, indices = index.search(c_window.astype(np.float32), k)
    mean_similarity = np.mean(similarities)
    print(f"mean: {mean_similarity}")

    return mean_similarity, mean_similarity<threshold
    
def euc_distance(c_window: np.ndarray, p_window: np.ndarray, threshold: float, k: int):
    """
    ユークリッド距離行列を計算し，行ごとの上位N個の平均ユークリッド距離を求める．
    平均類似度が閾値以下であるかを判定する．

    Args:
        c_window (np.ndarray): 比較するウィンドウ（2次元配列）．
        p_window (np.ndarray): 比較対象のウィンドウ（2次元配列）．
        threshold (float): 閾値．
        k (int): 上位k個の類似度を考慮する．

    Returns:
        tuple: 平均距離 (`mean_distance`) と閾値を超えているかの判定結果 (`True`/`False`)．
    """
    c_window = w_scaler(c_window)
    p_window = w_scaler(p_window)
    
    d = p_window.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(p_window.astype(np.float32))
    distances, indices = index.search(c_window.astype(np.float32), k)
    mean_distance = np.mean(distances)
    print(f"mean: {mean_distance}")

    return mean_distance, mean_distance>threshold

def euc_distance_HNSW(c_window: np.ndarray, p_window: np.ndarray, threshold: float, k: int):
    """
    ユークリッド距離行列を計算し，行ごとの上位N個の平均ユークリッド距離を求める．
    平均類似度が閾値以下であるかを判定する．

    Args:
        c_window (np.ndarray): 比較するウィンドウ（2次元配列）．
        p_window (np.ndarray): 比較対象のウィンドウ（2次元配列）．
        threshold (float): 閾値．
        k (int): 上位k個の類似度を考慮する．

    Returns:
        tuple: 平均距離 (`mean_distance`) と閾値を超えているかの判定結果 (`True`/`False`)．
    """
    c_window = w_scaler(c_window)
    p_window = w_scaler(p_window)
    
    d = p_window.shape[1]
    index = faiss.IndexHNSWFlat(d)
    index.add(p_window.astype(np.float32))
    distances, indices = index.search(c_window.astype(np.float32), k)
    mean_distance = np.mean(distances)
    print(f"mean: {mean_distance}")

    return mean_distance, mean_distance>threshold


    