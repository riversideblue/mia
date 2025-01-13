import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, mannwhitneyu, wasserstein_distance, entropy, ks_2samp
import pingouin as pg

from collections import deque
from datetime import datetime, timedelta

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

def call(method_code:int,c_window,p_window):

    method_dict = {
        0: independent_t_test,
        1: welchs_t_test,
        2: mann_whitney_u_test,
        3: ks_test,
        4: wasserstein_test,
        5: kl_divergence_test,
        6: hoteling_t2_test_with_library,
        7: bootstrap_test,
        8: permutation_test
    }

    method = method_dict.get(method_code)
    if method is None:
        raise ValueError(f"Invalid method_code: {method_code}")
    return method(c_window, p_window)

def call_v(c_window, p_window, threshold):
    
    similarity_matrix = cosine_similarity(c_window, p_window)
    max_similarity_list = np.max(similarity_matrix, axis=1)

    # 平均と最小値を計算
    mean_similarity = np.mean(max_similarity_list)
    # min_similarity = np.min(max_similarity_list)

    print(f'mean:{mean_similarity}')
    # print(f'min:{min_similarity}')

    return mean_similarity < threshold

def independent_t_test(c_window, p_window):
    stats, p_values = ttest_ind(c_window, p_window, axis=1, equal_var=True)
    mean_stats = np.nanmean(stats)
    mean_p_values = np.nanmean(p_values)
    if np.isnan(mean_stats):
        mean_stats = 0
    if np.isnan(mean_p_values):
        mean_p_values = 0
    print(mean_stats)
    print(mean_p_values)
    return mean_p_values

def welchs_t_test(cw, pw):
    stats, p_values = ttest_ind(cw, pw, axis=1, equal_var=False)
    mean_stats = np.mean(stats)
    if np.isnan(np.mean(stats)):
        return 0.0
    return mean_stats

def mann_whitney_u_test(c_window, p_window):
    stats, p_values = mannwhitneyu(c_window, p_window, axis=1, alternative='two-sided')
    mean_stats = np.nanmean(stats)
    if np.isnan(np.mean(stats)):
        return 0.0
    return mean_stats

def ks_test(c_window, p_window):
    stats, p_values = ks_2samp(c_window, p_window, axis=1)
    if np.isnan(np.mean(stats)):
        return 0.0
    return mean_stats

def wasserstein_test(c_window, p_window):
    distances = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance)(c, p) for c, p in zip(c_window, p_window)
    )
    mean_distance = np.mean(distances)  # 距離の平均を使用
    print(f"mean distance: {mean_distance}")
    return mean_distance

def kl_divergence_test(c_window, p_window, cum_test_static, alpha=0.05):
    kl_values = []
    for i in range(c_window.shape[1]):  # 各特徴量についてKLダイバージェンスを計算
        hist_c, _ = np.histogram(c_window[:, i], bins=50, density=True)
        hist_p, _ = np.histogram(p_window[:, i], bins=50, density=True)
        kl_div = entropy(hist_c + 1e-10, hist_p + 1e-10)  # 1e-10でゼロ割を防止
        kl_values.append(kl_div)
    mean_kl = np.mean(kl_values)  # KLダイバージェンスの平均を使用
    cum_test_static += mean_kl
    return cum_test_static > alpha

def hoteling_t2_test_with_library(c_window, p_window):
    # Hotelling's T-squared検定の実行
    t2_result = pg.multivariate_ttest(c_window, p_window)

    # 正しい列名を使用して値を取得
    t2_stat = t2_result.loc['hotelling', 'T2']  # 'T2'列からT-squared統計量を取得
    p_value = t2_result.loc['hotelling', 'pval']  # 'pval'列からp値を取得

    print(f"Hotelling's T-squared statistic: {t2_stat}")
    print(f"p-value: {p_value}")

    return p_value

def bootstrap_test(c_window, p_window, n_resamples=10000):
    def single_bootstrap_test(c, p):
        combined = np.concatenate([c, p])
        observed_stat = np.mean(c) - np.mean(p)
        bootstrap_stats = []
        for _ in range(n_resamples):
            resample_c = np.random.choice(combined, size=len(c), replace=True)
            resample_p = np.random.choice(combined, size=len(p), replace=True)
            bootstrap_stats.append(np.mean(resample_c) - np.mean(resample_p))
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        return p_value

    p_values = Parallel(n_jobs=-1)(
        delayed(single_bootstrap_test)(c, p) for c, p in zip(c_window, p_window)
    )
    print(f"Bootstrap test p-values: {p_values}")
    return 1-np.mean(p_values)

def permutation_test(c_window, p_window, n_permutations=1000):
    def single_permutation_test(c, p):
        combined = np.concatenate([c, p])
        print(combined)
        observed_stat = np.mean(c) - np.mean(p)
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_c = combined[:len(c)]
            perm_p = combined[len(c):]
            perm_stats.append(np.mean(perm_c) - np.mean(perm_p))
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        p_value-0.5
        return p_value

    p_values = Parallel(n_jobs=-1)(
        delayed(single_permutation_test)(c, p) for c, p in zip(c_window, p_window)
    )
    print(f"Permutation test p-values: {p_values}")
    return 1-np.mean(p_values)
