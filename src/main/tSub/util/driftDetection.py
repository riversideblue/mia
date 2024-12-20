import numpy as np
from collections import deque

from joblib import Parallel, delayed
from scipy.stats import ttest_ind, mannwhitneyu, wasserstein_distance, entropy, ks_2samp
import pingouin as pg


class Window:
    def __init__(self, cw_size, pw_size, row_len):
        self.c_window = deque(
            np.full((cw_size, row_len), np.nan, dtype=float),
            maxlen=cw_size
        )
        self.p_window = deque(
            np.full((pw_size, row_len), np.nan, dtype=float),
            maxlen=pw_size
        )
        self.cum_test_static = 0

    def update(self,row):
        self.p_window.append(self.c_window.popleft())
        self.c_window.append(np.array(row, dtype=float))

    def fnum_cw(self):
        rcv_current = np.array(self.c_window)[:, 3].astype(int)
        snd_current = np.array(self.c_window)[:, 4].astype(int)
        extracted_cw = rcv_current + snd_current
        return extracted_cw

    def fnum_pw(self):
        rcv_past = np.array(self.p_window)[:, 3].astype(int)
        snd_past = np.array(self.p_window)[:, 4].astype(int)
        extracted_pw = rcv_past + snd_past
        return extracted_pw

    def v2_cw(self):
        c_window_array = np.nan_to_num(np.array(self.c_window), nan=0)
        extracted_cw = c_window_array[:, [4, 6, 9, 10]].T
        return extracted_cw

    def v2_pw(self):
        p_window_array = np.nan_to_num(np.array(self.p_window), nan=0)
        extracted_pw = p_window_array[:, [4, 6, 9, 10]].T
        return extracted_pw

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
    return 1-mean_p_values

def welchs_t_test(c_window, p_window):
    stats, p_values = ttest_ind(c_window, p_window, axis=1, equal_var=False)
    mean_stats = np.nanmean(stats)
    mean_p_values = np.nanmean(p_values)
    if np.isnan(mean_stats):
        mean_stats = 0
    if np.isnan(mean_p_values):
        mean_p_values = 0
    print(mean_stats)
    print(mean_p_values)
    return 1-mean_p_values

def mann_whitney_u_test(c_window, p_window):
    stats, p_values = mannwhitneyu(c_window, p_window, axis=1, alternative='two-sided')
    mean_stats = np.nanmean(stats)
    mean_p_values = np.nanmean(p_values)
    if np.isnan(mean_stats):
        mean_stats = 0
    if np.isnan(mean_p_values):
        mean_p_values = 0
    print(mean_stats)
    print(mean_p_values)
    return 1-mean_p_values

def ks_test(c_window, p_window):
    stats, p_values = ks_2samp(c_window, p_window, axis=1)
    mean_stats = np.nanmean(stats)
    mean_p_values = np.nanmean(p_values)
    if np.isnan(mean_stats):
        mean_stats = 0
    if np.isnan(mean_p_values):
        mean_p_values = 0
    print(mean_stats)
    print(mean_p_values)
    return 1-mean_p_values

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
