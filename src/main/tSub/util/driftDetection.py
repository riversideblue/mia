import numpy as np
from collections import deque

from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wasserstein_distance, entropy
import pingouin as pg


class Window:
    def __init__(self, cw_size, pw_size, threshold, row_len):
        self.c_window = deque(
            np.full((cw_size, row_len), -threshold, dtype=object),
            maxlen=cw_size
        )
        self.p_window = deque(
            np.full((pw_size, row_len), -threshold, dtype=object),
            maxlen=pw_size
        )
        self.cum_test_static = 0

    def update(self,row):
        self.p_window.append(self.c_window.popleft())
        self.c_window.append(np.array(row, dtype=object))

    def fnum_cw(self):
        rcv_current = np.array(self.c_window)[:, 3].astype(int)
        snd_current = np.array(self.c_window)[:, 4].astype(int)
        extracted_current = rcv_current + snd_current
        return extracted_current

    def fnum_pw(self):
        rcv_past = np.array(self.p_window)[:, 3].astype(int)
        snd_past = np.array(self.p_window)[:, 4].astype(int)
        extracted_past = rcv_past + snd_past
        return extracted_past

def call(method_code:int,c_window,p_window,cum_test_static,threshold):

    method_dict = {
        0: independent_t_test,
        1: paired_t_test,
        2: welchs_t_test,
        3: mann_whitney_u_test,
        4: wasserstein_test,
        5: kl_divergence_test,
        6: hoteling_t2_test_with_library
    }

    method = method_dict.get(method_code)
    if method is None:
        raise ValueError(f"Invalid method_code: {method_code}")
    return method(c_window, p_window,cum_test_static,threshold)

def independent_t_test(c_window, p_window, cum_test_static, alpha=2.0):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowの分散は等しい×
        c_windowとp_windowは独立である×
    """
    t_stat, p_value = ttest_ind(c_window, p_window, equal_var=True)
    cum_test_static += (1-p_value)
    return cum_test_static > alpha

def paired_t_test(c_window, p_window, cum_test_static, alpha=0.05):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowは同一の対象である(独立でない)〇
        c_windowとp_windowは同じ長さである×
    """
    t_stat, p_value = ttest_rel(c_window, p_window)
    return p_value < alpha

def welchs_t_test(c_window, p_window, cum_test_static, alpha=0.3):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowは独立である×
    """
    t_stat, p_value = ttest_ind(c_window, p_window, equal_var=False)
    return p_value < alpha

def mann_whitney_u_test(c_window, p_window, cum_test_static, alpha=0.05):

    u_stat, p_value = mannwhitneyu(c_window, p_window, alternative='two-sided')
    cum_test_static += p_value
    return cum_test_static < alpha

def wasserstein_test(c_window, p_window, cum_test_static, alpha=0.05):
    distances = []
    for i in range(c_window.shape[1]):  # 各特徴量についてWasserstein距離を計算
        distance = wasserstein_distance(c_window[:, i], p_window[:, i])
        distances.append(distance)
    mean_distance = np.mean(distances)  # 距離の平均を使用
    print(f"平均Wasserstein距離: {mean_distance}")
    cum_test_static += mean_distance
    return cum_test_static > alpha

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

def hoteling_t2_test_with_library(c_window, p_window, alpha=0.05):
    # Pingouinを使用してHotelling's T-squared検定を実行
    t2_result = pg.multivariate_ttest(c_window, p_window)

    # 結果を取得
    t2_stat = t2_result.loc['T2', 'stat']
    p_value = t2_result.loc['T2', 'pval']

    print(f"Hotelling's T-squared statistic: {t2_stat}")
    print(f"p-value: {p_value}")

    return p_value < alpha, t2_result