import numpy as np
from collections import deque

from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu


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
        self.threshold = threshold

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

def call(method_code:int,c_window,p_window):

    method_dict = {
        0: independent_t_test,
        1: paired_t_test,
        2: welchs_t_test,
        3: mann_whitney_u_test
    }

    method = method_dict.get(method_code)
    if method is None:
        raise ValueError(f"Invalid method_code: {method_code}")
    return method(c_window, p_window)

def independent_t_test(c_window, p_window, alpha=0.05):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowの分散は等しい×
        c_windowとp_windowは独立である×
    """
    t_stat, p_value = ttest_ind(c_window, p_window, equal_var=True)
    return p_value < alpha

def paired_t_test(c_window, p_window, alpha=0.05):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowは同一の対象である(独立でない)〇
        c_windowとp_windowは同じ長さである×
    """
    t_stat, p_value = ttest_rel(c_window, p_window)
    return p_value < alpha

def welchs_t_test(c_window, p_window, alpha=0.3):
    """
        前提条件：
        c_windowとp_windowは正規分布に従う×
        c_windowとp_windowは独立である×
    """
    t_stat, p_value = ttest_ind(c_window, p_window, equal_var=False)
    return p_value < alpha

def mann_whitney_u_test(c_window, p_window, alpha=0.05):
    """
    Mann-Whitney U検定

    Parameters:
        c_window (list or np.ndarray): 現在のウィンドウデータ.
        p_window (list or np.ndarray): 過去のウィンドウデータ.
        alpha (float): 有意水準.

    Returns:
        bool: 帰無仮説を棄却するか（True: 棄却, False: 棄却しない）.
    """
    u_stat, p_value = mannwhitneyu(c_window, p_window, alternative='two-sided')
    return p_value < alpha