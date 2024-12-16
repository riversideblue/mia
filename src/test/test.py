import numpy as np
from scipy.stats import wasserstein_distance, entropy, mannwhitneyu


def mann_whitney_u_test(c_window, p_window, cum_test_static, alpha=0.05):

    u_stat, p_value = mannwhitneyu(c_window, p_window, alternative='two-sided')
    return p_value < alpha


def wasserstein_test_high_dim(c_window, p_window, threshold, alpha=0.05):
    """
    Wasserstein距離を使用した高次元データの2標本検定

    Parameters:
        c_window (np.ndarray): 現在のウィンドウデータ (高次元、shape = (n_samples, n_features)).
        p_window (np.ndarray): 過去のウィンドウデータ (高次元、shape = (n_samples, n_features)).
        threshold (float): Wasserstein距離の閾値.
        alpha (float): 有意水準.

    Returns:
        bool: True if the distributions are significantly different.
    """
    distances = []
    for i in range(c_window.shape[1]):  # 各特徴量についてWasserstein距離を計算
        distance = wasserstein_distance(c_window[:, i], p_window[:, i])
        distances.append(distance)
    mean_distance = np.mean(distances)  # 距離の平均を使用
    print(f"平均Wasserstein距離: {mean_distance}")
    return mean_distance > threshold

def kl_divergence_test_high_dim(c_window, p_window, threshold, alpha=0.05):
    """
    KLダイバージェンスを使用した高次元データの2標本検定

    Parameters:
        c_window (np.ndarray): 現在のウィンドウデータ (高次元、shape = (n_samples, n_features)).
        p_window (np.ndarray): 過去のウィンドウデータ (高次元、shape = (n_samples, n_features)).
        threshold (float): KLダイバージェンスの閾値.
        alpha (float): 有意水準.

    Returns:
        bool: True if the distributions are significantly different.
    """
    kl_values = []
    for i in range(c_window.shape[1]):  # 各特徴量についてKLダイバージェンスを計算
        hist_c, _ = np.histogram(c_window[:, i], bins=50, density=True)
        hist_p, _ = np.histogram(p_window[:, i], bins=50, density=True)
        kl_div = entropy(hist_c + 1e-10, hist_p + 1e-10)  # 1e-10でゼロ割を防止
        kl_values.append(kl_div)
    mean_kl = np.mean(kl_values)  # KLダイバージェンスの平均を使用
    print(f"平均KLダイバージェンス: {mean_kl}")
    return mean_kl > threshold

if __name__ == "__main__":
    # サンプル高次元データ生成
    current_window = np.random.normal(0, 1, (1000, 5))  # 1000サンプル、5次元
    past_window = np.random.normal(1, 1, (1000, 5))    # 1000サンプル、5次元

    # 閾値設定
    wasserstein_threshold = 0.2
    kl_threshold = 0.5

    # Wasserstein距離を使用した検定
    wasserstein_result = wasserstein_test_high_dim(current_window, past_window, wasserstein_threshold)
    print(f"Wasserstein Test Result: {'Different' if wasserstein_result else 'Similar'}")

    # KLダイバージェンスを使用した検定
    kl_result = kl_divergence_test_high_dim(current_window, past_window, kl_threshold)
    print(f"KL Divergence Test Result: {'Different' if kl_result else 'Similar'}")
