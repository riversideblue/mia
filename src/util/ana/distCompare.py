import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy, ks_2samp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def process_directory(dir_path):
    return calculate_distributions(dir_path)

def process_comparison(args):
    dist_x, dist_y = args
    return compare_distributions(dist_x, dist_y)

def calculate_distributions(dir_path):
    """指定されたディレクトリ内のCSVファイルの特徴量分布を計算する"""
    distributions = {}
    for file_name in sorted(os.listdir(dir_path)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dir_path, file_name)
            data = pd.read_csv(file_path)
            
            # 最初の3列をスキップ
            columns_to_process = data.columns[3:-1]  # 最初の3列以外を選択
            
            for column in columns_to_process:
                if np.issubdtype(data[column].dtype, np.number):  # 数値型の列をチェック
                    if column not in distributions:
                        distributions[column] = []
                    distributions[column].append(data[column].dropna().values)
    return distributions

from scipy.stats import ks_2samp

def compare_distributions(dist1, dist2):
    """二つの特徴量分布を比較してWasserstein距離、KLダイバージェンス、KS統計量を計算する"""
    comparison_results = {}
    all_columns = set(dist1.keys()).union(set(dist2.keys()))

    for column in all_columns:
        comparison_results[column] = {
            'wasserstein': None,
            'kl_divergence': None,
            'ks_statistic': None
        }
        if column in dist1 and column in dist2:
            data1 = np.concatenate(dist1[column])
            data2 = np.concatenate(dist2[column])

            # Wasserstein距離
            wasserstein = wasserstein_distance(data1, data2)

            # KLダイバージェンス
            hist1, bin_edges = np.histogram(data1, bins=50, density=True)
            hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
            hist1 += 1e-10  # 0除算回避のために小さい値を足す
            hist2 += 1e-10
            kl_div = entropy(hist1, hist2)

            # KS統計量
            ks_stat, _ = ks_2samp(data1, data2)

            comparison_results[column]['wasserstein'] = wasserstein
            comparison_results[column]['kl_divergence'] = kl_div
            comparison_results[column]['ks_statistic'] = ks_stat

    return comparison_results


if __name__ == "__main__":
    all_dir_path = input("分布を解析したいすべてのディレクトリが格納されたディレクトリパスを入力してください: ")
    output_dir = f"/mnt/nas0/g005/murasemaru/exp/other/{os.path.basename(all_dir_path)}"
    
    # 全ディレクトリの特徴量分布を計算（並列化）
    dir_paths = [os.path.join(all_dir_path, dir_name) for dir_name in sorted(os.listdir(all_dir_path)) if os.path.isdir(os.path.join(all_dir_path, dir_name))]
    
    with ProcessPoolExecutor() as executor:
        all_dir_dist = list(executor.map(process_directory, dir_paths))
    
    print("特徴量分布計算完了")

    # サブディレクトリ名を取得
    subdirs = [os.path.basename(path) for path in dir_paths]
    
    # 特徴量のリスト
    features = [
        "ex_address", "in_address", "daytime", "rcv_packet_count", "snd_packet_count", "tcp_count",
        "udp_count", "most_port", "port_count", "rcv_max_interval", "rcv_min_interval",
        "rcv_max_length", "rcv_min_length", "snd_max_interval", "snd_min_interval",
        "snd_max_length", "snd_min_length"
    ]

    w_dfs = []
    kl_dfs = []

    # 結果を保存するための追加
    ks_dfs = []

    for feature in features:
        w_df = pd.DataFrame(index=subdirs, columns=subdirs)
        kl_df = pd.DataFrame(index=subdirs, columns=subdirs)
        ks_df = pd.DataFrame(index=subdirs, columns=subdirs)

        # 比較処理を並列化
        with ProcessPoolExecutor() as executor:
            args = [(all_dir_dist[i], all_dir_dist[j]) for i in range(len(all_dir_dist)) for j in range(len(all_dir_dist))]
            results = list(executor.map(process_comparison, args))

        # 結果をデータフレームに格納
        for idx, (i, j) in enumerate([(i, j) for i in range(len(all_dir_dist)) for j in range(len(all_dir_dist))]):
            metrics = results[idx]
            for column, metric_values in metrics.items():
                if metric_values['wasserstein'] is not None:
                    w_df.iloc[i, j] = metric_values['wasserstein']
                if metric_values['kl_divergence'] is not None:
                    kl_df.iloc[i, j] = metric_values['kl_divergence']
                if metric_values['ks_statistic'] is not None:
                    ks_df.iloc[i, j] = metric_values['ks_statistic']

        # 結果を保存
        w_dfs.append(w_df)
        kl_dfs.append(kl_df)
        ks_dfs.append(ks_df)
        os.makedirs(f"{output_dir}", exist_ok=True)
        w_df.to_csv(f"{output_dir}/{feature}_w.csv")
        kl_df.to_csv(f"{output_dir}/{feature}_kl.csv")
        ks_df.to_csv(f"{output_dir}/{feature}_ks.csv")
        print(f"csvファイルを保存しました: {feature}")

    # 平均化して保存
    w_concatenated = pd.concat(w_dfs, keys=features)
    w_averaged_df = w_concatenated.groupby(level=1).mean()
    w_averaged_df.to_csv(f"{output_dir}/averaged_w.csv")

    kl_concatenated = pd.concat(kl_dfs, keys=features)
    kl_averaged_df = kl_concatenated.groupby(level=1).mean()
    kl_averaged_df.to_csv(f"{output_dir}/averaged_kl.csv")

    ks_concatenated = pd.concat(ks_dfs, keys=features)
    ks_averaged_df = ks_concatenated.groupby(level=1).mean()
    ks_averaged_df.to_csv(f"{output_dir}/averaged_ks.csv")

    print("全ての処理が完了しました．")