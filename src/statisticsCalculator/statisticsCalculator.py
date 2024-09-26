import numpy as np
np.set_printoptions(suppress=True)

import glob



txt_dir = "txt"

for file_path in glob.glob(txt_dir + "/" + "*.txt"):
    # ファイルを開いて内容を読み込む
    with open(file_path, 'r') as file:
        content = file.readlines()

    # 数値のみを抽出してリストに読み込む
    filtered_values = []
    for line in content:
        try:
            # 数値に変換できる場合のみリストに追加
            value = float(line.strip())
            filtered_values.append(value)
        except ValueError:
            # 数値に変換できない行は無視
            continue

    # 最小値、最大値、平均値、分散を計算
    min_value = np.min(filtered_values)
    max_value = np.max(filtered_values)
    mean_value = np.mean(filtered_values)
    variance = np.var(filtered_values)

    if "/" in file_path:
        fn = file_path.split("/")[1]
    elif "//" in file_path:
        fn = file_path.split("//")[1]
    else:
        fn = file_path

    # 結果の表示
    print(fn)
    print("-------------------------")
    print('{:.8f}'.format(min_value))
    print('{:.8f}'.format(max_value))
    print('{:.8f}'.format(mean_value))
    print('{:.8f}'.format(variance))
    print("-------------------------")
    print("")