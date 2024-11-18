import os
from datetime import timedelta, datetime

import pandas as pd


def  main():
    # csvファイルのtimestampを基準値に沿って平行移動する
    beginning_timestamp = "2018-05-21 16:01:07"
    target_csv_dir_path = "src/main/outputs/extracted/20241117174650"
    first_reading_flag = True
    shift = None
    print("time override")

    for dataset_file in os.listdir(target_csv_dir_path):
        print(f"{dataset_file} : processing")
        dataset_file_path = os.path.join(target_csv_dir_path,dataset_file)
        df = pd.read_csv(dataset_file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if first_reading_flag:
            shift = df["timestamp"].iloc[0] - datetime.strptime(beginning_timestamp, "%Y-%m-%d %H:%M:%S")
            first_reading_flag = False

        # shift分前に進める
        df['timestamp'] = df['timestamp'] - shift
        # 上書き保存
        df.to_csv(dataset_file_path, index=False)

if __name__ == "__main__":
    main()