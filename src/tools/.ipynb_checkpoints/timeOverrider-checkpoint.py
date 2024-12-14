import os
from datetime import timedelta, datetime

import pandas as pd

# --- csvファイルのtimestampを基準値に沿って平行移動するスクリプト --------------------------------------------------------------------- #
beginning_timestamp = "2022-01-01 09:02:39"
target_dir_path = "src/main/traffic_data/csv/unprocessed/AusEast202201"
# --- Create output directory
dt = datetime.strptime(beginning_timestamp, "%Y-%m-%d %H:%M:%S")
formatted_date = dt.strftime("%Y%m%d")
output_dir_path: str = f"src/main/traffic_data/csv/{os.path.basename(target_dir_path)}-fetch{formatted_date}"
os.makedirs(output_dir_path)
# ---------------------------------------------------------------------------------------------------------------------------------- #

first_reading_flag = True
shift = None
print("time override")
for dataset_file in os.listdir(target_dir_path):
    print(f"{dataset_file} : processing")
    dataset_file_path = os.path.join(target_dir_path,dataset_file)
    df = pd.read_csv(dataset_file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if first_reading_flag:
        shift = df["timestamp"].iloc[0] - datetime.strptime(beginning_timestamp, "%Y-%m-%d %H:%M:%S")
        first_reading_flag = False

    # shift分前に進める
    df['timestamp'] = df['timestamp'] - shift
    # 上書き保存

    output_file_name = df[0]["timestamp"].strftime("%Y%m%d")
    df.to_csv(f"{output_dir_path}/", index=False)