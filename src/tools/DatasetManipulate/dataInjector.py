import os
import pandas as pd
import csv
import shutil
from datetime import datetime, timedelta
from itertools import cycle
import time

# --- 設定 ----------------------------------------------------------------------- #
target_dir_path: str = "src/main/traffic_data/csv/unprocessed/2201Lab01"
source_dir_path: str = "src/main/traffic_data/csv/unprocessed/2201AusEast"
output_dir_path: str = f"src/main/traffic_data/csv/modified/{os.path.basename(target_dir_path)}+0"
# -------------------------------------------------------------------------------- #
t_csv_files = [f for f in sorted(os.listdir(target_dir_path)) if f.endswith('.csv')]
s_csv_files = [f for f in sorted(os.listdir(source_dir_path)) if f.endswith('.csv')]

# source_dir_pathから一時間分のデータを抽出 -> first_hour_data
s_row_list = []
first_time = None
for s_csv_file in s_csv_files:
    s_file_path: str = os.path.join(source_dir_path, s_csv_file)
    with open(s_file_path, mode="r") as s_row_iter:
        s_reader = csv.reader(s_row_iter)
        s_header = next(s_reader)
        s_ts_index = s_header.index("daytime")
        while True:
            try:
                s_row = next(s_reader)
                s_row_ts = datetime.strptime(s_row[s_ts_index], "%Y-%m-%d %H:%M:%S")
                if first_time is None:
                    first_time = s_row_ts
                if not s_row_ts < first_time + timedelta(hours=1):
                    break
                else:
                    s_row_list.append(s_row)
            except StopIteration:
                break
s_row_cycle = cycle(s_row_list)

if os.path.exists(output_dir_path):
    print("delete dir")
    shutil.rmtree(output_dir_path)
os.makedirs(output_dir_path, exist_ok=True)

total_files = len(t_csv_files)
for count, t_csv_file in enumerate(t_csv_files, start=1):
    t_file_path = os.path.join(target_dir_path, t_csv_file)
    print(t_file_path)
    df = pd.read_csv(t_file_path)
    daytime_index = df.columns.get_loc("daytime")
    new_data = []
    for _, row in df.iterrows():
        s_row = next(s_row_cycle)
        t_row = row.copy()
        s_row[daytime_index] = str(t_row["daytime"])
        new_data.append(t_row.tolist())
        new_data.append(s_row)
    new_df = pd.DataFrame(new_data, columns=df.columns)
    new_df = new_df.sort_values(by="daytime")
    print(new_df)
    new_df.to_csv(f"{output_dir_path}/{t_csv_file}", index=False)
    print(f"{count}/{total_files} : complete")

print("処理が完了しました")
