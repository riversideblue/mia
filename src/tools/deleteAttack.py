import csv
import os
from datetime import timedelta, datetime

import pandas as pd


# 指定された期間に存在する label = 1 のデータ行を削除
def main():

    # Input
    target_dir_path = "src/main/traffic_data/csv/wt2023"
    beginning_daytime = "2023-04-04 15:14:24"
    beginning_daytime = datetime.strptime(beginning_daytime, "%Y-%m-%d %H:%M:%S")
    attack_delete_range = 2880 # minute

    ad_delta = timedelta(minutes=attack_delete_range)
    timestamp = beginning_daytime + ad_delta
    ts_str = timestamp.strftime("%m%d%H%M%S")

    # --- Create output directory
    output_dir_path: str = f"src/main/traffic_data/csv/{os.path.basename(target_dir_path)}-da-{ts_str}"
    os.makedirs(output_dir_path)

    for count,target_file in enumerate(os.listdir(target_dir_path),start=1):
        target_file_path = f"{target_dir_path}/{target_file}"

        df = pd.read_csv(target_file_path)
        df['daytime'] = pd.to_datetime(df['daytime'])  # datetime型に変換
        filtered_df = df.drop(df[(df['daytime'] < timestamp) & (df['label'] == 1)].index)

        # 結果を新しいCSVファイルに保存
        filtered_df.to_csv(f'{output_dir_path}/{os.path.basename(target_file_path)}', index=False)
        print(f"{count}/{len(os.listdir(target_dir_path))} : complete")

    print("--- Complete processing")
    print(f"- delete {attack_delete_range} minute malicious data")
    print(f"- from {beginning_daytime} to {timestamp}")

if __name__ == "__main__":
    main()