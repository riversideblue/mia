import csv
import os
from datetime import timedelta, datetime

import pandas as pd


# 指定された期間に存在する label = 1 のデータ行を削除
def main():

    # Input
    target_dir_path = "src/main/traffic_data/csv/wt2022"
    beginning_daytime = "2022-08-03 16:50:07"
    beginning_daytime = datetime.strptime(beginning_daytime, "%Y-%m-%d %H:%M:%S")
    attack_delete_range = 600 # minute

    # --- Create output directory
    output_dir_path: str = f"src/main/traffic_data/csv/{os.path.basename(target_dir_path)}-da{attack_delete_range}"
    os.makedirs(output_dir_path)

    ad_delta = timedelta(minutes=attack_delete_range)
    timestamp = beginning_daytime + ad_delta

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