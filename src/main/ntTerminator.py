import csv
import os
import sys
from datetime import datetime, timedelta

import numpy as np

from main import modelEvaluator

def main(
        online_mode,
        datasets_folder_path,
        beginning_daytime,
        end_daytime,
        model,
        evaluate_unit_interval,
        training_results_list,
        evaluate_results_list
):

    y_true = []
    y_pred = []

    if online_mode:
        print("- < non-training/online mode activate >")
    else:
        print("- < non-training/offline mode activate >")

        first_timestamp_flag = True
        first_evaluate_flag = True
        end_flag = False
        next_evaluate_daytime = beginning_daytime
        timestamp = beginning_daytime

        for dataset_file in os.listdir(datasets_folder_path):
            if end_flag :
                return training_results_list,evaluate_results_list,end_daytime

            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む

                timestamp_index = headers.index("daytime")
                label_index = headers.index("label")

                for row in reader:
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")
                    feature = np.array(row[3:-1],dtype=float).reshape(1,-1)
                    target = int(row[label_index])

                    # --- Beginning and end filter
                    if first_timestamp_flag: # 最初の行のtimestamp
                        if timestamp > beginning_daytime:
                            print("- error : beginning_daytime should be within datasets range")
                            sys.exit(1)
                        first_timestamp_flag = False
                    elif timestamp < beginning_daytime: # beginning_daytime以前の行は読み飛ばす
                        pass
                    elif timestamp > end_daytime: # timestampがend_daytimeを超えた時
                        print("- < detected end_daytime >")
                        end_flag = True
                        break

                    else:
                        # --- Evaluate
                        if timestamp > next_evaluate_daytime:
                            if not first_evaluate_flag:
                                print("-- evaluate model")
                                evaluate_daytime = next_evaluate_daytime - timedelta(seconds=evaluate_unit_interval / 2)
                                evaluate_results_array = modelEvaluator.main(y_true, y_pred)
                                evaluate_results_array = np.append([evaluate_daytime],evaluate_results_array)
                                evaluate_results_list = np.vstack([evaluate_results_list, evaluate_results_array])

                            y_true = []
                            y_pred = []
                            next_evaluate_daytime += timedelta(seconds=evaluate_unit_interval)
                            first_evaluate_flag = False

                    # --- Prediction
                    y_pred.append(model(feature,training=False).numpy()[0][0])
                    y_true.append(target)

        # --- End static-offline processing
        if not end_flag:
            end_daytime = timestamp

    return training_results_list,evaluate_results_list,end_daytime