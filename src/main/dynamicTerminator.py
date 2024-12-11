import csv
import os
import sys
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from main import modelTrainer, modelEvaluator

def drift_detect(present_window,past_window,threshold):

    # 必要な列を整数型に変換して取り出し
    rcv_present = np.array(present_window)[:,3].astype(int)
    snd_present = np.array(present_window)[:,4].astype(int)
    fn_present = rcv_present + snd_present
    ave_fn_present = np.mean(fn_present)

    rcv_past = np.array(past_window)[:,3].astype(int)
    snd_past = np.array(past_window)[:,4].astype(int)
    fn_past = rcv_past + snd_past
    ave_fn_past = np.mean(fn_past)

    if abs(ave_fn_present - ave_fn_past) > threshold:
        return True
    return False

def main(
        online_mode,
        datasets_folder_path,
        output_dir_path,
        beginning_daytime,
        end_daytime,
        model,
        epochs,
        batch_size,
        evaluate_unit_interval,
        past_window_size,
        present_window_size,
        threshold,
        list_rtr_results,
        list_eval_results
):

    y_true = []
    y_pred = []
    counter = 0

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        drift_flag = False
        first_timestamp_flag = True
        first_evaluate_flag = True
        end_flag = False
        next_evaluate_daytime = beginning_daytime

        for dataset_file in os.listdir(datasets_folder_path):
            if end_flag :
                return list_rtr_results,list_eval_results,end_daytime

            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む
                timestamp_index = headers.index("daytime")
                label_index = headers.index("label")

                for row in reader:
                    batch = np.array(row[3:-1], dtype=np.float32)
                    target = int(row[label_index])
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")

                    # --- Beginning and end daytime filter
                    if first_timestamp_flag: # 最初の行のtimestamp
                        if timestamp > beginning_daytime:
                            print("- error : beginning_daytime should be within datasets range")
                            sys.exit(1)
                        else:
                            # Windowの定義
                            past_window = deque(
                                np.full((past_window_size, len(row)), -threshold, dtype=object),
                                maxlen=past_window_size
                            )
                            present_window = deque(
                                np.full((present_window_size, len(row)),  -threshold, dtype=object),
                                maxlen=present_window_size
                            )
                            first_timestamp_flag = False
                    elif timestamp < beginning_daytime: # beginning_daytime以前の行は読み飛ばす
                        pass
                    elif timestamp > end_daytime: # timestampがend_daytimeを超えた時
                        print("- < detected end_daytime >")
                        end_flag = True
                        break

                    else:
                        past_window.append(present_window.popleft())
                        present_window.append(np.array(row, dtype=object))

                        # --- Evaluate
                        if timestamp > next_evaluate_daytime:
                            if not first_evaluate_flag:
                                print("--- evaluate model")
                                evaluate_daytime = next_evaluate_daytime - timedelta(seconds=evaluate_unit_interval / 2)
                                evaluate_results_array = modelEvaluator.main(y_true, y_pred)
                                evaluate_results_array = np.append([evaluate_daytime], evaluate_results_array)
                                list_eval_results = np.vstack([list_eval_results, evaluate_results_array])
                            y_true = []
                            y_pred = []
                            next_evaluate_daytime += timedelta(seconds=evaluate_unit_interval)
                            first_evaluate_flag = False

                        # --- Prediction
                        y_pred.append(model.predict_on_batch(batch.reshape(1, -1))[0][0])
                        y_true.append(target)

                        # --- Drift detection
                        counter += 1
                        if counter >= present_window_size:
                            drift_flag = drift_detect(present_window,past_window,threshold)

                        # --- Retraining
                        if drift_flag:
                            print("--- Drift Detected")
                            counter = 0
                            df = pd.DataFrame(np.array(present_window))
                            features = df.iloc[:, 3:-1].astype(float)
                            targets = df.iloc[:, -1].astype(int)
                            retraining_daytime = datetime.strptime(df.iloc[-1,2], "%Y-%m-%d %H:%M:%S")
                            model, arr_rtr_results = modelTrainer.main(
                                model=model,
                                features=features,
                                targets=targets,
                                output_dir_path=output_dir_path,
                                epochs=epochs,
                                batch_size=batch_size,
                                retraining_daytime=retraining_daytime
                            )
                            list_rtr_results = np.vstack([list_rtr_results, arr_rtr_results])
                            drift_flag = False

        # --- End dynamic-offline processing
        if not end_flag:
            end_daytime = datetime.strptime(present_window[0][2], "%Y-%m-%d %H:%M:%S")

    return list_rtr_results,list_eval_results,end_daytime