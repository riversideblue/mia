import csv
import os

from datetime import datetime

import numpy as np
import driftDetection as DD
import tmClass

def main(
        online_mode,
        datasets_folder_path,
        o_dir_path,
        beginning_dtime,
        end_dtime,
        model,
        epochs,
        batch_size,
        eval_unit_int,
        past_w_size,
        present_w_size,
        threshold,
        rtr_results_list,
        eval_results_list
):

    t = tmClass.TerminateManager(beginning_dtime, end_dtime, eval_unit_int, o_dir_path, epochs, batch_size)

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        w = DD.Window(present_w_size, past_w_size, threshold, row_len=18)

        for dataset_file in os.listdir(datasets_folder_path):
            if t.end_flag :
                return rtr_results_list,eval_results_list,t.end_dtime

            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")

            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む

                timestamp_index = headers.index("daytime")
                label_index = headers.index("label")

                for row in reader:
                    feature = np.array(row[3:-1], dtype=np.float32)
                    target = int(row[label_index])
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")

                    if t.b_filter: # 開始フィルタ
                        if t.b_filtering(timestamp):
                            continue
                    elif t.e_filter(timestamp): # 終了フィルタ
                        break

                    w.update(row)
                    # --- Evaluate
                    if timestamp > t.next_eval_dtime:
                        eval_results_list = t.call_eval(eval_results_list)
                    # --- Prediction
                    t.call_pred(model, feature=feature,target=target)
                    # --- DD & Retraining
                    if DD.TTest(w.fnum_present(), w.fnum_past(), threshold):
                        rtr_results_list = t.call_rtr(model, w.present_window, rtr_results_list)

        # --- End dynamic-offline processing
        if not t.end_flag:
            t.end_dtime = datetime.strptime(w.present_window[0][2], "%Y-%m-%d %H:%M:%S")
            t.end_flag = True

    return rtr_results_list,eval_results_list,t,end_dtime