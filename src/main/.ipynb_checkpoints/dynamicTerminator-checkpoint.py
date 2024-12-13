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
    w = DD.Window(present_w_size, past_w_size, threshold, row_len=18)

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        for dataset_file in os.listdir(datasets_folder_path):
            if t.end_flag:break

            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            file = open(dataset_file_path, mode='r')
            reader = csv.reader(file)
            headers = next(reader)  # 最初の行をヘッダーとして読み込む
            timestamp_index = headers.index("daytime")
            label_index = headers.index("label")

            for row in reader:
                t.c_time = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")
                feature = np.array(row[3:-1], dtype=np.float32)
                target = int(row[label_index])

                if t.e_filtering(t.c_time):break
                elif t.b_flag:
                    if t.b_filtering(t.c_time):continue

                w.update(row)
                # --- Evaluate
                if t.c_time > t.next_eval_dtime:
                    eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model, feature=feature,target=target)
                # --- DD & Retraining
                if DD.TTest(w.fnum_present(), w.fnum_past(), threshold):
                    rtr_results_list = t.call_rtr(model, w.present_window, rtr_results_list)
            file.close()

    return rtr_results_list,eval_results_list,t.c_time