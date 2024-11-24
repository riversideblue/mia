import csv
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from main import modelTrainer, modelEvaluator


def main(
        online_mode,
        datasets_folder_path,
        output_dir_path,
        beginning_daytime,
        end_daytime,
        model,
        scaler,
        epochs,
        batch_size,
        static_interval,
        evaluate_unit_interval,
        training_results_list,
        evaluate_results_list
):
    # --- Confusion matrix = [tp,fn,fp,tp]
    confusion_matrix = np.empty((0, 4), dtype=int)
    retraining_feature_matrix = []
    evaluate_epoch_feature_matrix = []

    if online_mode:
        print("- < static/online mode activate >")
    else:
        print("- < static/offline mode activate >")

        first_timestamp_flag = True
        first_training_flag = True
        first_evaluate_flag = True
        end_flag = False
        scaled_flag = False
        next_retraining_daytime = beginning_daytime
        next_evaluate_daytime = beginning_daytime

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
                    feature = np.array(row[3:-1], dtype=float).reshape(1, -1)
                    target = int(row[label_index])

                    # --- Beginning and end filter
                    if first_timestamp_flag: # 最初の行のtimestamp
                        if timestamp > beginning_daytime:
                            print("- error : beginning_daytime should be within datasets range")
                            sys.exit(1)
                        else:
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
                                print("--- evaluate model")
                                evaluate_results_array = modelEvaluator.main(confusion_matrix)
                                evaluate_daytime = next_evaluate_daytime - timedelta(seconds=evaluate_unit_interval / 2)
                                evaluate_results_array = np.append([evaluate_daytime], evaluate_results_array)
                                evaluate_results_list = np.vstack([evaluate_results_list, evaluate_results_array])

                            next_evaluate_daytime += timedelta(seconds=evaluate_unit_interval)
                            first_evaluate_flag = False

                            # dataが存在しない区間は直前の結果を流用
                            while timestamp > next_evaluate_daytime:
                                print(f"- < no data range detected : {timestamp} >")
                                evaluate_results_array = evaluate_results_list[-1].copy()
                                evaluate_results_array[0] = next_evaluate_daytime - timedelta(
                                    seconds=evaluate_unit_interval / 2)
                                evaluate_results_array[8] = 0 # benign count = 0
                                evaluate_results_array[9] = 0 # malicious count = 0
                                evaluate_results_array[10] = 0 # flow num = 0
                                evaluate_results_array[11] = 0 # benign rate = 0
                                evaluate_results_list = np.vstack(
                                    [evaluate_results_list, evaluate_results_array])
                                next_evaluate_daytime += timedelta(seconds=evaluate_unit_interval)

                        # --- Prediction
                        prediction_value = model.predict(feature, verbose=0)
                        prediction_binary = (prediction_value >= 0.5).astype(int)
                        if target == 1:
                            if prediction_binary == 1:  # TP
                                confusion_matrix = np.vstack([confusion_matrix, [1, 0, 0, 0]])
                            elif prediction_binary == 0:  # FP
                                confusion_matrix = np.vstack([confusion_matrix, [0, 1, 0, 0]])
                        elif target == 0:
                            if prediction_binary == 1:  # FN
                                confusion_matrix = np.vstack([confusion_matrix, [0, 0, 1, 0]])
                            elif prediction_binary == 0:  # TN
                                confusion_matrix = np.vstack([confusion_matrix, [0, 0, 0, 1]])

                        # --- Training
                        if timestamp > next_retraining_daytime:
                            if not first_training_flag:
                                print("\n--- retraining model")
                                df_training = pd.DataFrame(retraining_feature_matrix)
                                model, training_results_array = modelTrainer.main(
                                    model=model,
                                    df=df_training,
                                    output_dir_path=output_dir_path,
                                    scalar=scaler,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    retraining_daytime=next_retraining_daytime
                                )
                                training_results_list = np.vstack([training_results_list, training_results_array])

                            retraining_feature_matrix = [row]
                            next_retraining_daytime += timedelta(seconds=static_interval)
                            first_training_flag = False

                            while timestamp > next_retraining_daytime:
                                print(f"- < no data range detected : {timestamp} >")
                                training_results_array = training_results_list[-1].copy()
                                training_results_array[0] = next_retraining_daytime - timedelta(seconds=static_interval/2)
                                training_results_array[3] = 0
                                training_results_array[4] = 0
                                training_results_array[5] = 0
                                training_results_array[6] = 0
                                training_results_list = np.vstack(
                                    [training_results_list, training_results_array])
                                next_retraining_daytime += timedelta(seconds=static_interval)

                        else:
                            retraining_feature_matrix.append(row)

        # --- End static-offline processing
        if not end_flag:
            end_daytime = datetime.strptime(retraining_feature_matrix[-1][2], "%Y-%m-%d %H:%M:%S")

    return training_results_list,evaluate_results_list,end_daytime