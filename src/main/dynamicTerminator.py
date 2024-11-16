import csv
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from main import modelTrainer, modelEvaluator, driftDetector


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
        evaluate_unit_interval,
        training_results_list,
        evaluate_results_list
):

    retraining_feature_matrix = []
    evaluate_epoch_feature_matrix = []

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")

        first_timestamp_flag = True
        first_training_flag = True
        first_evaluate_flag = True
        end_flag = False
        scaled_flag = False
        retraining_daytime = beginning_daytime
        evaluate_unit_end_daytime = beginning_daytime

        for dataset_file in os.listdir(datasets_folder_path):
            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む

                ex_addr_index = headers.index("timestamp")
                in_addr_index = headers.index("timestamp")
                timestamp_index = headers.index("timestamp")
                rcv_packet_count = headers.index("rcv_packet_count")
                snd_packet_count = headers.index("snd_packet_count")
                tcp_count = headers.index("tcp_count")
                udp_count = headers.index("udp_count")
                most_port = headers.index("most_port")
                port_count = headers.index("port_count")
                rcv_max_interval = headers.index("rcv_max_interval")
                rcv_min_interval = headers.index("rcv_min_interval")
                rcv_max_length = headers.index("rcv_max_length")
                rcv_min_length = headers.index("rcv_min_length")
                snd_max_interval = headers.index("snd_max_interval")
                snd_min_interval = headers.index("snd_min_interval")
                snd_max_length = headers.index("snd_max_length")
                snd_min_length = headers.index("snd_min_length")


                for row in reader:
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")

                    # --- Beginning and end filter

                    if first_timestamp_flag:
                        if beginning_daytime < timestamp:
                            print("beginning_daytime should be over datasets first timestamp")
                            sys.exit(1)
                        else:
                            first_timestamp_flag = False
                    elif beginning_daytime > timestamp:
                        pass
                    elif end_daytime < timestamp:
                        print("detected end_daytime")
                        break

                    # --- Drift detection

                    else:

                        drift_flag = driftDetector.main()

                        # --- Training
                        if drift_flag:
                            df_training = pd.DataFrame(retraining_feature_matrix)
                            retraining_daytime = df_training[2,-1] # データセット内の最後のフローがキャプチャされた時間
                            model, training_results_array = modelTrainer.main(
                                model=model,
                                df=df_training,
                                output_dir_path=output_dir_path,
                                scalar=scaler,
                                epochs=epochs,
                                batch_size=batch_size,
                                retraining_daytime=retraining_daytime
                            )
                            training_results_list = np.vstack(
                                [training_results_list, training_results_array])
                            retraining_feature_matrix = [row]
                        else:
                            retraining_feature_matrix.append(row)
                            print("... retraining matrix append")

                        # Evaluate
                        if timestamp > evaluate_unit_end_daytime:
                            if not first_evaluate_flag:
                                print("\n--- evaluate model")
                                evaluate_df = pd.DataFrame(evaluate_epoch_feature_matrix)
                                evaluate_results_array, scaled_flag = modelEvaluator.main(
                                    model=model,
                                    df=evaluate_df,
                                    scaler=scaler,
                                    scaled_flag=scaled_flag,
                                    evaluate_daytime=evaluate_unit_end_daytime - timedelta(
                                        seconds=evaluate_unit_interval / 2)
                                )
                                evaluate_results_list = np.vstack(
                                    [evaluate_results_list, evaluate_results_array])
                            evaluate_epoch_feature_matrix = [row]
                            evaluate_unit_end_daytime += timedelta(seconds=evaluate_unit_interval)

                            # dataが存在しない区間は直前の結果を流用
                            while timestamp > evaluate_unit_end_daytime:
                                print(f"\n- < no data range detected : {timestamp} >")
                                evaluate_results_array = evaluate_results_list[-1].copy()
                                evaluate_results_array[0] = evaluate_unit_end_daytime - timedelta(
                                    seconds=evaluate_unit_interval / 2)
                                evaluate_results_array[6] = 0 # benign count = 0
                                evaluate_results_array[7] = 0 # malicious count = 0
                                evaluate_results_array[8] = 0 # benign rate = 0
                                evaluate_results_array[9] = 0 # flow_num = 0
                                evaluate_results_list = np.vstack(
                                    [evaluate_results_list, evaluate_results_array])
                                evaluate_unit_end_daytime += timedelta(seconds=evaluate_unit_interval)
                        else:
                            evaluate_epoch_feature_matrix.append(row)

    # --- End dynamic-offline processing
    return training_results_list,evaluate_results_list
