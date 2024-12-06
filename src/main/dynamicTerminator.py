import csv
import os
import sys
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from main import modelTrainer, modelEvaluator

class DriftManager:
    def __init__(self,past_window_size,present_window_size,threshold):

        self.flow_count = 0
        self.past_window = deque(maxlen=past_window_size)
        self.past_window_size = past_window_size
        self.present_window = deque(maxlen=present_window_size)
        self.present_window_size = present_window_size
        self.first_wait_count = past_window_size + present_window_size
        self.threshold = threshold

    def detection(self,flow):

        if len(self.past_window) == self.past_window_size:
            self.present_window.append(self.past_window.popleft())

        self.past_window.append(flow)

        self.flow_count += 1

        if self.flow_count > self.first_wait_count:
            ave_past = sum(self.past_window) / self.past_window_size
            ave_present = sum(self.present_window) / self.present_window_size
            if abs(ave_present - ave_past) > self.threshold:
                return True

        return False


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
        past_window_size,
        present_window_size,
        threshold,
        training_results_list,
        evaluate_results_list
):
    y_true = []
    y_pred = []
    retraining_feature_matrix = []
    drift_manager = DriftManager(past_window_size,present_window_size,threshold)

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")

        first_timestamp_flag = True
        first_evaluate_flag = True
        end_flag = False
        next_evaluate_daytime = beginning_daytime

        for dataset_file in os.listdir(datasets_folder_path):
            if end_flag :
                return training_results_list,evaluate_results_list,end_daytime

            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む

                ex_addr_index = headers.index("timestamp")
                in_addr_index = headers.index("timestamp")
                timestamp_index = headers.index("daytime")
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
                label_index = headers.index("label")

                for row in reader:
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")
                    flow_num:int = int(row[rcv_packet_count]) + int(row[snd_packet_count])
                    feature = np.array(row[3:-1], dtype=float).reshape(1, -1)
                    target = int(row[label_index])

                    # --- Beginning and end daytime filter
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
                                evaluate_daytime = next_evaluate_daytime - timedelta(seconds=evaluate_unit_interval / 2)
                                evaluate_results_array = modelEvaluator.main(y_true, y_pred)
                                evaluate_results_array = np.append([evaluate_daytime], evaluate_results_array)
                                evaluate_results_list = np.vstack([evaluate_results_list, evaluate_results_array])

                            y_true = []
                            y_pred = []
                            next_evaluate_daytime += timedelta(seconds=evaluate_unit_interval)
                            first_evaluate_flag = False

                        # --- Prediction
                        y_pred.append(model(feature,training=False)[0][0].item())
                        y_true.append(target)

                        # --- Drift detection
                        drift_flag = drift_manager.detection(flow_num)

                        # --- Training
                        if drift_flag:
                            print("Drift Detected")
                            df_training = pd.DataFrame(retraining_feature_matrix)
                            print(df_training)
                            retraining_daytime = df_training.iloc[-1,2] # データセット内の最後のフローがキャプチャされた時間
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

        # --- End dynamic-offline processing
        if not end_flag:
            end_daytime = datetime.strptime(retraining_feature_matrix[-1][2], "%Y-%m-%d %H:%M:%S")

    return training_results_list,evaluate_results_list,end_daytime