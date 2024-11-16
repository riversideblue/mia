import csv
import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from main import modelCreator, dynamicTerminator, staticTerminator, modelEvaluator


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"Cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)

# ----- Main

def main():

    # --- Load settings
    settings = json.load(open("src/main/settings.json", "r"))
    settings["Log"] = {}

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")
    settings["Log"]["INIT_TIME"] = init_time

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings["OS"]["TF_CPP_MIN_LOG_LEVEL"]  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings["OS"]["TF_FORCE_GPU_ALLOW_GROWTH"]  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings["OS"]["CUDA_VISIBLE_DEVICES"]  # cpu : -1

    # --- Field
    datasets_folder_path: str = settings["DATASETS_FOLDER_PATH"]
    beginning_daytime = datetime.strptime(settings["BEGINNING_DAYTIME"], "%Y-%m-%d %H:%M:%S")
    settings["Log"]["BEGINNING_DAYTIME"] = beginning_daytime.isoformat()
    days: int = settings["TargetRange"]["DAYS"]
    hours: int = settings["TargetRange"]["HOURS"]
    minutes: int = settings["TargetRange"]["MINUTES"]
    seconds: int = settings["TargetRange"]["SECONDS"]
    target_range = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    end_daytime: datetime = beginning_daytime + target_range
    settings["Log"]["END_DAYTIME"] = end_daytime.isoformat()
    epochs: int = settings["TrainingDefine"]["EPOCHS"]
    batch_size: int = settings["TrainingDefine"]["BATCH_SIZE"]

    online_mode: bool = bool(settings["ONLINE_MODE"])
    retraining_mode = settings["RETRAINING_MODE"]
    static_interval: int = settings["STATIC_INTERVAL"]
    evaluate_unit_interval: int = settings["EVALUATE_UNIT_INTERVAL"]

    past_window_size = settings["DriftDetection"]["PAST_WINDOW_SIZE"]
    present_window_size = settings["DriftDetection"]["PRESENT_WINDOW_SIZE"]
    threshold = settings["DriftDetection"]["THRESHOLD"]

    scaler = StandardScaler()
    is_pass_exist(datasets_folder_path)

    # --- Create output directory
    output_dir_path: str = f"src/main/outputs/training/{init_time}"
    os.makedirs(output_dir_path)
    os.makedirs(f"{output_dir_path}/model_weights")

    # --- Set results
    training_results_column = ["daytime", "accuracy", "loss", "training_time", "benign_count", "malicious_count", "flow_num"]
    training_results_list = np.empty((0,len(training_results_column)),dtype=object)
    evaluate_results_column = ["daytime", "accuracy", "precision", "recall", "f1_score", "loss", "benign_count", "malicious_count", "benign_rate", "flow_num"]
    evaluate_results_list = np.empty((0,len(evaluate_results_column)),dtype=object)

    # --- Foundation model setting
    foundation_model_path = settings["FOUNDATION_MODEL_PATH"]
    model = modelCreator.main()
    if foundation_model_path == "":
        print("- start with new model ...")
    elif os.path.exists(foundation_model_path):
        with open(foundation_model_path, 'rb') as f:
            print(f"- start with exist model {foundation_model_path} ...")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        print("- invalid path")
        sys.exit(1)

    # --- Terminate
    if retraining_mode == "dynamic":
         training_results_list,evaluate_results_list = dynamicTerminator.main(
             online_mode= online_mode,
             datasets_folder_path=datasets_folder_path,
             output_dir_path=output_dir_path,
             beginning_daytime=beginning_daytime,
             end_daytime=end_daytime,
             model=model,
             scaler=scaler,
             epochs=epochs,
             batch_size=batch_size,
             evaluate_unit_interval=evaluate_unit_interval,
             past_window_size=past_window_size,
             present_window_size=present_window_size,
             threshold=threshold,
             training_results_list=training_results_list,
             evaluate_results_list=evaluate_results_list
         )
    elif retraining_mode == "static":
        training_results_list,evaluate_results_list = staticTerminator.main(
            online_mode= online_mode,
            datasets_folder_path=datasets_folder_path,
            output_dir_path=output_dir_path,
            beginning_daytime=beginning_daytime,
            end_daytime=end_daytime,
            model=model,
            scaler=scaler,
            epochs=epochs,
            batch_size=batch_size,
            static_interval=static_interval,
            evaluate_unit_interval=evaluate_unit_interval,
            training_results_list=training_results_list,
            evaluate_results_list=evaluate_results_list
        )
    elif retraining_mode == "non-training":

        evaluate_epoch_feature_matrix = []
        first_timestamp_flag = True
        first_evaluate_flag = True
        end_flag = False
        scaled_flag = False
        evaluate_unit_end_daytime = beginning_daytime

        for dataset_file in os.listdir(datasets_folder_path):
            if end_flag : break
            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            print(f"- {dataset_file} set now")
            with open(dataset_file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # 最初の行をヘッダーとして読み込む

                timestamp_index = headers.index("timestamp")

                for row in reader:
                    timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")

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
                        if timestamp > evaluate_unit_end_daytime:
                            if not first_evaluate_flag:
                                print("--- evaluate model")
                                evaluate_df = pd.DataFrame(evaluate_epoch_feature_matrix)
                                print(evaluate_unit_end_daytime)
                                evaluate_results_array, scaled_flag = modelEvaluator.main(
                                    model=model,
                                    df=evaluate_df,
                                    scaler=scaler,
                                    scaled_flag=scaled_flag,
                                    evaluate_daytime=evaluate_unit_end_daytime - timedelta(
                                        seconds=evaluate_unit_interval / 2)
                                )
                                evaluate_results_list = np.vstack([evaluate_results_list, evaluate_results_array])

                            evaluate_epoch_feature_matrix = [row]
                            evaluate_unit_end_daytime += timedelta(seconds=evaluate_unit_interval)
                            first_evaluate_flag = False

                            # dataが存在しない区間は直前の結果を流用
                            while timestamp > evaluate_unit_end_daytime:
                                print(f"- < no data range detected : {timestamp} >")
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
    else:
        print("retraining mode invalid")
        sys.exit(1)

    # --- Save settings_log
    with open(f"{output_dir_path}/settings_log_edge.json", "w") as f:
        json.dump(settings, f, indent=1)  # type:

    # --- Results processing
    additional_results_column = ["nmr_flow_num_ratio", "nmr_benign_ratio"]
    additional_results_list = []

    sum_flow_num = np.sum(evaluate_results_list[:, 8])

    # nmr_flow_num_ratio
    min_max_scaler = MinMaxScaler()
    flow_num_rate = evaluate_results_list[:, 8] / sum_flow_num
    reshaped_flow_num_rate = flow_num_rate.reshape(-1, 1)
    scaled_flow_num_rate = min_max_scaler.fit_transform(reshaped_flow_num_rate)
    additional_results_list.append(scaled_flow_num_rate.flatten())

    # nmr_benign_ratio
    reshaped_benign_ratio = evaluate_results_list[:, 7].reshape(-1, 1)
    scaled_benign_ratio = min_max_scaler.fit_transform(reshaped_benign_ratio)
    additional_results_list.append(scaled_benign_ratio.flatten())

    # Convert additional results to a 2D array
    additional_results_list = np.array(additional_results_list).T  # 転置して列形式に変換

    # training results
    training_results = pd.DataFrame(training_results_list, columns=training_results_column)
    training_results.to_csv(os.path.join(output_dir_path, "results_training.csv"), index=False)

    # evaluate results
    evaluate_results = pd.DataFrame(evaluate_results_list, columns=evaluate_results_column)
    additional_results = pd.DataFrame(additional_results_list, columns=additional_results_column)

    # Combine evaluate_results with additional_results
    evaluate_results = pd.concat([evaluate_results, additional_results], axis=1)
    evaluate_results.to_csv(os.path.join(output_dir_path, "results_evaluate.csv"), index=False)

    print("\n")
    model.summary()


if __name__ == "__main__":
    main()
