import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import modelCreator, dynamicTerminator, staticTerminator, ntTerminator

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
    output_dir_path: str = f"src/main/outputs/{init_time}"
    os.makedirs(output_dir_path)
    os.makedirs(f"{output_dir_path}/model_weights")

    # --- Set results
    training_results_column = ["daytime", "accuracy", "loss", "training_time", "benign_count", "malicious_count", "flow_num"]
    training_results_list = np.empty((0,len(training_results_column)),dtype=object)
    evaluate_results_column = ["daytime", "TP", "FN", "FP", "TN", "flow_num", "TP_rate", "FN_rate", "FP_rate", "TN_rate", "accuracy", "precision", "f1_score", "loss", "benign_rate"]
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

    processing_beginning_time = time.time()

    # --- Terminate
    if retraining_mode == "dynamic":
         training_results_list,evaluate_results_list,end_daytime = dynamicTerminator.main(
             online_mode= online_mode,
             datasets_folder_path=datasets_folder_path,
             o_dir_path=output_dir_path,
             beginning_dtime=beginning_daytime,
             end_dtime=end_daytime,
             model=model,
             epochs=epochs,
             batch_size=batch_size,
             eval_unit_int=evaluate_unit_interval,
             past_w_size=past_window_size,
             present_w_size=present_window_size,
             threshold=threshold,
             rtr_results_list=training_results_list,
             eval_results_list=evaluate_results_list
         )
    elif retraining_mode == "static":
        training_results_list,evaluate_results_list,end_daytime = staticTerminator.main(
            online_mode=online_mode,
            datasets_folder_path=datasets_folder_path,
            output_dir_path=output_dir_path,
            beginning_daytime=beginning_daytime,
            end_daytime=end_daytime,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            static_interval=static_interval,
            evaluate_unit_interval=evaluate_unit_interval,
            list_rtr_results=training_results_list,
            list_eval_results=evaluate_results_list
        )
    elif retraining_mode == "non-training":
        training_results_list,evaluate_results_list,end_daytime = ntTerminator.main(
            online_mode=online_mode,
            datasets_folder_path=datasets_folder_path,
            beginning_daytime=beginning_daytime,
            end_daytime=end_daytime,
            model=model,
            evaluate_unit_interval=evaluate_unit_interval,
            list_rtr_results=training_results_list,
            list_eval_results=evaluate_results_list
        )
    else:
        print("retraining mode invalid")
        sys.exit(1)

    processing_time = time.time() - processing_beginning_time

    # --- Save settings_log
    settings["Log"]["END_DAYTIME"] = end_daytime.isoformat()
    settings["Log"]["Processing_TIME"] = processing_time
    with open(f"{output_dir_path}/settings_log_edge.json", "w") as f:
        json.dump(settings, f, indent=1)  # type:

    # --- Results processing 修正する！！！！
    additional_results_column = ["nmr_fn_rate", "nmr_benign_rate"]
    additional_results_list = []

    sum_flow_num = np.sum(evaluate_results_list[:, 5])

    # nmr_flow_num_ratio
    min_max_scaler = MinMaxScaler()
    flow_num_rate = evaluate_results_list[:, 5] / sum_flow_num
    reshaped_flow_num_rate = flow_num_rate.reshape(-1, 1)
    scaled_flow_num_rate = min_max_scaler.fit_transform(reshaped_flow_num_rate)
    additional_results_list.append(scaled_flow_num_rate.flatten())

    # nmr_benign_ratio
    reshaped_benign_ratio = evaluate_results_list[:, 14].reshape(-1, 1)
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