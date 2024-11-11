import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler

from main import driftDetector, featuresExtractor, modelSender, modelTrainer, trafficServer, modelCreator, modelEvaluator


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"Cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)

# ----- Main

async def main():

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
    repeat_count: int = settings["TrainingDefine"]["REPEAT_COUNT"]

    online_mode: bool = bool(settings["ONLINE_MODE"])
    dynamic_mode: bool = bool(settings["RetrainingCycle"]["DYNAMIC_MODE"])
    static_interval: int = settings["RetrainingCycle"]["STATIC_INTERVAL"]
    evaluate_unit_interval: int = settings["EVALUATE_UNIT_INTERVAL"]

    scalar = StandardScaler()
    is_pass_exist(datasets_folder_path)
    end_daytime = beginning_daytime + target_range


    retraining_daytime = evaluate_unit_end_daytime = beginning_daytime
    retraining_feature_matrix = evaluate_epoch_feature_matrix = []
    training_first_reading_flag = evaluate_first_reading_flag = True
    scaled_flag = False

    # --- Create output directory
    output_dir_path: str = f"src/main/outputs/training/{init_time}"
    os.makedirs(output_dir_path)
    os.makedirs(f"{output_dir_path}/model_weights")

    # --- Set results
    training_results = pd.DataFrame(columns=["daytime", "training_time", "benign_count", "malicious_count"])
    training_results_list = training_results.values
    evaluate_results = pd.DataFrame(columns=["daytime","accuracy", "f1_score", "precision", "recall"])
    evaluate_results_list = evaluate_results.values

    # --- Create foundation model
    model = modelCreator.main()

    # --- Online mode or not

    if not online_mode:
        print("\n- offline mode activated")
        if dynamic_mode:
            print("- dynamic mode activated")
            for dataset_file in os.listdir(datasets_folder_path):
                dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
        else:
            print("- static mode activated")
            for dataset_file in os.listdir(datasets_folder_path):
                dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
                print("- file change")
                with open(dataset_file_path, mode='r') as file:
                    reader = csv.reader(file)
                    headers = next(reader)  # 最初の行をヘッダーとして読み込む
                    timestamp_index = headers.index("timestamp")  # "timestamp" カラムのインデックスを取得

                    for row in reader:
                        timestamp = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")
                        # 行のタイムスタンプが開始時刻より前だった場合は無視
                        if timestamp >= beginning_daytime:
                            # 行のタイムスタンプが終了時刻を超過していた場合処理中止
                            if timestamp > end_daytime:
                                break
                            else:
                                # --- Training
                                if timestamp > retraining_daytime:
                                    if not training_first_reading_flag:
                                        print("\n--- retraining model")
                                        df_training = pd.DataFrame(retraining_feature_matrix)
                                        model, training_results_array = modelTrainer.main(
                                            model=model,
                                            df=df_training,
                                            output_dir_path=output_dir_path,
                                            scalar=scalar,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            repeat_count=repeat_count,
                                            retraining_daytime=retraining_daytime
                                        )
                                        training_results_list = np.vstack([training_results_list, training_results_array])
                                    retraining_feature_matrix = [row]
                                    retraining_daytime += timedelta(seconds=static_interval)
                                    training_first_reading_flag = False
                                else:
                                    retraining_feature_matrix.append(row)
                                # --- Evaluate
                                if timestamp > evaluate_unit_end_daytime:
                                    if not evaluate_first_reading_flag:
                                        print("evaluate model ... ")
                                        evaluate_df = pd.DataFrame(evaluate_epoch_feature_matrix)
                                        evaluate_results_array, scaled_flag = modelEvaluator.main(
                                            model=model,
                                            evaluate_dataframe=evaluate_df,
                                            scaler=scalar,
                                            scaled_flag=scaled_flag,
                                            evaluate_daytime=evaluate_unit_end_daytime - timedelta(seconds=evaluate_unit_interval / 2)
                                        )
                                        evaluate_results_list = np.vstack([evaluate_results_list, evaluate_results_array])
                                    evaluate_epoch_feature_matrix = [row]
                                    evaluate_unit_end_daytime += timedelta(seconds=evaluate_unit_interval)
                                    evaluate_first_reading_flag = False
                                else:
                                    evaluate_epoch_feature_matrix.append(row)
                        else:
                            pass

    else:
        print("\n- online mode activated")

    # --- Save settings_log and results
    with open(f"{output_dir_path}/settings_log_edge.json", "w") as f:
        json.dump(settings, f, indent=1)  # type:

    evaluate_results = pd.concat([evaluate_results, pd.DataFrame(evaluate_results_list, columns=evaluate_results.columns)])
    evaluate_results.to_csv(os.path.join(output_dir_path, "results_evaluate.csv"), index=False)
    training_results = pd.concat([training_results, pd.DataFrame(training_results_list, columns=training_results.columns)])
    training_results.to_csv(os.path.join(output_dir_path, "results_training.csv"), index=False)

if __name__ == "__main__":
    asyncio.run(main())
