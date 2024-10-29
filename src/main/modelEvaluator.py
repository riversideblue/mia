import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from queue import Queue

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from main import modelCreator


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"Cannot find evaluate dataset: {os.path.basename(path)}")
        sys.exit(1)


def is_dataset_out_of_range(dataset_file_name, beginning_daytime, end_daytime):
    within_range_flag: bool = False
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print(f"evaluate: {dataset_captured_datetime} dataset ignored by Evaluate/BEGINNING_DAYTIME setting")
        within_range_flag = True
    elif not dataset_captured_datetime <= end_daytime:
        print(f"evaluate: {dataset_captured_datetime} dataset ignored by Evaluate/TargetRange settings")
        within_range_flag = True
    return within_range_flag


def save_results(results_list, results, output_dir):
    evaluate = pd.concat([results, pd.DataFrame(results_list, columns=results.columns)])
    evaluate.to_csv(os.path.join(output_dir, "results_evaluate.csv"), index=False)


def save_settings_log(settings_log, output_dir):
    with open(os.path.join(output_dir, "settings_log_evaluator.json"), "w") as f:
        json.dump(settings_log, f, indent=1)  # type: ignore


def model_evaluate(model, dataset_file, retraining_time, next_retraining_time, scaler, scaled_flag):

    df = pd.read_csv(dataset_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    explanatory_values = df[(df['timestamp'] >= retraining_time) & (df['timestamp'] < next_retraining_time)].iloc[:, 1:-1]
    target_values = df[(df['timestamp'] >= retraining_time) & (df['timestamp'] < next_retraining_time)].loc[:, "label"].values

    # --- Check dataset not null
    if len(df) == 0:
        print(f"specified test dataset file: {dataset_file} no data \n>")
        sys.exit()
    else:
        if not scaled_flag:
            print("feature matrix scaling first time")
            scaled_feature_matrix = scaler.fit_transform(explanatory_values)
            scaled_flag = True
        else:
            scaled_feature_matrix = scaler.transform(explanatory_values)

        # --- Prediction
        prediction_values = model.predict(scaled_feature_matrix)
        prediction_binary_values = (prediction_values >= 0.5).astype(int)

        # --- Evaluate
        accuracy = accuracy_score(target_values, prediction_binary_values)
        precision = precision_score(target_values, prediction_binary_values)
        recall = recall_score(target_values, prediction_binary_values)
        f1 = f1_score(target_values, prediction_binary_values)

        result = [retraining_time, accuracy, precision, recall, f1]
        retraining_time = next_retraining_time

        # --- Confusion matrix
        matrix = confusion_matrix(target_values, prediction_binary_values, labels=[0, 1])

        return result, matrix, scaled_flag, retraining_time


# ----- Model other
def main():
    # --- Load settings
    settings = json.load(open("src/main/settings.json", "r"))
    settings["Log"] = {}

    # --- Set results dataframe
    results = pd.DataFrame(columns=["flow_featured_time","accuracy", "f1_score", "precision", "recall"])
    results_list = results.values

    # --- Field
    settings["Log"]["Evaluate"] = {}
    online_mode: bool = settings["Evaluate"]["ONLINE_MODE"]
    model_path = settings["Evaluate"]["MODEL_PATH"]
    training_output_path: str = settings["Evaluate"]["TRAINING_OUTPUT_PATH"]
    datasets_folder_path: str = settings["Evaluate"]["DATASETS_FOLDER_PATH"]
    beginning_daytime = datetime.strptime(settings["Evaluate"]["BEGINNING_DAYTIME"], "%Y-%m-%d %H:%M:%S")
    settings["Log"]["Evaluate"]["BEGINNING_DAYTIME"] = beginning_daytime.isoformat()
    days: int = settings["Evaluate"]["TargetRange"]["DAYS"]
    hours: int = settings["Evaluate"]["TargetRange"]["HOURS"]
    minutes: int = settings["Evaluate"]["TargetRange"]["MINUTES"]
    seconds: int = settings["Evaluate"]["TargetRange"]["SECONDS"]
    end_daytime: datetime = beginning_daytime + timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    settings["Log"]["Evaluate"]["END_DAYTIME"] = end_daytime.isoformat()
    scaler = StandardScaler()
    scaled_flag = False
    is_pass_exist(training_output_path)
    is_pass_exist(datasets_folder_path)
    results_matrix = [[0, 0], [0, 0]]

    if online_mode:

        file_name_queue = Queue()
        for file_name in os.listdir(model_path):
            file_name_queue.put(file_name)

        while not file_name_queue.empty():

            file_name = file_name_queue.get()
            date_str = file_name.split('-weights')[0]
            retraining_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            pickle_path = f"{model_path}/{file_name}"

            model = modelCreator.main()
            with open(pickle_path, "rb") as f:
                init_weights = pickle.load(f)
                model.set_weights(init_weights) # インデント下げる

            next_file_name = file_name_queue.get()
            date_str = next_file_name.split('-weights')[0]
            next_retraining_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

            for dataset_file in os.listdir(datasets_folder_path):
                dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
                results_array, matrix, scaled_flag, retraining_time = (
                    model_evaluate(model, dataset_file_path, retraining_time, next_retraining_time, scaler, scaled_flag))
                results_list = np.vstack([results_list, results_array])
                results_matrix = matrix + results_matrix
                scaled_flag = scaled_flag
                if retraining_time == next_retraining_time:
                    next_file_name = file_name_queue.get()
                    date_str = next_file_name.split('-weights')[0]
                    next_retraining_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    else:
        # --- Load model weight from model path
        model = modelCreator.main()
        with open(model_path, "rb") as f:
            init_weights = pickle.load(f)
            model.set_weights(init_weights)

        # --- Calling
        for dataset_file in os.listdir(datasets_folder_path):
            dataset_file_path: str = f"{datasets_folder_path}/{dataset_file}"
            results_array, matrix, scaled_flag = model_evaluate(model, dataset_file_path, scaler, scaled_flag)
            results_list = np.vstack([results_list, results_array])
            results_matrix = matrix + results_matrix
            scaled_flag = scaled_flag

    save_results(
        results_list=results_list,
        results=results,
        output_dir=training_output_path
    )
    framed_matrix = pd.DataFrame(
        data=results_matrix,
        index=["PredictionPositive", "PredictionNegative"],
        columns=["TargetPositive", "TargetNegative"]
    )
    print(framed_matrix)

    # --- Save settings_log and results
    save_settings_log(settings, training_output_path)


if __name__ == "__main__":
    main()
