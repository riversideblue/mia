import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from main.other import modelCreator


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
    with open(os.path.join(output_dir, "settings_log.json"), "w") as f:
        json.dump(settings_log, f, indent=1)  # type: ignore


def model_evaluate(model, dataset_file, scaler, scaled_flag):
    evaluate_df = pd.read_csv(dataset_file)
    explanatory_values = evaluate_df.iloc[:, 3:-1].values
    target_values = evaluate_df.loc[:, "label"].values

    # --- Check dataset not null
    if len(evaluate_df) == 0:
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

        # --- Confusion matrix
        matrix = confusion_matrix(target_values, prediction_binary_values, labels=[0, 1])

        return [accuracy, precision, recall, f1], matrix, scaled_flag


# ----- Model other
def main():
    # --- Load settings
    settings = json.load(open("src/main/other/settings.json", "r"))
    settings["Log"] = {}

    # --- Set results dataframe
    results = pd.DataFrame(columns=["accuracy", "f1_score", "precision", "recall"])
    results_list = results.values

    # --- Field
    settings["Log"]["Evaluate"] = {}
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

    # --- Load model weight from model path
    model = modelCreator.main()
    with open(model_path, "rb") as f:
        init_weights = pickle.load(f)
        model.set_weights(init_weights)

    # --- Calling
    for dataset_file in os.listdir(datasets_folder_path):
        if not is_dataset_out_of_range(dataset_file, beginning_daytime, end_daytime):
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
