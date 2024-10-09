import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def is_dataset_within_range(dataset_file_name,beginning_daytime,end_daytime):
    within_range_flag: bool = True
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print(f">\n=  no dataset in specified beginning-daytime")
        within_range_flag = False
    elif not dataset_captured_datetime <= end_daytime:
        print(f">\n= no dataset in specified end-daytime")
        within_range_flag = False
    return within_range_flag


def model_evaluate(model, dataset_file, scaler):
    evaluate_df = pd.read_csv(dataset_file)

    # --- Check dataset not null
    if len(evaluate_df) == 0:
        print(f"= > specified test dataset file: {dataset_file} no data \n>")
        sys.exit()
    else:
        # --- Get
        feature_matrix = evaluate_df.iloc[:,3:-1].values
        scaled_feature_matrix = scaler.transform(feature_matrix)
        target_values = evaluate_df.loc[:,"label"].values

        # --- Prediction
        prediction_values = model.predict(scaled_feature_matrix)
        prediction_binary_values = (prediction_values >= 0.5).astype(int)

        # --- evaluate
        accuracy = accuracy_score(target_values, prediction_binary_values)
        precision = precision_score(target_values, prediction_binary_values)
        recall = recall_score(target_values, prediction_binary_values)
        f1 = f1_score(target_values, prediction_binary_values)

        return [accuracy,precision,recall,f1]


# ----- Model evaluate
def main(model, datasets_folder_path, scalar, beginning_daytime, end_daytime, results_list):

    # --- Calling
    for dataset_file in os.listdir(datasets_folder_path):
        if not is_dataset_within_range(dataset_file,beginning_daytime,end_daytime):
            break
        dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
        results_array = model_evaluate(model, dataset_file_path, scalar)
        results_list = np.vstack([results_list, results_array])

    return results_list