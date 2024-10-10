import glob
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

def is_dataset_out_of_range(dataset_file_name,beginning_daytime,end_daytime):
    within_range_flag: bool = False
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print(f"evaluate: {dataset_captured_datetime} dataset ignored by Evaluate/BEGINNING_DAYTIME setting")
        within_range_flag = True
    elif not dataset_captured_datetime <= end_daytime:
        print(f"evaluate: {dataset_captured_datetime} dataset ignored by Evaluate/TargetRange settings")
        within_range_flag = True
    return within_range_flag

def is_previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file

def save_results(evaluate_list, evaluate, output_dir):
    evaluate = pd.concat([evaluate, pd.DataFrame(evaluate_list, columns=evaluate.columns)])
    evaluate.to_csv(f"{output_dir}/results_evaluate.csv",index=False)

def save_settings_log(settings_log, output_dir):

    with open(f"{output_dir}/settings_log.json", "w") as f:
        json.dump(settings_log, f, indent=1) # type: ignore

def model_evaluate(model, dataset_file, scaler):

    previous_weights_file = is_previous_file_exist(f"{dataset_file}/model_weights",
                                                   "*-weights.pickle")

    evaluate_df = pd.read_csv(dataset_file)
    feature_matrix = evaluate_df.iloc[:,3:-1].values
    target_values = evaluate_df.loc[:,"label"].values

    # --- Check dataset not null
    if len(evaluate_df) == 0:
        print(f"specified test dataset file: {dataset_file} no data \n>")
        sys.exit()
    else:    # --- load previous model weights file if exist
        if previous_weights_file is not None:
            scaled_feature_matrix = scaler.transform(feature_matrix)
        else:
            scaled_feature_matrix = scaler.fit_transform(feature_matrix)
            print("previous -weights.pickle file: not found")
            print("initialize model weights ... ")


        # --- Prediction
        prediction_values = model.predict(scaled_feature_matrix)
        prediction_binary_values = (prediction_values >= 0.5).astype(int)

        # --- Evaluate
        accuracy = accuracy_score(target_values, prediction_binary_values)
        precision = precision_score(target_values, prediction_binary_values)
        recall = recall_score(target_values, prediction_binary_values)
        f1 = f1_score(target_values, prediction_binary_values)


        # --- Confusion matrix
        matrix = confusion_matrix(target_values, prediction_binary_values, labels=[0,1])

        return [accuracy,precision,recall,f1], matrix


# ----- Model other
def main():

    # --- Load settings
    settings = json.load(open('src/main/other/settings.json', 'r'))
    settings['Log'] = {}

    # --- Set results dataframe
    results = pd.DataFrame(columns=['accuracy','f1_score','precision','recall'])
    results_list = results.values

    # --- Field
    settings['Log']['Evaluate'] = {}
    model_path = settings['Evaluate']['MODEL_PATH']
    training_output_path:str = settings['Evaluate']['TRAINING_OUTPUT_PATH']
    datasets_folder_path:str = settings['Evaluate']['DATASETS_FOLDER_PATH']
    beginning_daytime = datetime.strptime(settings['Evaluate']['BEGINNING_DAYTIME'],"%Y-%m-%d %H:%M:%S")
    settings['Log']['Evaluate']['BEGINNING_DAYTIME'] = beginning_daytime.isoformat()
    days:int = settings['Evaluate']['TargetRange']['DAYS']
    hours:int = settings['Evaluate']['TargetRange']['HOURS']
    minutes:int = settings['Evaluate']['TargetRange']['MINUTES']
    seconds:int = settings['Evaluate']['TargetRange']['SECONDS']
    end_daytime:datetime = beginning_daytime + timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    settings['Log']['Evaluate']['END_DAYTIME'] = end_daytime.isoformat()

    scaler = StandardScaler()
    is_pass_exist(training_output_path)
    is_pass_exist(datasets_folder_path)
    results_matrix = [[0,0],[0,0]]

    # --- Create foundation model
    model = modelCreator.main()
    with open(model_path, 'rb') as f:
        print(f"previous -weights.pickle file: {model_path} found")
        init_weights = pickle.load(f)
        model.set_weights(init_weights)

    # --- Calling
    for dataset_file in os.listdir(datasets_folder_path):
        if not is_dataset_out_of_range(dataset_file,beginning_daytime,end_daytime):
            dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
            results_array,matrix = model_evaluate(model, dataset_file_path, scaler)
            results_list = np.vstack([results_list, results_array])
            results_matrix = matrix + results_matrix

    save_results(
        evaluate_list=results_list,
        evaluate=results,
        output_dir=training_output_path
    )
    framed_matrix = pd.DataFrame(
        data=results,
        index=["PredictionPositive","PredictionNegative"],
        columns=["TargetPositive", "TargetNegative"]
    )
    print(framed_matrix)

    # --- Save settings_log and results
    save_settings_log(settings,training_output_path)


if __name__ == "__main__":
    main()