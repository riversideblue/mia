import os
import sys

import pandas as pd


def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn


def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn

def model_evaluate(model, init_time, dataset_file, scaler, results_list):
    evaluate_df = pd.read_csv(dataset_file)
    if len(evaluate_df) != 0:
        x_test = evaluate_df.iloc[:,3:-1].values
        x_test = scaler.transform(x_test)
        y_test = evaluate_df.loc[:,"label"].values
    else:
        print(f"= > specified test dataset file: {dataset_file} no data \n>")
        sys.exit()

    print(x_test)
    print(y_test)


# ----- Model evaluate
def main(model, init_time, datasets_folder_path, scalar, beginning_daytime, end_daytime, results_list):

    # --- Calling
    for dataset_file in os.listdir(datasets_folder_path):
        dataset_file_path = os.path.join(datasets_folder_path, dataset_file)
        model_evaluate(model, init_time, dataset_file_path, scalar, results_list)

    return results_list