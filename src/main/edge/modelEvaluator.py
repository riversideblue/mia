import json

import pandas as pd


def set_results_frame():
    results_frame = pd.DataFrame(columns=['training_count','training_time','benign_count','malicious_count'])
    return results_frame

def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn


def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn


# ----- Model evaluate
def main(model, init_time,settings):

    # --- Setup results dataframe
    results = set_results_frame()

    # --- Field
    trained_model = model
    test_datasets_folder_path = settings['Datasets']['DIR_PATH']
    print("hello")