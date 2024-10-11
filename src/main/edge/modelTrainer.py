import glob
import os
from datetime import datetime
import time

import numpy as np

import pandas as pd
import pickle

def is_dataset_out_of_range(dataset_file_name,beginning_daytime,end_daytime):
    within_range_flag:bool = False
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print(f"training:  {dataset_captured_datetime} dataset ignored by Training/BEGINNING_DAYTIME setting")
        within_range_flag = True
    elif not dataset_captured_datetime <= end_daytime:
        print(f"training:  {dataset_captured_datetime} dataset ignored by Training/TargetRange setting")
        within_range_flag = True
    return within_range_flag

def is_previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file

def save_results(training_list, training, output_dir):
    training = pd.concat([training, pd.DataFrame(training_list, columns=training.columns)])
    training.to_csv(f"{output_dir}/results_training.csv",index=False)


def model_training(model, output_dir_path, dataset_file_path, scaler, epochs, batch_size, training_count, results_list):

    training_file_date = os.path.basename(dataset_file_path).split(".")[0]
    previous_weights_file = is_previous_file_exist(f"{output_dir_path}/model_weights",
                                                "*-weights.pickle")

    print(f"training_dataset_path: {dataset_file_path}")

    # --- csv processing
    df = pd.read_csv(dataset_file_path)
    explanatory_values = df.iloc[:, 3:-1].values
    target_values = df.loc[:, "label"].values

    # --- load previous model weights file if exist
    if previous_weights_file is not None:
        explanatory_values = scaler.transform(explanatory_values)
        with open(previous_weights_file, 'rb') as f:
            print(f"previous -weights.pickle file: {previous_weights_file} found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        explanatory_values = scaler.fit_transform(explanatory_values)
        print("previous -weights.pickle file: not found")
        print("initialize model weights ... ")

    # --- execute model training
    print(f"----- execute model_training ----- ")
    train_start_time = time.time()
    model.fit(explanatory_values, target_values, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    with open(f"{output_dir_path}/model_weights/{training_file_date}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f) # type: ignore

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for target_value in target_values:
        if int(target_value) == 0:
            benign_count += 1
        elif int(target_value) == 1:
            malicious_count += 1

    return model,[training_count, training_time, benign_count, malicious_count]

def main(model, output_dir_path, datasets_folder_path, scalar, beginning_daytime, end_daytime, repeat_count, epochs, batch_size, results, results_list):

    # Setup
    training_count:int = 0

    # --- Get each csv file in training dataset folder and calling model_training
    for dataset_file in os.listdir(datasets_folder_path):
        start = time.time()
        if not is_dataset_out_of_range(dataset_file, beginning_daytime, end_daytime):
            for i in range(repeat_count):
                training_count += 1
                dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
                model,results_array = model_training(
                    model=model,
                    output_dir_path=output_dir_path,
                    dataset_file_path=dataset_file_path,
                    scaler=scalar,
                    epochs=epochs,
                    batch_size=batch_size,
                    training_count=training_count,
                    results_list=results_list
                )
                results_list = np.vstack([results_list, results_array])
                end = time.time()
                print(f"----- done: {str(end-start)} ----- ")

    save_results(
        training_list=results_list,
        training=results,
        output_dir=output_dir_path
    )