import glob
import os
from datetime import datetime, timedelta
import time

import numpy as np

import pandas as pd
import pickle

def is_previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file

def save_results(training_list, training, output_dir):
    training = pd.concat([training, pd.DataFrame(training_list, columns=training.columns)])
    training.to_csv(f"{output_dir}/results_training.csv",index=False)


def model_training(model, output_dir_path, dataset_file_path, beginning_daytime, interval, target_end_daytime, scaler, epochs, batch_size, repeat_count):

    training_file_date = os.path.basename(dataset_file_path).split(".")[0]
    previous_weights_file = is_previous_file_exist(f"{output_dir_path}/model_weights",
                                                "*-weights.pickle")


    # --- csv processing
    df = pd.read_csv(dataset_file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    end_time = beginning_daytime + timedelta(seconds=interval)
    last_timestamp = df['timestamp'].iloc[-1] # データセットの最後の要素のtimestamp
    last_training_flag = False
    if end_time > last_timestamp:
        end_time = last_timestamp
        last_training_flag = True
    elif end_time > target_end_daytime:
        end_time = target_end_daytime
        last_training_flag = True
    explanatory_values = df[(df['timestamp'] >= beginning_daytime) & (df['timestamp'] < end_time)].iloc[:, 1:-1]
    target_values = df[(df['timestamp'] >= beginning_daytime) & (df['timestamp'] < end_time)].loc[:, "label"].values

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
    training_time_list = []
    for i in range(repeat_count):
        train_start_time = time.time()
        model.fit(explanatory_values, target_values, epochs=epochs, batch_size=batch_size)
        train_end_time = time.time()
        training_time_list.append(train_end_time - train_start_time)
    training_time = sum(training_time_list) / len(training_time_list)

    with open(f"{output_dir_path}/model_weights/{end_time}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f) # type: ignore

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for target_value in target_values:
        if int(target_value) == 0:
            benign_count += 1
        elif int(target_value) == 1:
            malicious_count += 1

    return model,end_time,[end_time, training_time, benign_count, malicious_count],last_training_flag

def main(model, output_dir_path, datasets_folder_path, scalar, beginning_daytime, target_range, interval, repeat_count, epochs, batch_size, results, results_list):

    # Setup
    target_end_daytime = beginning_daytime + target_range

    # --- Get each csv file in training dataset folder and calling model_training
    for dataset_file in os.listdir(datasets_folder_path):
        start = time.time()
        last_training_flag = False
        while True:
            print("XXXX")
            for i in range(repeat_count):
                dataset_file_path:str = f"{datasets_folder_path}/{dataset_file}"
                model,training_end_daytime,results_array,last_training_flag = model_training(
                    model=model,
                    output_dir_path=output_dir_path,
                    dataset_file_path=dataset_file_path,
                    beginning_daytime=beginning_daytime,
                    interval=interval,
                    target_end_daytime=target_end_daytime,
                    scaler=scalar,
                    epochs=epochs,
                    batch_size=batch_size,
                    repeat_count=repeat_count
                )
                results_list = np.vstack([results_list, results_array])
                beginning_daytime = training_end_daytime
            if last_training_flag:
                print("last training detected")
                end = time.time()
                print(f"----- reload done: {str(end-start)} ----- ")
                break

    save_results(
        training_list=results_list,
        training=results,
        output_dir=output_dir_path
    )