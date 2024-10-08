import glob
import os
from datetime import datetime
import time

import numpy as np

import pandas as pd
import pickle

def is_dataset_within_range(dataset_file_name,beginning_daytime,end_daytime):
    within_range_flag: bool = True
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print(f">\n=  target dataset captured time : {dataset_captured_datetime}"
              f">\n=  beginning-daytime : {beginning_daytime}"
              f">\n=  no dataset in specified beginning-daytime")
        within_range_flag = False
    elif not dataset_captured_datetime <= end_daytime:
        print(f">\n= target dataset captured time : {dataset_captured_datetime}"
              f">\n= end-daytime : {end_daytime}"
              f">\n= no dataset in specified end-daytime")
        within_range_flag = False
    return within_range_flag


def is_previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file


def model_training(model, init_time, dataset_file_path, scaler, epochs, batch_size, training_count, results_list):

    training_file_date = os.path.basename(dataset_file_path).split(".")[0]
    previous_weights_file = is_previous_file_exist(f"src/main/edge/outputs/{init_time}_executed/model_weights",
                                                "*-weights.pickle")

    print(f"= > training_dataset_path: {dataset_file_path} \n>")

    # --- csv processing
    df = pd.read_csv(dataset_file_path)
    df_x_label = df.iloc[:, 3:-1].values
    df_y_label = df.loc[:, "label"].values

    # --- load previous model weights file if exist
    if previous_weights_file is not None:
        df_x_label = scaler.transform(df_x_label)
        with open(previous_weights_file, 'rb') as f:
            print(f"= > previous -weights.pickle file: {previous_weights_file} found \n>")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        df_x_label = scaler.fit_transform(df_x_label)
        print("= > previous -weights.pickle file: not found \n>")
        print("= > initialize model weights ... \n>")

    x_train = df_x_label
    y_train = df_y_label

    # --- execute model training
    print("= > <<< execute model training ... >>> \n>\n")
    train_start_time = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    with open(f"src/main/edge/outputs/{init_time}_executed/model_weights/{training_file_date}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f) # type: ignore

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for y in y_train:
        if int(y) == 0:
            benign_count += 1
        elif int(y) == 1:
            malicious_count += 1

    results_list = np.vstack([
        results_list, [training_count, training_time, benign_count, malicious_count]
    ])

    return model,results_list

def main(model, init_time, datasets_folder_path, scalar, beginning_daytime, end_daytime, repeat_count, epochs, batch_size, results_list):

    # Setup
    training_count = 0

    # --- Get each csv file in training dataset folder and calling model_training
    for dataset_file in os.listdir(datasets_folder_path):
        start = time.time()
        if not is_dataset_within_range(dataset_file,beginning_daytime,end_daytime):
            break
        for i in range(repeat_count):
            print(">\n= > <<< calling model_training >>> \n>")
            training_count += 1
            dataset_file_path = os.path.join(datasets_folder_path, dataset_file)
            model,results_list = model_training(
                model=model,
                init_time=init_time,
                dataset_file_path=dataset_file_path,
                scaler=scalar,
                epochs=epochs,
                batch_size=batch_size,
                training_count=training_count,
                results_list=results_list
            )
            end = time.time()
            print(f"\n>\n= > <<< done: {str(end-start)} >>> ")

    return model,results_list