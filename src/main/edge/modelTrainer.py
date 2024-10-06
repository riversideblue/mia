import glob
import os
import sys
import json
from datetime import datetime, timedelta
import time

import numpy as np

import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"= > cannot find training dataset: {os.path.basename(path)} \n>")
        sys.exit(1)


def is_dataset_within_range(dataset_file_name,beginning_daytime,end_daytime):
    flag = True
    dataset_captured_datetime = datetime.strptime(dataset_file_name.split(".")[0] + "0000", "%Y%m%d%H%M%S")
    if not beginning_daytime <= dataset_captured_datetime:
        print("= > daytime when target dataset captured is before beginning-daytime specified \n>")
        flag = False
    elif not dataset_captured_datetime <= end_daytime:
        print("= > daytime when target dataset captured is after end-daytime specified\n>")
        flag = False
    return flag


def set_results_frame():
    results_frame = pd.DataFrame(columns=['training_count','training_time','benign_count','malicious_count'])
    return results_frame


def save_results(results_list, init_time, results):
    dir_name = f"outputs/{init_time}_executed"
    results = pd.concat([results, pd.DataFrame(results_list, columns=results.columns)])
    results.to_csv(f"{dir_name}/results.csv",index=False)


def is_previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file


def model_training(model, init_time, training_dataset_file_path, scaler, epochs, batch_size, training_count, results_list):

    training_file_date = os.path.basename(training_dataset_file_path).split(".")[0]
    previous_weights_file = is_previous_file_exist(f"outputs/{init_time}_executed/model_weights",
                                                "*-weights.pickle")

    print(f"= > training_dataset_path: {training_dataset_file_path} \n>")

    # --- csv processing
    df = pd.read_csv(training_dataset_file_path)
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

    with open(f"outputs/{init_time}_executed/model_weights/{training_file_date}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f)

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

def main(foundation_model,init_time,settings):

    # Setup results dataframe
    results = set_results_frame()

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1
    is_pass_exist(settings['Datasets']['DIR_PATH'])

    # --- Field
    model = foundation_model
    training_datasets_folder_path = settings['Datasets']['DIR_PATH']
    beginning_daytime = datetime.strptime(settings['Datasets']['BEGINNING_DAYTIME'],"%Y%m%d%H%M%S")
    days = settings['Datasets']['LearningRange']['DAYS']
    hours = settings['Datasets']['LearningRange']['HOURS']
    minutes = settings['Datasets']['LearningRange']['MINUTES']
    seconds = settings['Datasets']['LearningRange']['SECONDS']
    epochs = settings['LearningDefine']['EPOCHS']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    end_daytime = beginning_daytime + timedelta(days=days,hours=hours,minutes=minutes,seconds=seconds)
    scaler = StandardScaler()
    results_list = results.values
    training_count = 0

    # --- Get each csv file in training dataset folder and calling model_training
    for training_dataset_file in os.listdir(training_datasets_folder_path):
        start = time.time()
        if not is_dataset_within_range(training_dataset_file,beginning_daytime,end_daytime):
            break
        for i in range(repeat_count):
            print(">\n= > <<< calling model_training >>> \n>")
            training_count += 1
            training_dataset_file_path = os.path.join(training_datasets_folder_path, training_dataset_file)
            model,results_list = model_training(
                model,
                init_time,
                training_dataset_file_path,
                scaler,
                epochs,
                batch_size,
                training_count,
                results_list
            )
            end = time.time()
            print(f"\n>\n= > <<< done: {str(end-start)} >>> ")

    # --- save settings and results
    settings['Log']['BEGINNING_DAYTIME'] = beginning_daytime.isoformat()
    settings['Log']['END_DAYTIME'] = end_daytime.isoformat()
    with open(f"outputs/{init_time}_executed/settings_log.json", "w") as f:
        json.dump(settings, f, indent=1)
    save_results(results_list, init_time, results)

    return model