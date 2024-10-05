import glob
import os
import sys
import json
from datetime import datetime, timedelta
import time

import numpy as np
import pytz

import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


def pass_check(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)


def output_mkdir(initiation_time, settings):
    dir_name = f"outputs/{initiation_time}_executed"
    os.makedirs(dir_name)
    os.makedirs(f"{dir_name}/model_weights")
    with open(f'{dir_name}/settings_log.json','w') as f:
        json.dump(settings,f,indent=1)


def set_results_frame():
    results_frame = pd.DataFrame(columns=['training_time','benign_count','malicious_count'])
    return results_frame


def save_results(results_list, initiation_time, results):
    dir_name = f"outputs/{initiation_time}_executed"
    results.loc = results_list
    results.to_csv(f"{dir_name}/results.csv")


def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn


def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn


def previous_file_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file


def model_training(model, initiation_time, training_dataset_file_path, scaler, epochs, batch_size):

    training_file_date = os.path.basename(training_dataset_file_path).split(".")[0]
    previous_weights_file = previous_file_exist(f"outputs/{initiation_time}_executed/model_weights",
                                                "*-weights.pickle")

    print("-training_dataset_path: " + training_dataset_file_path)

    # --- csv processing
    df = pd.read_csv(training_dataset_file_path)
    df_x_label = df.iloc[:, 3:-1].values
    df_y_label = df.loc[:, "label"].values

    # --- load previous model weights file if exist
    if previous_weights_file is not None:
        df_x_label = scaler.transform(df_x_label)
        with open(previous_weights_file, 'rb') as f:
            print(f"-previous -weights.pickle file: {previous_weights_file} found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        df_x_label = scaler.fit_transform(df_x_label)
        print("-previous -weights.pickle file: not found")
        print("--initialize model weights ... ")

    x_train = df_x_label
    y_train = df_y_label

    # --- execute model training
    print("\n >>>>> execute model training ... <<<<< \n")
    train_start_time = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    with open(f"outputs/{initiation_time}_executed/model_weights/{training_file_date}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f)

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for y in y_train:
        if int(y) == 0:
            benign_count += 1
        elif int(y) == 1:
            malicious_count += 1

    return [training_time,benign_count,malicious_count]


# ----- Model evaluate
def model_eval(model, initiation_time, test_dataset):
    df = pd.read_csv(test_dataset)
    return model


def main(model):

    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    initiation_time = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- prepare settings and results
    settings = json.load(open("settings.json", "r"))
    output_mkdir(initiation_time, settings)
    results = set_results_frame()

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1
    pass_check(settings['Datasets']['DIR_PATH'])

    # --- Field
    foundation_model = model
    training_datasets_folder_path = settings['Datasets']['DIR_PATH']
    beginning_time = datetime.strptime(settings['Datasets']['BEGINNING_TIME'],"%Y%m%d%H%M%S")
    days = settings['Datasets']['LearningRange']['DAYS']
    hours = settings['Datasets']['LearningRange']['HOURS']
    minutes = settings['Datasets']['LearningRange']['MINUTES']
    seconds = settings['Datasets']['LearningRange']['SECONDS']
    epochs = settings['LearningDefine']['EPOCHS']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    end_time = beginning_time + timedelta(days=days,hours=hours,minutes=minutes,seconds=seconds)

    scaler = StandardScaler()
    results_list = results.values

    # --- Get each csv file in training dataset folder and start training model
    for training_dataset_file in os.listdir(training_datasets_folder_path):
        start = time.time()
        print("\n -------------------------calling model_training----------------------------- \n")
        training_dataset_file_path = os.path.join(training_datasets_folder_path, training_dataset_file)
        results_list = np.vstack([
            results_list,
            model_training(
                foundation_model,
                initiation_time,
                training_dataset_file_path,
                scaler,
                epochs,
                batch_size
            )
        ])
        end = time.time()
        print("\n -------------------------done: "+str(end-start)+"----------------------------- ")

    # --- Evaluate and tuning model


    # --- save settings and results

    with open("settings.json", "w") as f:
        settings['Log']['INITIATION_TIME'] = initiation_time
        settings['Log']['BEGINNING_TIME'] = beginning_time.isoformat()
        settings['Log']['END_TIME'] = end_time.isoformat()
        json.dump(settings, f, indent=1)
    save_results(results_list, initiation_time, results)