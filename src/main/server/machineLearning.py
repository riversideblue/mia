import glob
import os
import sys
import json
from datetime import datetime
import time
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
    with open(f"outputs/{initiation_time}_executed/results.json","w") as f:
        json.dump({
            "training_time": [],
            "benign_records": [],
            "malicious_records": []
        },f,indent=1)
    with open(f'{dir_name}/settings_log.json','w') as f:
        json.dump(settings,f,indent=1)

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

def model_training(model, initiation_time, training_dataset_file_path, scaler, epochs, batch_size, results):

    training_file_date = os.path.basename(training_dataset_file_path).split(".")[0]
    previous_weights_file = previous_file_exist(f"outputs/{initiation_time}_executed/model_weights",
                                                "*-weights.pickle")

    print("\n -------------------------called model_training----------------------------- \n")
    print("-initiation_time: " + initiation_time)
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
    print(f"\n-done: training time {training_time}s")
    results["training_time"].append(training_time)

    with open(f"outputs/{initiation_time}_executed/model_weights/{training_file_date}-weights.pickle", 'wb') as f:
        print(
            "-output path: " + f"outputs/{initiation_time}_executed/model_weights/{training_file_date}-weights.pickle")
        pickle.dump(model.get_weights(), f)

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for y in y_train:
        if int(y) == 0:
            benign_count += 1
        elif int(y) == 1:
            malicious_count += 1
    print("-training: benign " + str(benign_count) + " records")
    print("-training: malicious " + str(malicious_count) + " records")

    results["benign_records"].append(benign_count)
    results["malicious_records"].append(malicious_count)
    print(results)
    print(type(results))

    with open(f"outputs/{initiation_time}_executed/results.json","w") as f:
        json.dump(results, f, separators=(',', ':'))

    return model


# ----- Model evaluate
def model_eval(model, initiation_time, test_dataset):
    df = pd.read_csv(test_dataset)
    return model


def main(model):
    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    initiation_time = datetime.now(jst).strftime("%Y%m%d%H%M%S%f")[:14]

    # --- load setting.json and results.json
    settings = json.load(open("settings.json", "r"))
    output_mkdir(initiation_time, settings)
    results = json.load(open(f"outputs/{initiation_time}_executed/results.json"))

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1
    pass_check(settings['DatasetsFolderPath']['TRAINING'])
    pass_check(settings['DatasetsFolderPath']['TEST'])

    # --- Field

    foundation_model = model
    training_datasets_folder_path = settings['DatasetsFolderPath']['TRAINING']
    test_datasets_folder_path = settings['DatasetsFolderPath']['TEST']
    epochs = settings['LearningDefine']['EPOCHS']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    beginning_date = settings['LearningDefine']['BEGINNING_DATE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    days = settings['LearningDefine']['Range']['DAYS']
    hours = settings['LearningDefine']['Range']['HOURS']
    scaler = StandardScaler()


    # --- Get each csv file in training dataset folder and start training model

    for training_dataset_file in os.listdir(training_datasets_folder_path):
        training_dataset_file_path = os.path.join(training_datasets_folder_path, training_dataset_file)
        model_training(
            foundation_model,
            initiation_time,
            training_dataset_file_path,
            scaler,
            epochs,
            batch_size,
            results
        )

    # --- Evaluate and tuning model

    for test_dataset_file in os.listdir(test_datasets_folder_path):
        test_dataset_file_path = os.path.join(test_datasets_folder_path, test_dataset_file)
        model_eval(
            foundation_model,
            initiation_time,
            test_dataset_file_path
        )
