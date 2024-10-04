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

    if not os.path.exists(path): # error >> argument 1 name
        print(f"cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)

def output_mkdir(current_time):
    dir_name = f"outputs/{current_time}_executed"
    os.makedirs(dir_name)
    os.makedirs(f"{dir_name}/weights")
    os.makedirs(f"{dir_name}/results")

def writeResults(current_time, training_dataset_path, test_dataset_path, word, result):
    results_dir_name = f"outputs/{current_time}_executed/results"
    f = open(f"{results_dir_name}/{os.path.basename(training_dataset_path)}_{os.path.basename(test_dataset_path)}_{word}.txt","a")
    f.write(str(result) + "\n")
    f.close()

def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn

def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn

def model_training(model, current_time, training_dataset_path, test_dataset_path,epochs,batch_size):

    training_file_date = None
    first_training_flag = False

    print("TRAINING:" + training_dataset_path)

    # --- csv processing
    df = pd.read_csv(training_dataset_path)
    df_x_label = df.iloc[:, 3:-1].values
    if not first_training_flag:
        df_x_label = StandardScaler().fit_transform(df_x_label)
        first_training_flag = True
    else:
        df_x_label = StandardScaler().transform(df_x_label)
    df_y_label = df.loc[:, "label"].values
    x_train = df_x_label
    y_train = df_y_label

    # --- pickle
    if training_file_date is not None:  # online training
        with open(f"outputs/{current_time}_executed/weights/{training_file_date}-weights.pickle", 'rb') as f:
            print(f"PARAM: {training_file_date}-weights.pickle:found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)

    # --- execute model training
    print("Start Model Training ...")
    train_start_time = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f"Training Time {train_time}s >>> DONE")
    writeResults(current_time,training_dataset_path,test_dataset_path,"training-time", train_time)

    training_file_date = os.path.basename(training_dataset_path).split(".")[0]
    print("training_dataset_path:" + training_dataset_path)
    print("training_date:" +training_file_date)
    print(f"outputs/{current_time}_executed/weights/{training_file_date}-weights.pickle")
    with open(f"outputs/{current_time}_executed/weights/{training_file_date}-weights.pickle", 'wb') as f:
        print("PARAM:" + training_file_date + "-weights.pickle:saved")
        pickle.dump(model.get_weights(), f)
    # ---------------------------------------------------------------------------------
    train_benign_count = 0
    train_malicious_count = 0
    for y in y_train:
        if int(y) == 0:
            train_benign_count += 1
        elif int(y) == 1:
            train_malicious_count += 1
    print("TRAINING:ben " + str(train_benign_count) + " records")
    print("TRAINING:mal " + str(train_malicious_count) + " records")
    writeResults(current_time,training_dataset_path,test_dataset_path,"benign-records", train_benign_count)
    writeResults(current_time,training_dataset_path,test_dataset_path,"malicious-records", train_malicious_count)

    return model

# ----- Model evaluate
def model_eval(model,test_dataset):
    df = pd.read_csv(test_dataset)
    return model

def main(model):

    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    current_time = datetime.now(jst).strftime("%Y%m%d%H%M%S%f")[:14]

    # --- load setting.json
    settings = json.load(open("settings.json", "r"))

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1
    pass_check(settings['DatasetsFolderPath']['TRAINING'])
    pass_check(settings['DatasetsFolderPath']['TEST'])

    # --- Field

    foundation_model = model
    training_datasets_path = settings['DatasetsFolderPath']['TRAINING']
    test_datasets_path = settings['DatasetsFolderPath']['TEST']
    epochs = settings['LearningDefine']['EPOCHS']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    beginning_date = settings['LearningDefine']['BEGINNING_DATE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    days = settings['LearningDefine']['Range']['DAYS']
    hours = settings['LearningDefine']['Range']['HOURS']

    # --- Get each csv file in training dataset folder and start training model

    output_mkdir(current_time)
    for training_dataset in os.listdir(training_datasets_path):
        training_dataset_path = os.path.join(training_datasets_path, training_dataset)
        model_training(foundation_model,
                       current_time,
                       training_dataset_path,
                       test_datasets_path,
                       epochs,
                       batch_size
                       )

    # --- Evaluate and tuning model