import os
import sys
import logging
import json
from datetime import datetime

import pandas as pd

# ----- logging setting

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s >> %(message)s')

# ----- load setting.json

settings = json.load(open("settings.json", "r"))

# ----- OS environment settings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']    #log amount
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']    #gpu mem limit
os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']    #cpu : -1

# ----- Path Check

if not os.path.exists(settings['DatasetFolderPath']['TRAINING']): # error >> argument 1 name
    logging.error(f"cannot find training dataset: {os.path.basename(settings['DatasetFolderPath']['TRAINING'])}")
    sys.exit(1)
elif not os.path.exists(settings['DatasetFolderPath']['TEST']): # error >> argument 2 name
    logging.error(f"cannot find test dataset: {os.path.basename(settings['DatasetFolderPath']['TEST'])}")
    sys.exit(1)

# ------ Output mkdir

string_datetime_now =  datetime.now().strftime("%Y%m%d%H%M%S%f")[:14]
dir_name = f"outputs/{string_datetime_now}"
os.makedirs(dir_name)
os.makedirs(f"{dir_name}/weight")
os.makedirs(f"{dir_name}/result")

# ----- Model training

def model_training(model, training_dataset):
    df = pd.read_csv(training_dataset)
    print("TRAINING:" + training_dataset)
    print(df)

    global latest_date
    global first_training_flag


    df_x = df.iloc[:, 3:-1].values
    if first_training_flag == False:
        df_x = scaler.fit_transform(df_x)
        first_training_flag = True
    else:
        df_x = scaler.transform(df_x)
    df_y = df.loc[:, "label"].values
    x_train = df_x
    y_train = df_y
    # ---------------------------------------------------------------------------------
    if latest_date != None:  # online training
        with open(weights_dir_name + "/" + latest_date + "-weights.pickle", 'rb') as f:
            print("PARAM:" + latest_date + "-weights.pickle:found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)

    train_start_time = time.time()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    writeResults("training-time", train_time)

    latest_date = csv_file_name.split("/")[1].split(".")[0]
    with open(weights_dir_name + "/" + latest_date + "-weights.pickle", 'wb') as f:
        print("PARAM:" + latest_date + "-weights.pickle:saved")
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
    writeResults("benign-records", train_benign_count)
    writeResults("malicious-records", train_malicious_count)

    return model

# ----- Model evaluate

def model_eval(model,test_dataset):
    df = pd.read_csv(test_dataset)
    return model

# ----- Main

if __name__ == "__main__":

    # --- Field

    foundation_model="a"
    training_dataset_path = settings['DatasetFolderPath']['TRAINING']
    test_dataset_path = settings['DatasetFolderPath']['TEST']
    epoch = settings['LearningDefine']['EPOCH']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    beginning_date = settings['LearningDefine']['BEGINNING_DATE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    days = settings['LearningDefine']['Range']['DAYS']
    hours = settings['LearningDefine']['Range']['HOURS']

    # --- Get each csv file in training dataset folder and training model

    for item in os.listdir(training_dataset_path):
        item_path = os.path.join(training_dataset_path, item)
        model_training(foundation_model,item_path)

    # --- Evaluate and tuning model

