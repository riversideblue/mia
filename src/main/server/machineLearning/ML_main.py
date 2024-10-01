import os
import sys
import logging
import json
from datetime import datetime

import pandas as pd

# ----- Path Check
def pass_check(path):

    if not os.path.exists(path): # error >> argument 1 name
        logging.error(f"cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)

# ----- Output mkdir
def output_mkdir():
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

    return model

# ----- Model evaluate
def model_eval(model,test_dataset):
    df = pd.read_csv(test_dataset)
    return model

# ----- Main
if __name__ == "__main__":

    # --- logging setting
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s >> %(message)s')

    # --- load setting.json
    settings = json.load(open("settings.json", "r"))

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']    #log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']    #gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']    #cpu : -1

    pass_check(settings['DatasetFolderPath']['TRAINING'])
    pass_check(settings['DatasetFolderPath']['TEST'])
    output_mkdir()

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