import os
import sys
import logging
import json
from datetime import datetime

import pandas as pd

# ----- load setting.json

settings = json.load(open("mT_settings.json", "r"))

# ----- logging setting

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s >> %(message)s')

# ----- Path Check

if not os.path.exists(settings['DatasetPath']['training']): # error >> argument 1 name
    logging.error(f"cannot find training dataset: {os.path.basename(settings['DatasetPath']['training'])}")
    sys.exit(1)
elif not os.path.exists(settings['DatasetPath']['test']): # error >> argument 2 name
    logging.error(f"cannot find test dataset: {os.path.basename(settings['DatasetPath']['test'])}")
    sys.exit(1)
else: # success
    tr_path = settings['DatasetPath']['training']
    tr_name = os.path.basename(tr_path)
    te_path = settings['DatasetPath']['test']
    te_name = os.path.basename(te_path)

# ----- OS environment settings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']    #log amount
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']    #gpu mem limit
os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']    #cpu : -1

# # ログでパスを確認
# logging.info(f"Training dataset path: {tr_path}")
# logging.info(f"Test dataset path: {te_path}")
# logging.info(f"{os.path.basename(settings['DatasetPath']['training'])}")

# ------ Output mkdir

string_datetime_now =  datetime.now().strftime("%Y%m%d%H%M%S%f")[:14]
dir_name = f"outputs/{string_datetime_now}"
os.makedirs(dir_name)
os.makedirs(f"{dir_name}/weight")
os.makedirs(f"{dir_name}/result")

# ----- Model training script

def model_training(model,training_dataset):
    df = pd.read_csv(training_dataset)
    print("TRAINING:" + training_dataset + ":found")
    return model

# ----- Model evaluate script

def model_eval(model,test_dataset):
    df = pd.read_csv(test_dataset)
    return model

# ----- Main script