import glob
import os
import time
import pickle

import numpy as np


def is_previous_model_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file

def main(model, features, targets, output_dir_path, epochs, batch_size, retraining_daytime):

    # --- execute model training
    print(f"execute model_training ...")
    training_time_list = []

    train_start_time = time.time()
    history = model.fit(features, targets, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    accuracy = history.history["accuracy"][-1]
    loss = history.history["loss"][-1]

    training_time_list.append(train_end_time - train_start_time)
    training_time = sum(training_time_list) / len(training_time_list)

    str_retraining_daytime = retraining_daytime.strftime("%Y-%m-%dT%H:%M:%S")
    with open(f"{output_dir_path}/model_weights/{str_retraining_daytime}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f) # type: ignore

    # --- count benign and malicious
    benign_count = np.sum(targets == 0)
    malicious_count = np.sum(targets == 1)

    # count training data number

    flow_num = targets.shape[0]

    return model,[retraining_daytime, accuracy, loss, training_time, benign_count, malicious_count, flow_num]
