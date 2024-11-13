import glob
import os
import time
import pickle

def is_previous_model_exist(dir_name, file_format):
    file_list = sorted(glob.glob(os.path.join(dir_name, file_format)))
    previous_weight_file = None
    if file_list:
        previous_weight_file = file_list[-1]
    return previous_weight_file

def main(model, df, output_dir_path, scalar, epochs, batch_size, repeat_count, retraining_daytime):

    previous_weights_file = is_previous_model_exist(f"{output_dir_path}/model_weights",
                                                   "*-weights.pickle")

    features = df.iloc[:,3:-1].astype(float)
    targets = df.iloc[:,-1].astype(float)

    # --- load previous model weights file if exist
    if previous_weights_file is not None:
        features = scalar.transform(features)
        with open(previous_weights_file, 'rb') as f:
            print(f"previous -weights.pickle file: {previous_weights_file} found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        features = scalar.fit_transform(features)
        print("previous -weights.pickle file: not found")
        print("initialize model weights ... ")

    # --- execute model training
    print(f"execute model_training ...")
    training_time_list = []
    for i in range(repeat_count):
        train_start_time = time.time()
        model.fit(features, targets, epochs=epochs, batch_size=batch_size)
        train_end_time = time.time()
        training_time_list.append(train_end_time - train_start_time)
    training_time = sum(training_time_list) / len(training_time_list)

    with open(f"{output_dir_path}/model_weights/{retraining_daytime}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f) # type: ignore

    # --- count benign and malicious
    benign_count = 0
    malicious_count = 0
    for target in targets:
        if int(target) == 0:
            benign_count += 1
        elif int(target) == 1:
            malicious_count += 1

    # count training data number

    flow_num = df.shape[0]

    return model,[retraining_daytime, flow_num, training_time, benign_count, malicious_count]
