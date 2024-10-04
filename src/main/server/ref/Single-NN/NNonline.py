import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    #log amount
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    #gpu mem limit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #cpu : -1

import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
import fireducks.pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import glob
import pickle
import sys
import time
import datetime

from nnModelDefiner import createModel



EPOCHS = 30
BATCH_SIZE = 10
DATE = "202208"
DAYS = 3
N_TRAINING = 24
#------------------------------------------------------------------------------------
# [1]:train dir name [2]:test dir name
args = sys.argv
dataset_csv_dir_name = os.path.basename(args[1])
evaluate_csv_dir_name = str(args[2]).replace("/","")
#------ option -------------
if len(args) >= 4:
    if str(args[3]) == "-e":
        EPOCHS = int(args[4])
    if str(args[3]) == "-b":
        BATCH_SIZE = int(args[4])
if len(args) >= 6:
    if str(args[5]) == "-e":
        EPOCHS = int(args[6])
    if str(args[5]) == "-b":
        BATCH_SIZE = int(args[6])
#------------------------------------------------------------------------------------
str_dt_now = str(datetime.datetime.now()).split(" ")[0].replace("-","")+str(datetime.datetime.now()).split(" ")[1].split(".")[0].replace(":","")

weights_dir_name = str_dt_now + "-Single-NN-" + dataset_csv_dir_name + "-" + evaluate_csv_dir_name + "-model_weights"
if not os.path.isdir(weights_dir_name):
    os.mkdir(weights_dir_name)

results_dir_name = str_dt_now + "-Single-NN-" + dataset_csv_dir_name + "-" + evaluate_csv_dir_name + "-results"
if not os.path.isdir(results_dir_name):
    os.mkdir(results_dir_name)
#------------------------------------------------------------------------------------
before_csv_file_name = ""
before_evaluate_csv_file_name = ""
latest_date = None
tmp_accuracy = -1
tmp_f1 = -1
break_flag = False
scaler = StandardScaler()
first_training_flag = False
#------------------------------------------------------------------------------------



def train(model, csv_file_name):
    global latest_date
    global first_training_flag
    
    df = pd.read_csv(csv_file_name)
    print("TRAINING:" + csv_file_name + ":found")

    df_x = df.iloc[:,3:-1].values
    if first_training_flag == False:  
        df_x = scaler.fit_transform(df_x)
        first_training_flag = True
    else:
        df_x = scaler.transform(df_x)
    df_y = df.loc[:,"label"].values
    x_train = df_x
    y_train = df_y
    #---------------------------------------------------------------------------------
    if latest_date != None: # online training
        with open(weights_dir_name + "/" + latest_date +"-model_weights.pickle", 'rb') as f:
            print("PARAM:" + latest_date + "-model_weights.pickle:found")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)

    train_start_time = time.time()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    writeResults("training-time", train_time)

    latest_date = csv_file_name.split("/")[1].split(".")[0]
    with open(weights_dir_name +  "/" + latest_date + "-model_weights.pickle", 'wb') as f:
        print("PARAM:" + latest_date + "-model_weights.pickle:saved")
        pickle.dump(model.get_weights(), f)
    #---------------------------------------------------------------------------------
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



def evaluate(model, evaluate_csv_file_name):
    global tmp_accuracy
    global tmp_f1
    global break_flag

    csv_file_name = evaluate_csv_file_name
    if csv_file_name in glob.glob(evaluate_csv_dir_name + "/*.csv"):
        print("TEST:" + csv_file_name + ":found")
        evaluate_df = pd.read_csv(csv_file_name)

        if len(evaluate_df) != 0:
            x_test = evaluate_df.iloc[:,3:-1].values
            x_test = scaler.transform(x_test)
            y_test = evaluate_df.loc[:,"label"].values

        else:   # no data
            print("TEST:" + evaluate_csv_file_name + ":no data")
            print("TEST:" + before_evaluate_csv_file_name + ":used")
            evaluate_df = pd.read_csv(before_evaluate_csv_file_name)

            x_test = evaluate_df.iloc[:,3:-1].values
            x_test = scaler.transform(x_test)
            y_test = evaluate_df.loc[:,"label"].values            

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    #----------------------------------------------------------------
    y_pred_origin = model.predict(x_test, batch_size=BATCH_SIZE)

    y_pred_list = []
    for y in y_pred_origin:
        y_pred_list.append(round(float(y)))

    pred_benign_count = 0
    pred_malisious_count = 0
    for y in y_pred_list:
        if y == 0:  # benign
            pred_benign_count += 1
        elif y == 1:    # malisious
            pred_malisious_count += 1
    print("----------------------------------------------------------")        
    print("PREDICT: benign:" + str(pred_benign_count) + " malisious:" + str(pred_malisious_count))

    y_test_list = []
    true_benign_count = 0
    true_malicious_count = 0
    for y in y_test:
        y_test_list.append(int(y))
        if y == 0:
            true_benign_count += 1
        elif y == 1:
            true_malicious_count += 1
    print("ANSWER : benign:" + str(true_benign_count) + " malisious:" + str(true_malicious_count))
    #---------------------------------------------------------------

    precision = precision_score(y_test_list, y_pred_list)
    recall = recall_score(y_test_list, y_pred_list)
    f1 = f1_score(y_test_list, y_pred_list)

    print("F1 SCORE:", f1)

    writeResults("precision", precision)
    writeResults("recall", recall)
    writeResults("f1", f1)



def writeResults(word, result):
    f = open(results_dir_name + "/" + "NNonline_" + dataset_csv_dir_name.split("_")[0] + "_" + evaluate_csv_dir_name.split("_")[0] + "_" + word + ".txt", "a")
    f.write(str(result) + "\n")
    f.close()


def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn


def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn


if __name__ == "__main__":  #main

    day = 3
    hour = 16
    for i in range(DAYS * N_TRAINING):
        if break_flag == True:
            break
        if day < 10:
            str_day = "0" + str(day)
        else:
            str_day = str(day)
        if hour < 10:
            str_hour = "0" + str(hour)
        else:
            str_hour = str(hour)
        #-----------------------------------------------------------------------------------
        csv_file_name = dataset_csv_dir_name + "/" + DATE + str_day + str_hour + ".csv"
        if csv_file_name in glob.glob(dataset_csv_dir_name + "/*.csv"):
            model = createModel()
            model = train(model, csv_file_name)
            setBeforeDatasetCsvFile(csv_file_name)

        else:   # not found
            print("TRAINING:" + csv_file_name + ":not found")
            model = createModel()
            model = train(model, before_csv_file_name)
        #------------------------------------------------------------------------------------

        eva_hour = hour + 1
        if eva_hour < 10:
            eva_str_hour = "0" + str(eva_hour)
            eva_str_day = str_day
        elif eva_hour == 24:
            eva_str_hour = "00"
            eva_day = day + 1
            if eva_day < 10:
                eva_str_day = "0" + str(eva_day)
            else:
                eva_str_day = str(eva_day)
        else:
            eva_str_hour = str(eva_hour)
            eva_str_day = str_day

        csv_file_name = evaluate_csv_dir_name + "/" + DATE + eva_str_day + eva_str_hour + ".csv"
        if csv_file_name in glob.glob(evaluate_csv_dir_name + "/*.csv"):
            evaluate(model, csv_file_name)
            setBeforeEvaluateCsvFile(csv_file_name)

        else:   # not found
            print("TEST:" + csv_file_name + ":not found")
            evaluate(model, before_evaluate_csv_file_name)

        print("")
        hour += 1
        if hour == 24:
            hour = 0
            day += 1