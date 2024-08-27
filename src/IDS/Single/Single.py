import warnings
warnings.simplefilter('ignore')

import fireducks.pandas as pd
import numpy as np

#----------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#----------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import glob
import pickle
import sys
import time
import os
import datetime



DATE = "202201"
DAYS = 30
N_TRAINING = 24
#------------------------------------------------------------------------------------
# [1]:train dir name [2]:test dir name [3]:used algorithm
args = sys.argv
dataset_csv_dir_name = str(args[1]).replace("/","")
evaluate_csv_dir_name = str(args[2]).replace("/","")
algorithm = str(args[3])
#--------------------------------------------------
no_update_flag = False

N_NEIGHBORS = 5
N_ESTIMATORS = 100
MAX_DEPTH = 5

if len(args) > 4:
    if "-S" in args:
        S_index = args.index("-S")
        MODEL_INDEX = int(args[S_index + 1])
        no_update_flag = True
    if "-k" in args:
        k_index = args.index("-k")
        N_NEIGHBORS = int(args[k_index + 1])
    if "-e" in args:
        e_index = args.index("-e")
        N_ESTIMATORS = int(args[e_index + 1])
    if "-d" in args:
        d_index = args.index("-d")
        MAX_DEPTH = int(args[d_index + 1])

if algorithm == "KNN":
    MODEL = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
elif algorithm == "DT":
    MODEL = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=0)
elif algorithm == "RF":
    MODEL = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=0)
elif algorithm == "GBDT":
    MODEL = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=0)
else:
    pass
#---------------------------------------------------
str_dt_now = str(datetime.datetime.now()).split(" ")[0].replace("-","")+str(datetime.datetime.now()).split(" ")[1].split(".")[0].replace(":","")
results_dir_name = str_dt_now + "-Single-" + algorithm + "-" + dataset_csv_dir_name + "-" + evaluate_csv_dir_name + "-results"
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



def train(csv_file_name):
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
    model = MODEL

    train_start_time = time.time()
    model.fit(x_train, y_train)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    writeResults("training-time", train_time)

    latest_date = csv_file_name.split("/")[1].split(".")[0]
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

    #----------------------------------------------------------------
    pred_start_time = time.time()
    y_pred = model.predict(x_test)
    pred_end_time = time.time()

    pred_time = pred_end_time - pred_start_time
    writeResults("prediction-time", pred_time)

    pred_benign_count = 0
    pred_malisious_count = 0
    for y in y_pred:
        if y == 0:  # benign
            pred_benign_count += 1
        elif y == 1:    # malisious
            pred_malisious_count += 1
    print("----------------------------------------------------------")        
    print("PREDICT: benign:" + str(pred_benign_count) + " malisious:" + str(pred_malisious_count))

    true_benign_count = 0
    true_malicious_count = 0
    for y in y_test:
        if y == 0:
            true_benign_count += 1
        elif y == 1:
            true_malicious_count += 1
    print("ANSWER : benign:" + str(true_benign_count) + " malisious:" + str(true_malicious_count))
    #---------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("F1 SCORE:", f1)

    writeResults("precision", precision)
    writeResults("recall", recall)
    writeResults("f1", f1)



def writeResults(word, result):
    f = open(results_dir_name + "/" + algorithm + "single_" + dataset_csv_dir_name.split("_")[0] + "_" + evaluate_csv_dir_name.split("_")[0] + "_" + word + ".txt", "a")
    f.write(str(result) + "\n")
    f.close()


def setBeforeDatasetCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn


def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn


if __name__ == "__main__":  #main
    train_count = 0

    day = 1 
    hour = 0
    print("")
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
        if (no_update_flag == True) and (train_count <= MODEL_INDEX): 
            if csv_file_name in glob.glob(dataset_csv_dir_name + "/*.csv"):
                model = train(csv_file_name)
                setBeforeDatasetCsvFile(csv_file_name)
            else:   # not found
                print("TRAINING:" + csv_file_name + ":not found")
                model = train(before_csv_file_name)
            train_count += 1

        if no_update_flag == False:
            if csv_file_name in glob.glob(dataset_csv_dir_name + "/*.csv"):
                model = train(csv_file_name)
                setBeforeDatasetCsvFile(csv_file_name)
            else:   # not found
                print("TRAINING:" + csv_file_name + ":not found")
                model = train(before_csv_file_name)
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

        #------------------------------------------------------------------------------------
        csv_file_name = evaluate_csv_dir_name + "/" + DATE + eva_str_day + eva_str_hour + ".csv"
        if csv_file_name in glob.glob(evaluate_csv_dir_name + "/*.csv"):
            evaluate(model, csv_file_name)
            setBeforeEvaluateCsvFile(csv_file_name)
        else:   # not found
            print("TEST:" + csv_file_name + ":not found")
            evaluate(model, before_evaluate_csv_file_name)
        #------------------------------------------------------------------------------------

        print("")
        hour += 1
        if hour == 24:
            hour = 0
            day += 1