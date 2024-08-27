import os
import warnings
warnings.simplefilter('ignore')

import flwr as fl
import fireducks.pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss

import glob
import sys
import time
import datetime

from DecisionTableClassifier import *



MAX_DEPTH = 5
MAX_LEAF_NODES = 16

LEFT_TREES_MULTIPLIER = 1
OWN_TREE_INFLUENCE = 3

DATE = "202201"
DAYS = 7
N_TRAINING = 24
#------------------------------------------------------------------------------------
results_dir_list = glob.glob("*-results")
for dir_name in results_dir_list:
    if dir_name == "*Single*":
        results_dir_list.remove(dir_name)

int_last_date = 0
for dir_name in results_dir_list:
    date = int(dir_name.split("-")[0])
    if date > int_last_date:
        int_last_date = date

RESULTS_DIR_NAME = str(date) + "-results"
print("DIR:" + RESULTS_DIR_NAME + ":set")
#------------------------------------------------------------------------------------
# [1]:train dir name [2]:test dir name [3]:number of clients [4]:client id
args = sys.argv
dataset_csv_dir_name = str(args[1]).replace("/","") 
evaluate_csv_dir_name = str(args[2]).replace("/","")
N_CLIENTS = int(args[3])
CLIENT_ID = int(args[4])
print("CLIENT ID:", CLIENT_ID)

# [5]:left_trees_multiplier [6]:own_tree_influence
if len(args) > 5:
    LEFT_TREES_MULTIPLIER = int(args[5])

    if args[6] != "None":
        OWN_TREE_INFLUENCE = float(args[6])
    else:
        OWN_TREE_INFLUENCE = None   # simple majority rule

LEFT_TREES = N_CLIENTS * LEFT_TREES_MULTIPLIER
#------------------------------------------------------------------------------------
before_csv_file_name = ""
before_evaluate_csv_file_name = ""
tmp_accuracy = -1
tmp_f1 = -1
break_flag = False
first_trained_flag = False
before_evaluate_csv_used_flag = False
#-----------------------------------------------------------------------------------


def client(csv_file_name, elapsed_time_count, model):
    global first_trained_flag
    global before_evaluate_csv_used_flag

    before_evaluate_csv_used_flag = False
    #------------------------------------ get training data -----------------------------------
    df = pd.read_csv(csv_file_name)

    if len(df) != 0: 
        x_train = df.iloc[:,3:-1].values
        y_train = df.loc[:,"label"].values

    else:   # empty data
        print("TRAINING:" + csv_file_name + ":no data")
        print("TRAINING:" + before_csv_file_name + ":used")
        df = pd.read_csv(before_csv_file_name)
        x_train = df.iloc[:,3:-1].values
        y_train = df.loc[:,"label"].values

    dataset_records = len(x_train)
    print("TRAINING:dataset " + str(dataset_records) + " records")

    training_benign_count = 0
    training_malicious_count = 0
    for y in y_train:
        if int(y) == 0: # benign
            training_benign_count += 1
        elif int(y) == 1:   # malicious
            training_malicious_count += 1

    print("TRAINING:ben " + str(training_benign_count) + " records")
    print("TRAINING:mal " + str(training_malicious_count) + " records")
    writeResults("benign-records", training_benign_count, elapsed_time_count)
    writeResults("malicious-records", training_malicious_count, elapsed_time_count)
    #--------------------------------------------------------------------------------

    if first_trained_flag == False: # Round:0
        model.fit(x_train, y_train)
        first_trained_flag = True

    #---------------------------------- get test data --------------------------------
    date = csv_file_name.split("/")[1].split(".")[0]
    month = date[:6]
    day = int(date[6:8])
    hour = int(date[8:10])

    hour += 1
    if hour == N_TRAINING:
        hour = 0
        day += 1
    if day < 10:
        str_day = "0" + str(day)
    else:
        str_day = str(day)
    if hour < 10:
        str_hour = "0" + str(hour)
    else:
        str_hour = str(hour)

    evaluate_csv_file_name = evaluate_csv_dir_name + "/" + month + str_day + str_hour + ".csv"
    if evaluate_csv_file_name in glob.glob(evaluate_csv_dir_name + "/*.csv"):
        print("TEST:" + evaluate_csv_file_name + ":found")
        evaluate_df = pd.read_csv(evaluate_csv_file_name)

        if len(evaluate_df) != 0:
            x_test = evaluate_df.iloc[:,3:-1].values
            y_test = evaluate_df.loc[:,"label"].values
            setBeforeEvaluateCsvFile(evaluate_csv_file_name)

        else:   # empty data
            print("TEST:" + evaluate_csv_file_name + ":no data")
            print("TEST:" + before_evaluate_csv_file_name + ":used")
            evaluate_df = pd.read_csv(before_evaluate_csv_file_name)
            x_test = evaluate_df.iloc[:,3:-1].values
            y_test = evaluate_df.loc[:,"label"].values        

    else:   # not found
        print("TEST:" + evaluate_csv_file_name + ":not found")
        print("TEST:" + before_evaluate_csv_file_name + ":used")
        evaluate_df = pd.read_csv(before_evaluate_csv_file_name)
        x_test = evaluate_df.iloc[:,3:-1].values
        y_test = evaluate_df.loc[:,"label"].values

    evaluate_records = len(x_test)
    print("TEST:" + str(evaluate_records) + " records")
    #--------------------------------------------------------------------------------

    class Client(fl.client.NumPyClient):
        def get_parameters(self, config): # server get params
            return model.get_params()

        def fit(self, parameters, config):
            train_start_time = time.time()
            model.fit(x_train, y_train)
            train_end_time = time.time()

            train_time = train_end_time - train_start_time
            writeResults("training-time", train_time, elapsed_time_count)

            return model.get_params(), 1, {}

        def evaluate(self, parameters, config):
            global tmp_accuracy
            global tmp_f1
            global break_flag

            model.set_params(parameters)
            model.thin_trees()
            print("TREES:", model.get_n_trees())
            
            loss = log_loss(y_test, model.predict_proba(x_test))
            #----------------------------------------------------------------
            predict_start_time = time.time()
            y_pred = model.predict(x_test)
            predict_end_time = time.time()
            predict_time = predict_end_time - predict_start_time

            pred_benign_count = 0
            pred_malisious_count = 0
            for y in y_pred:
                if y == 0:  # benign
                    pred_benign_count += 1
                elif y == 1:    # malisious
                    pred_malisious_count += 1
            print("----------------------------------------------------------")
            print("PREDICT: benign:" + str(pred_benign_count) + " malisious:" + str(pred_malisious_count))
            #------------------------------------------------------------------
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

            #print("ACCURACY:", accuracy)
            print("PRECISIN:", precision)
            print("RECALL  :", recall)
            print("F1 SCORE:", f1)
            print("----------------------------------------------------------")

            writeResults("precision", precision, elapsed_time_count)
            writeResults("recall", recall, elapsed_time_count)
            writeResults("f1", f1, elapsed_time_count)
            writeResults("prediction-time", predict_time, elapsed_time_count)
            #------------------------------------------------------------------

            return loss, len(x_test), {"accuracy": accuracy}
    #----------------------------- start Flower client ----------------------------
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client())



def writeResults(word, result, elapsed_time_count):
    f = open(RESULTS_DIR_NAME + "/" + "DTclient_" + dataset_csv_dir_name.split("_")[0] + "_" + evaluate_csv_dir_name.split("_")[0] + "_" + word + ".txt", "a")
    f.write(str(result) + "\n")
    if elapsed_time_count == DAYS * N_TRAINING:
        f.write("------------------\n")
    f.close()

def setBeforeTrainingCsvFile(fn):
    global before_csv_file_name
    before_csv_file_name = fn

def setBeforeEvaluateCsvFile(fn):
    global before_evaluate_csv_file_name
    before_evaluate_csv_file_name = fn


if __name__ == "__main__":  #main

    day = 1     # start day
    hour = 0    # start hour
    elapsed_time_count = 0

    model = DecisionTableClassifier(
        max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAF_NODES,
        client_id=CLIENT_ID, n_clients=N_CLIENTS, 
        left_trees=LEFT_TREES, own_tree_influence=OWN_TREE_INFLUENCE)

    for i in range(DAYS * N_TRAINING):
        
        if break_flag == True:
            break
        else:
            elapsed_time_count += 1
        #------------------------------
        if day < 10:
            str_day = "0" + str(day)
        else:
            str_day = str(day)
        if hour < 10:
            str_hour = "0" + str(hour)
        else:
            str_hour = str(hour)
        #------------------------------
        csv_file_name = dataset_csv_dir_name + "/" + DATE + str_day + str_hour + ".csv"
        if csv_file_name in glob.glob(dataset_csv_dir_name + "/*.csv"):
            print("TRAINING:" + csv_file_name + ":found")
            client(csv_file_name, elapsed_time_count, model)
            setBeforeTrainingCsvFile(csv_file_name)
            print("")

        else:   # not found
            print("TRAINING:" + csv_file_name + ":not found")
            print("TRAINING:" + before_csv_file_name + ":used")
            client(before_csv_file_name, elapsed_time_count, model)
            print("")
        #--------------------------------
        hour += 1
        if hour == N_TRAINING:
            hour = 0
            day += 1