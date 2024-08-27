import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    #log amount
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    #gpu mem limit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #cpu : -1

import warnings
warnings.simplefilter('ignore')

import flwr as fl
import tensorflow as tf
import fireducks.pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import glob
import sys
import time
import datetime

from nnModelDefiner import createModel



EPOCHS = 30
BATCH_SIZE = 10

DATE = "202201"
DAYS = 7
N_TRAINING = 24
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
# [1]:train dir name [2]:test dir name
args = sys.argv
dataset_csv_dir_name = str(args[1]).replace("/","") 
evaluate_csv_dir_name = str(args[2]).replace("/","")
#------------- option -------------
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
#------------------------------------------------------------------------------
before_csv_file_name = ""
before_evaluate_csv_file_name = ""
tmp_accuracy = -1
tmp_f1 = -1
break_flag = False
scaler = StandardScaler()
first_trained_flag = False
before_evaluate_csv_used_flag = False
#------------------------------------------------------------------------------


def client(csv_file_name, elapsed_time_count):
    global first_trained_flag
    global before_evaluate_csv_used_flag

    before_evaluate_csv_used_flag = False
    #------------------------------------ get training data --------------------------
    df = pd.read_csv(csv_file_name)

    if len(df) != 0: 
        df_x = df.iloc[:,3:-1].values
        if first_trained_flag == False:  
            x_train = scaler.fit_transform(df_x)
            first_trained_flag = True
        else:
            x_train = scaler.transform(df_x)
        y_train = df.loc[:,"label"].values

    else:   # empty data
        print("TRAINING:" + csv_file_name + ":no data")
        print("TRAINING:" + before_csv_file_name + ":used")
        df = pd.read_csv(before_csv_file_name)

        df_x = df.iloc[:,3:-1].values  
        x_train = scaler.transform(df_x)
        y_train = df.loc[:,"label"].values

    dataset_records = len(x_train)
    print("TRAINING:dataset " + str(dataset_records) + " records")

    training_benign_records = 0
    training_malicious_records = 0
    for y in y_train:
        if int(y) == 0: # benign
            training_benign_records += 1
        elif int(y) == 1:   #malicious
            training_malicious_records += 1

    print("TRAINING:ben " + str(training_benign_records) + " records")
    print("TRAINING:mal " + str(training_malicious_records) + " records")
    writeResults("benign-records", training_benign_records, elapsed_time_count)
    writeResults("malicious-records", training_malicious_records, elapsed_time_count)

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
            x_test = scaler.transform(x_test)
            y_test = evaluate_df.loc[:,"label"].values
            setBeforeEvaluateCsvFile(evaluate_csv_file_name)

        else:   # empty data
            print("TEST:" + evaluate_csv_file_name + ":no data")
            print("TEST:" + before_evaluate_csv_file_name + ":used")
            evaluate_df = pd.read_csv(before_evaluate_csv_file_name)
            x_test = evaluate_df.iloc[:,3:-1].values
            x_test = scaler.transform(x_test)
            y_test = evaluate_df.loc[:,"label"].values
            before_evaluate_csv_used_flag = True  

    else:   # not found
        print("TEST:" + evaluate_csv_file_name + ":not found")
        print("TEST:" + before_evaluate_csv_file_name + ":used")
        evaluate_df = pd.read_csv(before_evaluate_csv_file_name)
        x_test = evaluate_df.iloc[:,3:-1].values
        x_test = scaler.transform(x_test)
        y_test = evaluate_df.loc[:,"label"].values
        before_evaluate_csv_used_flag = True

    evaluate_records = len(x_test)
    print("TEST:" + str(evaluate_records) + " records")
    #--------------------------------------------------------------------------------

    model = createModel()

    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            train_start_time = time.time()
            model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
            train_end_time = time.time()

            train_time = train_end_time - train_start_time
            writeResults("training-time", train_time, elapsed_time_count)

            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            global tmp_accuracy
            global tmp_f1
            global break_flag
            
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
            #----------------------------------------------------------------
            predict_start_time = time.time()
            y_pred_origin = model.predict(x_test, batch_size=BATCH_SIZE)
            predict_end_time = time.time()
            predict_time = predict_end_time - predict_start_time

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
            #------------------------------------------------------------------
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

            #print("ACCURACY:", accuracy)
            print("PRECISIN:", precision)
            print("RECALL  :", recall)
            print("F1 SCORE:", f1)
            print("----------------------------------------------------------")

            #writeResults("accuracy", accuracy, elapsed_time_count)
            writeResults("precision", precision, elapsed_time_count)
            writeResults("recall", recall, elapsed_time_count)
            writeResults("f1", f1, elapsed_time_count)
            writeResults("prediction-time", predict_time, elapsed_time_count)
            #------------------------------------------------------------------

            return loss, len(x_test), {"accuracy": accuracy}
    #----------------------------- start Flower client ----------------------------
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client())


def writeResults(word, result, elapsed_time_count):
    f = open(RESULTS_DIR_NAME + "/" + "NNclient_" + dataset_csv_dir_name.split("_")[0] + "_" + evaluate_csv_dir_name.split("_")[0] + "_" + word + ".txt", "a")
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

    day = 1
    hour = 0
    elapsed_time_count = 0
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
            client(csv_file_name, elapsed_time_count)
            setBeforeTrainingCsvFile(csv_file_name)
            print("")

        else:   # not found
            print("TRAINING:" + csv_file_name + ":not found")
            print("TRAINING:" + before_csv_file_name + ":used")
            client(before_csv_file_name, elapsed_time_count)
            print("")
        #--------------------------------
        hour += 1
        if hour == N_TRAINING:
            hour = 0
            day += 1