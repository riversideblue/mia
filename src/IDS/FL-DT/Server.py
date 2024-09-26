import flwr as fl

import datetime
import pickle
import glob
import os
import time
import sys



DATE = "202201"
DAYS = 7
N_TRAINING = 24 # number of training in 1 day

N_ROUNDS = 1  # number of rounds in 1 training

str_dt_now = str(datetime.datetime.now()).split(" ")[0].replace("-","")+str(datetime.datetime.now()).split(" ")[1].split(".")[0].replace(":","")

PARAMS_DIR_NAME = str_dt_now + "-params"
if not os.path.isdir(PARAMS_DIR_NAME):
    os.mkdir(PARAMS_DIR_NAME)
    print("DIR:" + PARAMS_DIR_NAME + ":made")

RESULTS_DIR_NAME = str_dt_now + "-results"
if not os.path.isdir(RESULTS_DIR_NAME):
    os.mkdir(RESULTS_DIR_NAME)
    print("DIR:" + RESULTS_DIR_NAME + ":made")
#------------------------------------------------------------------------------------
# [1]:number of clients
args = sys.argv
MIN_AVAILABLE_CLIENTS = int(args[1])
MIN_FIT_CLIENTS = int(args[1])
#------------------------------------------------------------------------------------

def server(date):

    class SaveModelStrategy(fl.server.strategy.FedAvg): 
        def aggregate_fit(self, server_round, results, failures):
            aggregated_weights = super().aggregate_fit(server_round, results, failures)
            if aggregated_weights is not None:
                with open(PARAMS_DIR_NAME +  "/" + date + "-param.pickle", 'wb') as f:
                    print("PARAM:" + date + "-param.pickle:saved")
                    pickle.dump(aggregated_weights, f)  # save param 
            return aggregated_weights

    #------------------- get initial param --------------------------
    init_param = None
    pickle_list = glob.glob(PARAMS_DIR_NAME + "/*.pickle")
    if len(pickle_list) != 0:
        latest = 0
        for file_path in pickle_list:
            int_date = int(file_path.split("/")[1].split("-")[0]) 
            if int_date > latest:
                latest = int_date

        with open(PARAMS_DIR_NAME + "/" + str(latest) +"-param.pickle", 'rb') as f:
            print("PARAM:" + str(latest) + "-param.pickle:found")
            init_weights = pickle.load(f)
            init_param = init_weights[0]
    else:
        pass
    #--------------------------------------------------------------
    strategy = SaveModelStrategy(
        min_fit_clients = MIN_FIT_CLIENTS,  #join training
        min_available_clients = MIN_AVAILABLE_CLIENTS,    #join FL system
        initial_parameters = init_param
    )
    #-----------------------start Flower server--------------------
    fl.server.start_server(
        server_address = "localhost:8080",
        config = fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy = strategy
    )


if __name__ == "__main__":

    day = 1
    hour = 0
    for i in range(DAYS * N_TRAINING):
        if day < 10:
            str_day = "0" + str(day)
        else:
            str_day = str(day)
        if hour < 10:
            str_hour = "0" + str(hour)
        else:
            str_hour = str(hour)
        
        #---------------------------------
        date = DATE + str_day + str_hour
        server(date)
        print("")
        #---------------------------------

        hour += 1
        if hour == N_TRAINING:
            hour = 0
            day += 1