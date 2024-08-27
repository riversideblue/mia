#!/bin/bash
#---------- server -------------------
# args[1]:number of clients
#---------- client --------------------
# args[1]:dataset csv dir name
# args[2]:evaluate csv dir name
#--------------------------------------



nohup python3 Server.py 4 &
nohup python3 NNclient.py australiaeast-Jan-lab01-Jan-mixed_csv/ australiaeast-Jan-lab01-Jan-mixed_csv/ > nn-c0.txt &
nohup python3 NNclient.py eastus-Jan-lab02-Jan-mixed_csv/ eastus-Jan-lab02-Jan-mixed_csv/ > nn-c1.txt &
nohup python3 NNclient.py uksouth-Jan-lab03-Jan-mixed_csv/ uksouth-Jan-lab03-Jan-mixed_csv/ > nn-c2.txt &
python3 NNclient.py japaneast-Jan-lab04-Jan-mixed_csv/ japaneast-Jan-lab04-Jan-mixed_csv/