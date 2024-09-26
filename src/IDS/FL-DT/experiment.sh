#!/bin/bash
#---------- server -------------------
# args[1]:number of clients
#---------- client --------------------
# args[1]:dataset csv dir name
# args[2]:evaluate csv dir name
#--------------------------------------



nohup python3 Server.py 4 &
nohup python3 DTclient.py australiaeast-Jan-lab01-Jan-mixed_csv/ australiaeast-Jan-lab01-Jan-mixed_csv/ 4 0 > dt-c0.txt &
nohup python3 DTclient.py eastus-Jan-lab02-Jan-mixed_csv/ eastus-Jan-lab02-Jan-mixed_csv/ 4 1 > dt-c1.txt &
nohup python3 DTclient.py uksouth-Jan-lab03-Jan-mixed_csv/ uksouth-Jan-lab03-Jan-mixed_csv/ 4 2 > dt-c2.txt &
python3 DTclient.py japaneast-Jan-lab04-Jan-mixed_csv/ japaneast-Jan-lab04-Jan-mixed_csv/ 4 3