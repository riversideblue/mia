#!/bin/bash



python3 DTsingle.py australiaeast-Jan-lab01-Jan-mixed_csv/ australiaeast-Jan-lab01-Jan-mixed_csv/ > dts0.txt &
python3 DTsingle.py eastus-Jan-lab02-Jan-mixed_csv/ eastus-Jan-lab02-Jan-mixed_csv/ > dts1.txt &
python3 DTsingle.py uksouth-Jan-lab03-Jan-mixed_csv/ uksouth-Jan-lab03-Jan-mixed_csv/ > dts2.txt &
python3 DTsingle.py japaneast-Jan-lab04-Jan-mixed_csv/ japaneast-Jan-lab04-Jan-mixed_csv/