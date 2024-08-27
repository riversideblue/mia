#!/bin/bash



python3 Single.py australiaeast-Jan-lab01-Jan-mixed_csv/ australiaeast-Jan-lab01-Jan-mixed_csv/ RF > dts0.txt &
python3 Single.py eastus-Jan-lab02-Jan-mixed_csv/ eastus-Jan-lab02-Jan-mixed_csv/ RF > dts1.txt &
python3 Single.py uksouth-Jan-lab03-Jan-mixed_csv/ uksouth-Jan-lab03-Jan-mixed_csv/ RF > dts2.txt &
python3 Single.py japaneast-Jan-lab04-Jan-mixed_csv/ japaneast-Jan-lab04-Jan-mixed_csv/ RF

#python3 Single.py australiaeast-Jan-lab01-Jan-mixed_csv/ australiaeast-Jan-lab01-Jan-mixed_csv/ GBDT -S 0 > dts0.txt &
#python3 Single.py eastus-Jan-lab02-Jan-mixed_csv/ eastus-Jan-lab02-Jan-mixed_csv/ GBDT -S 0 > dts1.txt &
#python3 Single.py uksouth-Jan-lab03-Jan-mixed_csv/ uksouth-Jan-lab03-Jan-mixed_csv/ GBDT -S 0 > dts2.txt &
#python3 Single.py japaneast-Jan-lab04-Jan-mixed_csv/ japaneast-Jan-lab04-Jan-mixed_csv/ GBDT -S 0 