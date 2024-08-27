import fireducks.pandas as pd

import glob
import os
import sys
import datetime



DATE = "202201"
DAYS = 30

HOURS = 24
#---------------------------------------
args = sys.argv

csv_dir_count = 0
csv_dir_list = []
for i in range(len(args)-1):
    csv_dir_count += 1
    csv_dir_list.append(str(args[i+1]))
#---------------------------------------
str_dt_now = str(datetime.datetime.now()).split(" ")[0].replace("-","")
output_dir_name = str_dt_now + "-created-" + str(csv_dir_count) + "dirs-mixed_csv"
if not os.path.exists(output_dir_name):
    os.mkdir(output_dir_name)
#---------------------------------------


def mixCsv(csv_file_list, output_fn):
    list = []
    for csv_path in csv_file_list:
        list.append(pd.read_csv(csv_path))
    df = pd.concat(list, axis=0, sort=False)

    if os.path.exists(output_dir_name):
        df.to_csv(output_dir_name + "/" + output_fn, index=False)
        print(output_fn + ":saved")


day = 1 
hour = 0
for i in range(DAYS * HOURS):
    if day < 10:
        str_day = "0" + str(day)
    else:
        str_day = str(day)
    if hour < 10:
        str_hour = "0" + str(hour)
    else:
        str_hour = str(hour)

    csv_file_list =[]
    for csv_dir in csv_dir_list:
        csv_files = glob.glob(csv_dir + "*.csv")
        if csv_dir + DATE + str_day + str_hour + ".csv" in csv_files:
            csv_file_list.append(csv_dir + DATE + str_day + str_hour + ".csv")

    output_fn = DATE + str_day + str_hour + ".csv"
    mixCsv(csv_file_list, output_fn)

    hour += 1
    if hour == 24:
        hour = 0
        day += 1

print("saved:")
print(output_dir_name + "/")
print("")