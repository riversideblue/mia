import glob
import os
import csv
import random
import sys



LEFT_RATE = 0.9
SHEVED_LABEL = "1"
#-----------------------------------------------------------
args = sys.argv
target_dir_name = str(args[1]).replace("/","")

if len(args) == 3:
    LEFT_RATE = float(args[2])

if len(args) == 4:
    LEFT_RATE = float(args[2])
    SHEVED_LABEL = str(args[3])

output_dir_name = target_dir_name.split("_")[0] + "-sheved_csv"
#------------------------------------------------------------

def sheveCsv(csv_path):
    csv_file_name = csv_path.split("/")[1]

    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)
    else:
        pass

    with open(output_dir_name + "/" + csv_file_name, "w") as wf:
        writer = csv.writer(wf, lineterminator="\n")

        label_column = 0
        first_row_flag = False
        with open(csv_path) as rf:
            reader = csv.reader(rf)
            for row in reader:
                if first_row_flag == False:
                    for i in range(len(row)):
                        if row[i] == "label":
                            label_column = i
                    writer.writerow(row)        
                    first_row_flag = True

                else:   # not first row
                    if row[label_column] == SHEVED_LABEL:
                        if random.random() < LEFT_RATE: # left
                            writer.writerow(row)
                        else:                           # removed
                            pass

                    else:
                        writer.writerow(row)



for path in glob.glob(target_dir_name + "/" + "*.csv"):
    sheveCsv(path)

print("")