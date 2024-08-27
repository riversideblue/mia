import glob
import os
import csv
import sys
import shutil



args = sys.argv
target_dir_name = str(args[1]).replace("/","")

output_dir_name = target_dir_name.split("_")[0] + "-cleaned_csv"
cleaned_count = 0


def cleanCsv(csv_path):
    global cleaned_count

    csv_file_name = csv_path.split("/")[1]

    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)

    with open(csv_path) as rf:
        reader = csv.reader(rf)
        row_count = 0
        for row in reader:
            row_count += 1

    print(csv_path + ":" + str(row_count-1) + " rows", end="")
    if row_count <= 1:
        print(" => no data")
        cleaned_count += 1

    else:
        print("")
        shutil.copy(csv_path, output_dir_name + "/" + csv_file_name)
        #print(output_dir_name + "/" + csv_file_name + ":saved")


for path in glob.glob(target_dir_name + "/" + "*.csv"):
    cleanCsv(path)

print("removed csv:" + str(cleaned_count))
print("")