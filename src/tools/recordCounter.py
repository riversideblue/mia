import csv
import sys
import glob



record_count = 0
mal_count = 0
ben_count = 0

args = sys.argv
dir_name = str(args[1])

csv_file_list = glob.glob(dir_name + "*.csv")
for csv_path in csv_file_list:
    with open(csv_path) as rf:
        reader = csv.reader(rf)
        for row in reader:
            record_count += 1
            if row[17] == "1":
                mal_count += 1
            elif row[17] == "0":
                ben_count += 1

        record_count -= 1   #header row

print("total records")
print("malicious records")
print("benign records")
print("---------------")
print(str(record_count))
print(str(mal_count))
print(str(ben_count))