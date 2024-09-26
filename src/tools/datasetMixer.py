import fireducks.pandas as pd

import glob
import os
import sys



args = sys.argv
dir_name1 = str(args[1]).replace("/","")
if "-mixed" in dir_name1:
    out_dir_name1 = dir_name1.replace("-mixed","")
else:
    out_dir_name1 = dir_name1

dir_name2 = str(args[2]).replace("/","")
if "-mixed" in dir_name2:
    out_dir_name2 = dir_name2.replace("-mixed","")
else:
    out_dir_name2 = dir_name2

output_dir_name = out_dir_name1.split("_")[0] + "-" + out_dir_name2.split("_")[0] + "-mixed_csv"

dir_name1_list = glob.glob(dir_name1 + "/" + "*.csv")
dir_name2_list = glob.glob(dir_name2 + "/" + "*.csv")

def mixCsv(fn1_path, fn2_path):
    print(fn1_path + " + " + fn2_path)
    output_fn = fn1_path.split("/")[1]

    list = []
    list.append(pd.read_csv(fn1_path))
    list.append(pd.read_csv(fn2_path))
    
    df = pd.concat(list, axis=0, sort=False)
    if os.path.exists(output_dir_name):
        df.to_csv(output_dir_name + "/" + output_fn, index=False)
    else:
        os.mkdir(output_dir_name)
        df.to_csv(output_dir_name + "/" + output_fn, index=False)

for fn1_path in dir_name1_list:
    fn1 = fn1_path.split("/")[1]
    for fn2_path in dir_name2_list:
        fn2 = fn2_path.split("/")[1]
        if fn1 == fn2:
            mixCsv(fn1_path, fn2_path)
        else:
            pass

print(output_dir_name + "/")
print("")