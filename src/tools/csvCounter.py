import sys
import glob

args = sys.argv
dir_name = str(args[1])
print(len(glob.glob(dir_name + "*.csv")))