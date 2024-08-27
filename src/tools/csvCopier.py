import os
import sys
import glob
import shutil



INITIAL_DAY = 1
INITIAL_HOUR = 0
HOURS = 24
#------------------------------------------------
args = sys.argv
copy_target_dir = str(args[1]).replace("/","")
year_month = str(args[2])
copied_days = int(args[3])
#------------------------------------------------

def transformString(number):
    if number < 10:
        str_number = "0" + str(number)
    else:
        str_number = str(number)

    return str_number 


if __name__ == "__main__":  #main

    copy_target_fn = year_month + transformString(INITIAL_DAY) + transformString(INITIAL_HOUR) + ".csv"
    if copy_target_dir + "/" + copy_target_fn in glob.glob(copy_target_dir + "/*.csv"):

        file_count = 0
        for i in range(copied_days):
            str_day = transformString(i + INITIAL_DAY)

            for l in range(HOURS):
                if l + INITIAL_HOUR >= HOURS:
                    pass
                else:
                    str_hour = transformString(l + INITIAL_HOUR)
                    fn = year_month + str_day + str_hour + ".csv"
                    if fn == copy_target_fn:
                        pass
                    else:
                        shutil.copy(copy_target_dir + "/" + copy_target_fn, copy_target_dir + "/" + fn)
                        file_count += 1
                        print(str(file_count) + ":" + fn + ":saved")
                    
    else:
        print("copy target file not found!")