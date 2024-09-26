import fireducks.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import glob
import sys
import os



y_top = 1.0 # vertical axis upper limit
y_bottom = 0.0  # vertical axis lower limit

font_size = 16  # default font size

start_d = '2022-01-01 00:00'
end_d = '2022-01-08 00:00'
interval_hours = 24 # horizontal axis scale interval  

all_color_list = ["darkorange", "royalblue", "forestgreen", "dimgrey", "mediumorchid"]

txt_dir = 'txt' # txt file dir
save_dir = "save" # saved file dir

saved_file_name = "graph"
#------------------------------------------------------------------------
legend_list = []
args = sys.argv

legend_count = 0
for i in range(len(args)-1):
    legend_list.append(args[i+1])   # set legends from command line args
    legend_count += 1
#-----------------------------------------------------------------------

color_list = []
for clr in all_color_list:
    color_list.append(clr)

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

acc_content_list = []
acc_label = "Accuracy"
acc_file_name = saved_file_name + "_acc"

pre_content_list = []
pre_label = "Precision"
pre_file_name = saved_file_name + "_pre"

rec_content_list = []
rec_label = "Recall"
rec_file_name = saved_file_name + "_rec"

f1s_content_list = []
f1s_label = "F1 score"
f1s_file_name = saved_file_name + "_f1"

txt_fn_list= []
for full_file_path in glob.glob(txt_dir + "/" + "*"):
    file_path = full_file_path.replace(txt_dir + "/", "")

    evl = file_path.split("_")[-1]
    txt_fn = file_path.replace("_"+evl, "")
    if txt_fn not in txt_fn_list:
        txt_fn_list.append(txt_fn)

    if "accuracy" in evl:
        with open(full_file_path, 'r') as file:
            acc_content_list.append(file.readlines())
    elif "precision" in evl:
        with open(full_file_path, 'r') as file:
            pre_content_list.append(file.readlines())
    elif "recall" in evl:
        with open(full_file_path, 'r') as file:
            rec_content_list.append(file.readlines())
    elif "f1" in evl:
        with open(full_file_path, 'r') as file:
            f1s_content_list.append(file.readlines())
    else:
        pass

if len(legend_list) == 0:
    for tfn in txt_fn_list:
        print(tfn)
        legend_list.append(tfn)

#--------------------------------------------------------

def extract_numbers(lines):
    numbers = []
    for line in lines:
        try:
            number = float(line.strip())
            numbers.append(number)
        except ValueError:
            continue
    return numbers

acc_data_list = []
for cont in acc_content_list: 
    data = extract_numbers(cont)
    acc_data_list.append(data)

pre_data_list = []
for cont in pre_content_list: 
    data = extract_numbers(cont)
    pre_data_list.append(data)

rec_data_list = []
for cont in rec_content_list: 
    data = extract_numbers(cont)
    rec_data_list.append(data)

f1s_data_list = []
for cont in f1s_content_list: 
    data = extract_numbers(cont)
    f1s_data_list.append(data)

#----------------------------------------------------------------
start_date = pd.Timestamp(start_d)
end_date = pd.Timestamp(end_d)
hours = pd.date_range(start=start_date, end=end_date, freq='H')

plt.figure(figsize=(8.3, 3.8))  # figure size(width, height)
#----------------------------------------------------------------



#-------------------------- accuracy ----------------------------
if len(acc_data_list) != 0:
    plt.rcParams["font.size"] = font_size

    for i in range(len(acc_data_list)):
        plt.plot(hours[1:], acc_data_list[i], label=legend_list[i], color=color_list[i])

    plt.xlim(left=hours[1], right=end_date)
    plt.ylim(bottom=y_bottom, top=y_top)

    # 補助線の有無
    plt.grid(False)

    # 凡例
    plt.legend(loc=(0.7, 0.05), fontsize=17)

    # 黒い枠線を追加
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # 軸ラベルを設定
    #plt.xlabel("Date", fontsize=15)
    plt.ylabel(acc_label, fontsize=19)

    # 横軸のフォーマットとメモリ間隔を設定、日付を斜めに表示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    plt.xticks([start_date] + list(hours[::interval_hours]), rotation=15)

    # 補助メモリの向き
    plt.tick_params(direction='in')

    # 余白
    plt.subplots_adjust(left=0.1, right=0.92, bottom=0.14, top=0.95)

    # PDFとPNGで出力
    output_pdf_path = save_dir + "/" + acc_file_name + '.pdf'
    output_png_path = save_dir + "/" + acc_file_name + '.png'
    plt.savefig(output_pdf_path)
    plt.savefig(output_png_path)

#-------------------------- precision ----------------------------
if len(pre_data_list) != 0:
    plt.clf()
    plt.rcParams["font.size"] = font_size

    for i in range(len(pre_data_list)):
        plt.plot(hours[1:], pre_data_list[i], label=legend_list[i], color=color_list[i])

    plt.xlim(left=hours[1], right=end_date)
    plt.ylim(bottom=y_bottom, top=y_top)

    # 補助線の有無
    plt.grid(False)

    # 凡例
    plt.legend(loc=(0.7, 0.05), fontsize=17)

    # 黒い枠線を追加
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # 軸ラベルを設定
    #plt.xlabel("Date", fontsize=15)
    plt.ylabel(pre_label, fontsize=19)

    # 横軸のフォーマットとメモリ間隔を設定、日付を斜めに表示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    plt.xticks([start_date] + list(hours[::interval_hours]), rotation=15)

    # 補助メモリの向き
    plt.tick_params(direction='in')

    # 余白
    plt.subplots_adjust(left=0.1, right=0.92, bottom=0.14, top=0.95)

    # PDFとPNGで出力
    output_pdf_path = save_dir + "/" + pre_file_name + '.pdf'
    output_png_path = save_dir + "/" + pre_file_name + '.png'
    plt.savefig(output_pdf_path)
    plt.savefig(output_png_path)

#-------------------------- recall ----------------------------
if len(rec_data_list) != 0:
    plt.clf()
    plt.rcParams["font.size"] = font_size

    for i in range(len(rec_data_list)):
        plt.plot(hours[1:], rec_data_list[i], label=legend_list[i], color=color_list[i])

    plt.xlim(left=hours[1], right=end_date)
    plt.ylim(bottom=y_bottom, top=y_top)

    # 補助線の有無
    plt.grid(False)

    # 凡例
    plt.legend(loc=(0.7, 0.05), fontsize=17)

    # 黒い枠線を追加
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # 軸ラベルを設定
    #plt.xlabel("Date", fontsize=15)
    plt.ylabel(rec_label, fontsize=19)

    # 横軸のフォーマットとメモリ間隔を設定、日付を斜めに表示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    plt.xticks([start_date] + list(hours[::interval_hours]), rotation=15)

    # 補助メモリの向き
    plt.tick_params(direction='in')

    # 余白
    plt.subplots_adjust(left=0.1, right=0.92, bottom=0.14, top=0.95)

    # PDFとPNGで出力
    output_pdf_path = save_dir + "/" + rec_file_name + '.pdf'
    output_png_path = save_dir + "/" + rec_file_name + '.png'
    plt.savefig(output_pdf_path)
    plt.savefig(output_png_path)

#-------------------------- f1 score ----------------------------
if len(f1s_data_list) != 0:
    plt.clf()
    plt.rcParams["font.size"] = font_size

    for i in range(len(f1s_data_list)):
        plt.plot(hours[1:], f1s_data_list[i], label=legend_list[i], color=color_list[i])

    plt.xlim(left=hours[1], right=end_date)
    plt.ylim(bottom=y_bottom, top=y_top)

    # 補助線の有無
    plt.grid(False)

    # 凡例
    plt.legend(loc=(0.7, 0.05), fontsize=17)

    # 黒い枠線を追加
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # 軸ラベルを設定
    #plt.xlabel("Date", fontsize=15)
    plt.ylabel(f1s_label, fontsize=19)

    # 横軸のフォーマットとメモリ間隔を設定、日付を斜めに表示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    plt.xticks([start_date] + list(hours[::interval_hours]), rotation=15)

    # 補助メモリの向き
    plt.tick_params(direction='in')

    # 余白
    plt.subplots_adjust(left=0.1, right=0.92, bottom=0.14, top=0.95)

    # PDFとPNGで出力
    output_pdf_path = save_dir + "/" + f1s_file_name + '.pdf'
    output_png_path = save_dir + "/" + f1s_file_name + '.png'
    plt.savefig(output_pdf_path)
    plt.savefig(output_png_path)

#----------------------------------------------------------------------------------------