import warnings

import pandas as pd
import pytz
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.sendrecv import sniff

warnings.simplefilter('ignore')

import numpy as np

from datetime import datetime
import os
import ipaddress
import json


def callback_online(packet):
    if packet[Ether].type == 0x0800 and \
            packet[IP].src not in ex_addr_list and \
            packet[IP].dst not in ex_addr_list and \
            (packet[IP].proto == 6 or packet[IP].proto == 17):
        print(packet.summuerize())
    return

def callback_offline(packet):
    return



def write_csv(explanatory_variable_list, explanatory_variable, output_dir):
    explanatory_variable = pd.concat([explanatory_variable, pd.DataFrame(explanatory_variable_list, columns=explanatory_variable.columns)])
    explanatory_variable.to_csv(f"{output_dir}/results_training.csv",index=False)

def online():
    print("online mode")
    while 1:
        sniff(prn = callback_online, timeout = captime, store = False)

def offline(file_path):
            if os.path.getsize(file_path) == 0:
                print(file_path + ":no data")
            else:
                print(file_path + ":sniffing...")
                sniff(offline = file_path, prn = callback_offline, store = False)
                print(file_path + ":finish")

# 与えられたフォルダまたはファイルに含まれるトラヒックデータpcapから特徴量を抽出してcsvファイルに書き込む
if __name__ == "__main__":

    # --- Load settings
    settings = json.load(open('src/main/edge/settings.json', 'r'))

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- Field
    online_mode = settings["FeatureExtract"]["ONLINE_MODE"]
    traffic_data_path = settings["FeatureExtract"]["TRAFFIC_DATA_PATH"]
    explanatory_variable_column = settings["FeatureExtract"]["EXPLANATORY_VARIABLE_COLUMN"]
    pcap_fl_id = settings["FeatureExtract"]["PCAP_FILE_IDENTIFIER"]  # pcap file identifier (例: ".pcap")
    subdir_year_month = settings["FeatureExtract"]["SUBDIRECTORY_YEAR_MONTH"]
    subdir_start_d = settings["FeatureExtract"]["SUBDIRECTORY_ELAPSED_DAY"]  # subdir start day

    timeout = settings["FeatureExtract"]["FLOW_TIMEOUT"]  # connection timeout [seconds]
    timeout_check_interval = settings["FeatureExtract"]["PACKET_INTERVAL_CHECK_TIMEOUT"]  # check timeout at N packets intervals

    malicious_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["MALICIOUS"]
    benign_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["BENIGN"]
    ex_addr_list = settings["FeatureExtract"]["NetworkAddress"]["EXCEPTION"]

    captime = settings["FeatureExtract"]["CAPTURE_TIME"]  # capture time for online mode [seconds]
    pcap_saved_dir_name = settings["FeatureExtract"]["PCAP_FILES_SAVED_DIR_NAME"]

    # --- Create output directory for csv
    outputs_dir_path: str = f"src/main/edge/outputs/{init_time}_extracted/features"
    os.makedirs(outputs_dir_path)

    # --- Set results DataFrame
    explanatory_variables = pd.DataFrame(columns=explanatory_variable_column)
    explanatory_variable_list = explanatory_variables.values

    malicious_address_list = []
    for net in malicious_network_address_list:
        for malicious in ipaddress.ip_network(net):
            malicious_address_list.append(str(malicious))
    arr_malicious_address_list = np.array(malicious_address_list)  # list to ndarray

    benign_address_list = []
    for net in benign_network_address_list:
        for benign in ipaddress.ip_network(net):
            benign_address_list.append(str(benign))
    arr_benign_address_list = np.array(benign_address_list)

    first_packet_flag = False
    now_csv_file_name = ""

    connecting_address_list = []
    connecting_object_list = []
    output_list = []

    timeout_count = 0
    current_time = 0

    packet_list = []
    termination_flag = False
    start_datetime = ""

    # --- Online mode
    if online_mode:
        online()

    # --- Offline mode
    else:
        if os.path.isdir(traffic_data_path):
            print("folder")
            for pcap_file in os.listdir(traffic_data_path):
                pcap_file_path:str=os.path.join(traffic_data_path,pcap_file)
                offline(pcap_file_path)
                write_csv(explanatory_variable_list=explanatory_variable_list,explanatory_variable=explanatory_variables,output_dir=outputs_dir_path)
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("file")
            offline(traffic_data_path)
            write_csv(explanatory_variable_list=explanatory_variable_list,explanatory_variable=explanatory_variables,output_dir=outputs_dir_path)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")