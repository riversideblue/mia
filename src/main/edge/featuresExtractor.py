import multiprocessing
import threading
import time
import warnings
from collections import Counter

import pandas as pd
import pytz
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.sendrecv import sniff

warnings.simplefilter('ignore')

import numpy as np

from datetime import datetime, timezone, timedelta
import os
import ipaddress
import json

def extract_features_from_packet(pkt, malicious_addresses, benign_addresses):

    src = str(pkt[IP].src)
    dst = str(pkt[IP].dst)
    length = str(pkt[IP].len)
    if pkt[IP].proto == 6:
        protocol = "tcp"
    elif pkt[IP].proto == 17:
        protocol = "udp"
    else:
        protocol = None

    # src : malicious address
    #パケットの送信元IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    if src in malicious_addresses:
        print("- src : malicious")
        direction = "rcv"
        external_network_address = str(pkt[IP].src)
        internal_network_address = str(pkt[IP].dst)
        label = "1"
        if protocol == "tcp":
            port = str(pkt[TCP].sport)
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = str(pkt[UDP].sport)
            tcp_flag = "-1"
        else:
            port = None
            tcp_flag = None

    # dst : malicious address
    # パケットの宛先IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    elif dst in malicious_addresses:
        print("- dst : malicious address")
        direction = "snd"
        external_network_address = str(pkt[IP].dst)
        internal_network_address = str(pkt[IP].src)
        label = "1"
        if protocol == "tcp":
            port = str(pkt[TCP].dport)
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = str(pkt[UDP].dport)
            tcp_flag = "-1"
        else:
            port = None
            tcp_flag = None

    # src : benign address
    # パケットの送信元IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、宛先IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif src in benign_addresses and dst not in benign_addresses:
        print("- src : benign address")
        direction = "rcv"
        external_network_address = str(pkt[IP].src)
        internal_network_address = str(pkt[IP].dst)
        label = "0"
        if protocol == "tcp":
            port = str(pkt[TCP].sport)
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = str(pkt[UDP].sport)
            tcp_flag = "-1"
        else:
            port = None
            tcp_flag = None

    # dst : benign address
    #パケットの宛先IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、送信元IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif dst in benign_addresses and src not in benign_addresses:
        print("- dst : benign address")
        direction = "snd"
        external_network_address = str(pkt[IP].dst)
        internal_network_address = str(pkt[IP].src)
        label = "0"
        if protocol == "tcp":
            port = str(pkt[TCP].dport)
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = str(pkt[UDP].dport)
            tcp_flag = "-1"
        else:
            port = None
            tcp_flag = None

    # それ以外のパケットは関係のないものとして破棄
    else:
        print("unrelated packet detected : ignore")
        return (None,None),None

    return (
        external_network_address, # 外部ネットワークのアドレス
        internal_network_address # 内部ネットワークのアドレス
    ),[
        direction, # 通信の方向が外部 => 内部なら「snd」，内部 => 外部なら「rcv」 ? ここは疑問が残る
        protocol, # IPレイヤーのアプリケーションプロトコル
        port, # IPレイヤーの使用されているポート
        tcp_flag,
        length, # IPレイヤーのパケット長 ここはEthernetレイヤーじゃなくていいのか？
        label # 悪性なら1、良性なら0
    ]

def extract_features_from_flow(packets_in_flow):

    print("extract feature from flow")
    field = packets_in_flow.values()

    # --- rcv/snd count
    rcv_snd_counter = Counter([row[0] for row in field])
    rcv_packet_count = rcv_snd_counter["rcv"]
    snd_packet_count = rcv_snd_counter["snd"]

    # --- tcp/udp count
    tcp_udp_counter = Counter([row[1] for row in field])
    tcp_count = tcp_udp_counter["tcp"]
    udp_count = tcp_udp_counter["udp"]

    # --- most port/count
    port_counter = Counter([row[2] for row in field])
    port_counter_tuple = port_counter.most_common(1)
    most_port = port_counter_tuple[0][0]
    port_count = port_counter_tuple[0][1]

    # --- rcv max/min interval
    rcv_time_list = [key for key, value in packets_in_flow.items() if value[0] == "rcv"]
    if len(rcv_time_list) > 1:
        print(rcv_time_list)
        rcv_interval = [j - i for i, j in zip(rcv_time_list, rcv_time_list[1:])]
        print(rcv_interval)
        rcv_max_interval = max(rcv_interval)
        rcv_min_interval = min(rcv_interval)
    else:
        print("element only 1")
        rcv_max_interval = rcv_min_interval = 60  # デフォルト値

    # --- rcv max/min length
    rcv_length_list = [row[4] for row in field if row[3] == "rcv"]
    rcv_max_length = max(rcv_length_list)
    rcv_min_length = min(rcv_length_list)

    # --- snd max/min interval
    snd_time_list = [key for key, value in packets_in_flow.items() if value[0] == "snd"]
    if len(snd_time_list) > 1:
        snd_interval = [j - i for i, j in zip(snd_time_list, snd_time_list[1:])]
        snd_max_interval = max(snd_interval)
        snd_min_interval = min(snd_interval)
    else:
        snd_max_interval = snd_min_interval = 60  # デフォルト値

    # --- snd max/min length
    snd_length_list = [row[4] for row in field if row[3] == "rcv"]
    snd_max_length = max(snd_length_list)
    snd_min_length = min(snd_length_list)

    # --- label
    label = field[0][5]

    return [
        rcv_packet_count,
        snd_packet_count,
        tcp_count,
        udp_count,
        most_port,
        port_count,
        rcv_max_interval,
        rcv_min_interval,
        rcv_max_length,
        rcv_min_length,
        snd_max_interval,
        snd_min_interval,
        snd_max_length,
        snd_min_length,
        label
    ]

class FlowManager:

    def __init__(self, eal, maa, baa, das):

        self.ex_address_list = eal
        self.malicious_address_array = maa
        self.benign_address_array = baa
        self.delete_after_seconds = das
        manager = multiprocessing.Manager()
        self.flow_manager = manager.dict()
        self.featured_flow_matrix = manager.dict()

    def delete_flow(self,key): # flow_boxからフローを削除
        flow = self.flow_manager.pop(key)
        print("flow")
        print(flow)
        print(type(flow))
        self.featured_flow_matrix[key[0]] = extract_features_from_flow(flow)

    def is_flow_exist(self, captured_time, src, dst):

        if not self.flow_manager:
            return None
        else:
            # --- is all flow timeout ?
            delete_key_list = []
            for key in self.flow_manager:
                if key[0] - captured_time > float(self.delete_after_seconds):
                    print("key delete")
                    delete_key_list.append(key)
                else:
                    print("break")
                    break
            for key in delete_key_list:
                del self.flow_manager[key]

            # --- is flow pkt specified existing ?
            for key in self.flow_manager:
                if key[1] == src and \
                    key[2] == dst:
                    print("is_flow_exist : Flow found (src -> dst)")
                    return key  # if exist return key
                elif key[1] == dst and \
                        key[2] == src:
                    print("is_flow_exist : Flow found (src -> dst)")
                    return key  # if exist return key
            print("is_flow_exist : flow not found")
            return None

    def callback(self,pkt):
        print("----- callback")
        try:
            if Ether in pkt:
                if IP in pkt:
                    if pkt[Ether].type == 0x0800 and \
                        pkt[IP].src not in self.ex_address_list and \
                        pkt[IP].dst not in self.ex_address_list and \
                        (pkt[IP].proto == 6 or pkt[IP].proto == 17):

                        key = self.is_flow_exist(float(pkt.time), str(pkt[IP].src), str(pkt[IP].dst))
                        addr,field = extract_features_from_packet(pkt, self.malicious_address_array,
                                                             self.benign_address_array)
                        if field is not None:
                            if key is None:
                                new_flow_key = (float(pkt.time),str(addr[0]),str(addr[1]))
                                self.flow_manager[new_flow_key]={ new_flow_key[0] : field }
                                print(self.flow_manager)
                                threading.Timer(60,lambda: self.delete_flow(new_flow_key)).start()
                            else:
                                self.flow_manager[key][float(pkt.time)] = field
                                print(self.flow_manager)
                else:
                    pass
            else:
                pass
        except IndexError:
            pass

def online(manager):
    print("- online mode")
    while 1:
        sniff(prn=manager.callback, timeout = captime, store = False, filter="ip and (tcp or udp)")

def offline(file_path, manager):
    print("- offline mode")
    if os.path.getsize(file_path) == 0:
        print(os.path.basename(file_path) + ":no data")
    else:
        try:
            sniff(offline=file_path, prn=manager.callback, store=False, filter="ip and (tcp or udp)")
        except Exception as e:
            print(f"Error reading packet: {e}")
        print(os.path.basename(file_path) + ":finish")


# 与えられたフォルダまたはファイルに含まれるトラヒックデータ（pcapファイル）から特徴量を抽出してcsvファイルに書き込む
if __name__ == "__main__":

    start_time = time.time()  # 処理開始時刻

    # --- Load settings
    settings = json.load(open('src/main/edge/settings.json', 'r'))

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- Field

    online_mode = settings["FeatureExtract"]["ONLINE_MODE"]
    traffic_data_path = settings["FeatureExtract"]["TRAFFIC_DATA_PATH"]
    feature_matrix_column = np.array(settings["FeatureExtract"]["EXPLANATORY_VARIABLE_COLUMN"])
    pcap_fl_id = settings["FeatureExtract"]["PCAP_FILE_IDENTIFIER"]  # pcap file identifier (例: ".pcap")
    subdir_year_month = settings["FeatureExtract"]["SUBDIRECTORY_YEAR_MONTH"]
    subdir_start_d = settings["FeatureExtract"]["SUBDIRECTORY_ELAPSED_DAY"]  # subdir start day

    timeout = settings["FeatureExtract"]["FLOW_TIMEOUT"]  # connection timeout [seconds]
    timeout_check_interval = settings["FeatureExtract"]["PACKET_INTERVAL_CHECK_TIMEOUT"]  # check timeout at N packets intervals

    malicious_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["MALICIOUS"]
    malicious_address_list = []
    for net in malicious_network_address_list:
        for malicious in ipaddress.ip_network(net):
            malicious_address_list.append(str(malicious))
    malicious_address_array = np.array(malicious_address_list)  # list to ndarray

    benign_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["BENIGN"]
    benign_address_list = []
    for net in benign_network_address_list:
        for benign in ipaddress.ip_network(net):
            benign_address_list.append(str(benign))
    benign_address_array = np.array(benign_address_list)

    ex_addr_list = settings["FeatureExtract"]["NetworkAddress"]["EXCEPTION"]

    captime = settings["FeatureExtract"]["CAPTURE_TIME"]  # capture time for online mode [seconds]
    pcap_saved_dir_name = settings["FeatureExtract"]["PCAP_FILES_SAVED_DIR_NAME"]
    delete_after_seconds = settings["FeatureExtract"]["DELETE_AFTER_SECONDS"]

    # --- Create output directory for csv
    outputs_dir_path: str = f"src/main/edge/outputs/{init_time}_extracted/features"
    os.makedirs(outputs_dir_path)

    # --- Set results DataFrame
    feature_matrix = pd.DataFrame(columns=feature_matrix_column)
    feature_matrix_list = feature_matrix.values

    # --- build constractor
    flow_manager = FlowManager(ex_addr_list,malicious_address_array,benign_address_array,delete_after_seconds)

    # --- Online mode
    if online_mode:
        online(flow_manager)

    # --- Offline mode
    # --- データセット内のpcapファイルごとに特徴量抽出を行う
    else:
        if os.path.isdir(traffic_data_path):
            print("folder")
            for pcap_file in os.listdir(traffic_data_path):
                print("-----" + pcap_file + " found")
                pcap_file_path:str = os.path.join(traffic_data_path,pcap_file)
                offline(pcap_file_path,flow_manager)
                feature_matrix_list = flow_manager.featured_flow_matrix
                feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
                feature_matrix.to_csv(f"{outputs_dir_path}/{pcap_file}.csv",index=False)
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("-----" + os.path.basename(traffic_data_path) + " found")
            offline(traffic_data_path,flow_manager)

            # feature_matrix_list = flow_manager.flow_box
            # feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
            # feature_matrix.to_csv(f"{outputs_dir_path}/{os.path.splitext(os.path.basename(traffic_data_path))[0]}.csv",index=False)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")
    end_time = time.time()  # 処理終了時刻
    elapsed_time = end_time - start_time
    with open("output.txt", "w") as f:
        print(flow_manager.featured_flow_matrix, file=f)
    print(f"処理時間: {elapsed_time}秒")