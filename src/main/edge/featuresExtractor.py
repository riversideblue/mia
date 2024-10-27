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

def labeling_features(feature):
    # --- if feature seems to be malicious
    if feature == 1:
        labeled_feature = np.concatenate((feature, [1]), axis=1)
    else:
    # --- if feature seems to be benign
        labeled_feature = np.concatenate((feature, [0]), axis=1)
    return labeled_feature

def extract_features_from_packet(pkt, malicious_addresses, benign_addresses):

    # --- Fields
    external_network_address = "external_network_address" # 外部ネットワークのアドレス
    internal_network_address = "internal_network_address" # 内部ネットワークのアドレス
    direction = "direction" # 通信の方向が外部 => 内部なら「snd」，内部 => 外部なら「rcv」 ? ここは疑問が残る
    capture_time_jst = "capture_time_jst" # パケットがキャプチャされた日本標準時間
    protocol = "protocol" # IPレイヤーのアプリケーションプロトコル
    port = "port" # IPレイヤーの使用されているポート
    length = "length" # IPレイヤーのパケット長 ここはEthernetレイヤーじゃなくていいのか？
    tcp_flag = "tcp_flag" #
    label = "label" # 悪性なら1、良性なら0

    src = str(pkt[IP].src)
    dst = str(pkt[IP].dst)
    length = str(pkt[IP].len)
    if pkt[IP].proto == 6:
        protocol = "tcp"
    elif pkt[IP].proto == 17:
        protocol = "udp"

    capture_time_jst = datetime.fromtimestamp(float(pkt.time)).astimezone(timezone(timedelta(hours=9)))

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

    # それ以外のパケットは関係のないものとして破棄
    else:
        print("unrelated packet detected : ignore")
        return None

    return [capture_time_jst,
            external_network_address,
            internal_network_address,
            direction,
            protocol,
            port,
            tcp_flag,
            length,
            label]

def extract_features_from_flow(packets_in_flow):

    print("extract feature")

    # --- rcv/snd count
    rcv_snd_counter = Counter([row[3] for row in packets_in_flow])
    rcv_packet_count = rcv_snd_counter["rcv"]
    snd_packet_count = rcv_snd_counter["snd"]

    # --- tcp/udp count
    tcp_udp_counter = Counter([row[4] for row in packets_in_flow])
    tcp_count = tcp_udp_counter["tcp"]
    udp_count = tcp_udp_counter["udp"]

    # --- most port/count
    port_counter = Counter([row[5] for row in packets_in_flow])
    port_counter_tuple = port_counter.most_common(1)
    most_port = port_counter_tuple[0]
    port_count = port_counter_tuple[1]

    # --- rcv max/min interval
    rcv_time_list = [row[0] for row in packets_in_flow if row[3] == "rcv"]
    rcv_interval = [j - i for i, j in zip(rcv_time_list, rcv_time_list[1:])]
    rcv_max_interval = max(rcv_interval)
    rcv_min_interval = min(rcv_interval)

    # --- rcv max/min length
    rcv_length_list = [row[7] for row in packets_in_flow if row[3] == "rcv"]
    rcv_max_length = max(rcv_length_list)
    rcv_min_length = min(rcv_length_list)

    # --- snd max/min interval
    snd_time_list = [row[0] for row in packets_in_flow if row[3] == "snd"]
    snd_interval = [j - i for i, j in zip(snd_time_list, snd_time_list[1:])]
    snd_max_interval = max(snd_interval)
    snd_min_interval = min(snd_interval)

    # --- snd max/min length
    snd_length_list = [row[7] for row in packets_in_flow if row[3] == "rcv"]
    snd_max_length = max(snd_length_list)
    snd_min_length = min(snd_length_list)

    # --- label
    label = packets_in_flow[0][8]

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

        # flow_box = [
        #  [
        #   [capture_time_jst,external_network_address,internal_network_address,direction,protocol,port,tcp_flag,length,label],
        #   [capture_time_jst,external_network_address,internal_network_address,direction,protocol,port,tcp_flag,length,label],
        #   [capture_time_jst,external_network_address,internal_network_address,direction,protocol,port,tcp_flag,length,label]
        #  ]
        # ]
        self.flow_box = [[[]]]

        self.feature_matrix = [[]]

        # time_manager = {
        # flow_id : pkt_captured_time
        # }
        self.time_manager = {}

    def delete_flow(self,flow_id): # flow_boxからフローを削除

        print(extract_features_from_flow(self.flow_box.pop(flow_id)))
        if not self.feature_matrix[0]:
            self.feature_matrix = extract_features_from_flow(self.flow_box.pop(flow_id))
        else:
            self.feature_matrix.append(extract_features_from_flow(self.flow_box.pop(flow_id)))

    def add_to_new_flow(self, flow_id, field):
        new_flow = [field]
        self.time_manager[flow_id] = field[0]
        if not self.flow_box[0][0]:
            self.flow_box[0] = new_flow
        else:
            self.flow_box.append(new_flow)

    def add_to_existing_flow(self, flow_id, field):
        # 新しいパケットをフローに追加
        new_packet = field
        self.flow_box[flow_id].append(new_packet)

    def is_flow_exist(self, pkt):

        if not self.flow_box[0][0]:
            print("is_flow_exist : flow_box empty")
            return False,0
        else:
            print("is_flow_exist : flow_box not empty")

            # --- is all flow timeout ?
            pkt_captured_time = datetime.fromtimestamp(float(pkt.time)).astimezone(timezone(timedelta(hours=9)))
            for i in range(len(self.time_manager)):
                delta = self.time_manager[i] - pkt_captured_time
                if delta.total_seconds() > float(self.delete_after_seconds):
                    del self.flow_box[i]

            # --- is flow pkt specified existing ?
            for flow_id in range(len(self.flow_box)):
                if self.flow_box[flow_id][0][1] == pkt[IP].src and \
                        self.flow_box[flow_id][0][2] == pkt[IP].dst:
                    print("is_flow_exist : Flow found (src -> dst)")
                    return True,flow_id  # if exist return flow_id
                elif self.flow_box[flow_id][0][1] == pkt[IP].dst and \
                        self.flow_box[flow_id][0][2] == pkt[IP].src:
                    print("is_flow_exist : Flow found (dst -> src)")
                    return True,flow_id  # if exist return flow_id
            print("is_flow_exist : flow not found")
            return False,len(self.flow_box)

    def callback(self,pkt):
        print("----- callback")
        try:
            if Ether in pkt:
                if IP in pkt:
                    if pkt[Ether].type == 0x0800 and \
                        pkt[IP].src not in self.ex_address_list and \
                        pkt[IP].dst not in self.ex_address_list and \
                        (pkt[IP].proto == 6 or pkt[IP].proto == 17):

                        flow_exist,flow_id = self.is_flow_exist(pkt)
                        field = extract_features_from_packet(pkt, self.malicious_address_array,
                                                             self.benign_address_array)

                        if field is not None:
                            if not flow_exist:
                                self.add_to_new_flow(flow_id, field)
                                threading.Timer(60, self.delete_flow, args=(flow_id,)).start()
                            else:
                                self.add_to_existing_flow(flow_id, field)
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
        sniff(offline = file_path, prn=manager.callback, store = False, filter="ip and (tcp or udp)") # storeとは？
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
                feature_matrix_list = flow_manager.flow_box
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
        print(flow_manager.feature_matrix, file=f)
    print(f"処理時間: {elapsed_time}秒")