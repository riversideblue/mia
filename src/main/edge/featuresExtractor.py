import threading
import warnings

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

def extract_features_from_packet(pkt):

    # --- Fields
    capture_time_jst = datetime.fromtimestamp(pkt.time).astimezone(timezone(timedelta(hours=9)))
    protocol = ""
    src = str(pkt[IP].src)
    dst = str(pkt[IP].dst)
    label = ""
    direction = ""
    port = 0
    tcp_flag = ""
    length = int(pkt[IP].len)   # feat: length(int) : 64

    #IPのプロトコル番号が6の時、プロトコルをtcp
    if pkt[IP].proto == 6:
        protocol = "tcp"   # feat: protocol(str) : "tcp"
    #IPのプロトコル番号が17の時、プロトコルをudp
    elif pkt[IP].proto == 17:
        protocol = "udp"   # feat: protocol(str) : "udp"

    # src : malicious address
    #パケットの送信元IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    if src in arr_malicious_address_list:
        direction = "snd"
        label = "1"         # feat: label(str)   : "1"
        if protocol == "tcp":
            port = pkt[TCP].sport   # feat: port(int) : 23
            tcp_flag = str(pkt[TCP].flags)  # feat: tcpflag(str) : "S"
        elif protocol == "udp":
            port = pkt[UDP].sport
            tcp_flag = "-1"

    # dst : malicious address
    #パケットの宛先IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    elif dst in arr_malicious_address_list:
        direction = "rcv"
        label = "1"
        if protocol == "tcp":
            port = pkt[TCP].dport
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].dport
            tcp_flag = "-1"

    # src : benign address
    #パケットの送信元IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、宛先IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif src in arr_benign_address_list and dst not in arr_benign_address_list:
        direction = "snd"
        label = "0"
        if protocol == "tcp":
            port = pkt[TCP].sport
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].sport
            tcp_flag = "-1"

    # dst : benign address
    #パケットの宛先IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、送信元IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif dst in arr_benign_address_list and src not in arr_benign_address_list:
        direction = "rcv"
        label = "0"
        if protocol == "tcp":
            port = pkt[TCP].dport
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].dport
            tcp_flag = "-1"

            print(capture_time_jst,
                  src,
                  dst,
                  direction,
                  port,
                  tcp_flag,
                  length,
                  protocol,
                  label)

    return (capture_time_jst,
            src,
            dst,
            direction,
            port,
            tcp_flag,
            length,
            protocol,
            label)

def extract_features_from_flow(pkt):
    print("extract feature")

    return [0,1,2,3,4,5]

class FlowManager:

    def __init__(self, eal, das, fml):
        self.ex_address_list = eal
        self.delete_after_seconds = das
        self.flow_box = fml[np.newaxis,np.newaxis,:]
        a = np.array(object=[],ndmin=3)
        print(self.flow_box)
        print(a)


    def add_new_flow(self):
        new_flow = np.array(self.flow_box[0,0,:],ndmin=3)
        self.flow_box = np.concatenate((self.flow_box,new_flow),axis=0) # 新しいフローを配列に追加、番号を記憶
        threading.Timer(self.delete_after_seconds, self.delete_flow).start()

    def delete_flow(self,flow_id): # flow_boxからフローを削除
        self.flow_box = np.delete(self.flow_box,flow_id,axis=0)

    def add_new_packet(self, flow_id, pkt):
        # 新しいパケットをフローに追加
        feature = extract_features_from_packet(pkt)
        self.flow_box = np.concatenate((self.flow_box[flow_id],feature),axis=0) # 新しいフローをキューに追加、番号を記憶

    def is_flow_exist(self, pkt):
        flow_id = 0
        for flow_id in range(self.flow_box.shape[0]):  # axis 0 = flow_id
            if self.flow_box[flow_id, :, 0] == pkt[IP].src and \
                    self.flow_box[flow_id, :, 1] == pkt[IP].dst:
                return True, flow_id  # if exist return flow_id
            else:
                pass
        return False, flow_id + 1  # not exist

    def callback(self,pkt):
        if pkt[Ether].type == 0x0800 and \
            pkt[IP].src not in self.ex_address_list and \
            pkt[IP].dst not in self.ex_address_list and \
            (pkt[IP].proto == 6 or pkt[IP].proto == 17):
            print("OK")

            # もしパケットが新しいフローであると判断できる場合
            # 新しいflow_idにlabeled_featuresを追加
            # 追加してから60秒待機 その間他のパケットによる同一フローへの追加を許可する
            # つまりこのコールバック関数を待機状態にしたうえで他のコールバック関数が呼ばれるように処理を一時的に他のプロセスに受け渡す必要がある
            # 待機が終了したらフローから特徴量を抽出する

            flow_exist,flow_id = self.is_flow_exist(pkt)
            if not flow_exist:
                print("flow_non-exist")
                self.add_new_flow()
                self.add_new_packet(flow_id,pkt)
            # もしパケットが以前のフローであると判断できる場合
            else:
                print("flow_exist")
                self.add_new_packet(flow_id,pkt)

def online(flow_manager):
    print("online mode")
    while 1:
        sniff(prn=flow_manager.callback, timeout = captime, store = False)

def offline(file_path,flow_manager):
    print("offline mode")
    if os.path.getsize(file_path) == 0:
        print(file_path + ":no data")
    else:
        print(file_path + ":sniffing...")
        sniff(offline = file_path, prn=flow_manager.callback, store = False)
        print(file_path + ":finish")
    return feature_matrix_list

# パターン1
# settings.jsonを読み込む done
# featuresExtractor一回の実行処理に対してひとつのfeature_matrix=pd.DataFrameを定義 done
# 処理高速化のためDataFrameからvalueを抽出feature_matrix_list:np.arrayを定義 done
# ディレクトリかファイルかを判断，ディレクトリならディレクトリ内に含まれるすべてのpcapファイルを読み込む done
# 読み込んだpcapファイルについて，すべての（フローごと？）のパケットのデータを読み込む
# それぞれパケットデータに対して必要な特徴量を抽出し，一時的に配列に登録する(explanatory_variable_array)
# それぞれのパケットデータに対してラベル付けを行う(target_variable)
# explanatory_variable_arrayとtarget_variableを合わせてfeature_matrix_arrayに登録
# 読み込んだpcapファイルについて特徴量の抽出が完了したら，feature_matrix_listをまとめてfeature_matrixに登録
# feature_matrixをもとにcsvファイルを出力
# すべてのpcapファイルの読み込みが終わるまで継続

# パターン2 とりあえずこっち　上手くいかなかったらパターン1に変更
# settings.jsonを読み込む done
# featuresExtractor一回の実行処理に対してひとつのfeature_matrix=pd.DataFrameを定義 done
# 処理高速化のためDataFrameからvalueを抽出feature_matrix_list:np.arrayを定義 done
# ディレクトリかファイルかを判断，ディレクトリならディレクトリ内に含まれるすべてのpcapファイルを読み込む done
# 読み込んだpcapファイルについてパケットをひとつづつ読み込む done
# 読み込んだパケットをフローごとにまとめる done
# フローごとのパケットデータに対して必要な特徴量を抽出し，一時的に配列に登録する(explanatory_variable_array)　done
# それぞれのパケットデータに対してラベル付けを行う(target_variable) done
# explanatory_variable_arrayとtarget_variableを合わせてfeature_matrix_arrayに登録
# すべてのpcapファイルの読み込みが終わるまで継続
# すべてのpcapファイルについて特徴量の抽出が完了したら，feature_matrix_listをまとめてfeature_matrixに登録
# feature_matrixをもとにcsvファイルを出力
# とてつもない大きさの配列リストを保持し続ける必要があるという懸念 DataFrameで保持する必要はあるのか
# しかし全ての特徴量が同一csvファイルにあるのは便利，実環境に近い

# 与えられたフォルダまたはファイルに含まれるトラヒックデータ（pcapファイル）から特徴量を抽出してcsvファイルに書き込む
if __name__ == "__main__":

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
    arr_malicious_address_list = np.array(malicious_address_list)  # list to ndarray

    benign_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["BENIGN"]
    benign_address_list = []
    for net in benign_network_address_list:
        for benign in ipaddress.ip_network(net):
            benign_address_list.append(str(benign))
    arr_benign_address_list = np.array(benign_address_list)

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
    print("feature_matrix")
    print(feature_matrix_list)

    flow_manager = FlowManager(ex_addr_list,delete_after_seconds,feature_matrix_column)

    # --- Online mode
    if online_mode:
        online(flow_manager)

    # --- Offline mode
    # --- データセット内のpcapファイルごとに特徴量抽出を行う
    else:
        if os.path.isdir(traffic_data_path):
            print("folder")
            for pcap_file in os.listdir(traffic_data_path):
                pcap_file_path:str=os.path.join(traffic_data_path,pcap_file)
                feature_matrix_list = offline(pcap_file_path,flow_manager)
                feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
                feature_matrix.to_csv(f"{outputs_dir_path}/{pcap_file}.csv",index=False)
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("file")
            feature_matrix_list = offline(traffic_data_path,flow_manager)
            feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
            feature_matrix.to_csv(f"{outputs_dir_path}/{os.path.splitext(os.path.basename(traffic_data_path))[0]}.csv",index=False)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")