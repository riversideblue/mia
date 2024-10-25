import threading
import time
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

def extract_features_from_packet(pkt,malicious_address_array,benign_address_array):

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

    capture_time_jst = datetime.fromtimestamp(float(pkt.time)).astimezone(timezone(timedelta(hours=9))).strftime("%Y%m%d%H%M%S%f")
    print("pkt.time")
    print(pkt.time)
    # src : malicious address
    #パケットの送信元IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    if src in malicious_address_array:
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
    elif dst in malicious_address_array:
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
    elif src in benign_address_array and dst not in benign_address_array:
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
    elif dst in benign_address_array and src not in benign_address_array:
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

    print("ctj :"+str(capture_time_jst))
    print("external_network_address :"+external_network_address)
    print("internal_network_address :"+internal_network_address)
    print("direction :"+direction)
    print("port :"+str(port))
    print("tf :"+tcp_flag)
    print("len :"+length)
    print("proto :"+protocol)
    print("label :"+label)

    return [str(capture_time_jst),
            external_network_address,
            internal_network_address,
            direction,
            port,
            tcp_flag,
            length,
            protocol,
            label]

def extract_features_from_flow(flow_box,flow_id):
    print("extract feature")

    return [0,1,2,3,4,5]

class FlowManager:

    def __init__(self, eal, maa, baa, das):

        self.ex_address_list = eal
        self.malicious_address_array = maa
        self.benign_address_array = baa
        self.delete_after_seconds = das

        # --- extract_features_from_packetにからのパケットを渡しヘッダを回収する
        # empty_packet = Ether()
        # packet_field = np.array(extract_features_from_packet(empty_packet,maa,baa))
        self.flow_box = [[[]]]

    def delete_flow(self,flow_id): # flow_boxからフローを削除
        extract_features_from_flow(self.flow_box,flow_id)
        self.flow_box = np.delete(self.flow_box,flow_id,axis=0)

    def add_new_flow(self,field):
        new_flow = [field]
        if not self.flow_box[0][0]:
            # flow_boxが初期状態の場合
            print("add_new_flow : flow_box empty")
            self.flow_box[0] = new_flow
        else:
            print("add_new_flow : there is some flow")
            self.flow_box.append(new_flow)
            threading.Timer(self.delete_after_seconds, self.delete_flow).start()

    def add_new_packet(self, flow_id, field):
        # 新しいパケットをフローに追加
        new_packet = field
        self.flow_box[flow_id].append(new_packet)
        print(f"add_new_flow : 保存されているフローの数: {len(self.flow_box)}")


    def is_flow_exist(self, pkt):

        # flow_boxが空であることを確認する
        # 空であればFalseと
        # flow_idごとに0番目のパケットの1番目(外部ネットワークアドレス)と2番目(内部ネットワークアドレス)がパケットと一致するかどうか確認
        # 一致すればTrueとその時のflow_idを返す
        # 一致しなければFalseと最後尾のflow_idを返す
        # print(f"is_flow_exist : 保存されているフローの数: {len(self.flow_box)}")
        # print(f"is_flow_exist : 0番目のフローに含まれているパケット数: {len(self.flow_box[0])}")
        # print(f"is_flow_exist : 0番目のフローに含まれている0番目のパケットに含まれているフィールドの数: {len(self.flow_box[0][0])}")
        if not self.flow_box[0][0]:
            print("is_flow_exist : flow_box empty")
            return False,0
        else:
            print("is_flow_exist : flow_box not empty")
            for flow_id in range(len(self.flow_box)):  # axis 0 = flow_id
                # print(f"is_flow_exist : Checking flow_id {flow_id}")
                # print(f"Ex_addr in box : {self.flow_box[flow_id][0][1]}")
                # print(f"pkt_src : {pkt[IP].src}")
                # print(f"pkt_dst : {pkt[IP].dst}")
                # print(f"In_addr in box : {self.flow_box[flow_id][0][2]}")
                # print(f"pkt_src : {pkt[IP].src}")
                # print(f"pkt_dst : {pkt[IP].dst}")
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

                        # もしパケットが新しいフローであると判断できる場合
                        # 新しいflow_idにlabeled_featuresを追加
                        # 追加してから60秒待機 その間他のパケットによる同一フローへの追加を許可する
                        # つまりこのコールバック関数を待機状態にしたうえで他のコールバック関数が呼ばれるように処理を一時的に他のプロセスに受け渡す必要がある
                        # 待機が終了したらフローから特徴量を抽出する
                        # print("- is_flow_exist")
                        flow_exist,flow_id = self.is_flow_exist(pkt)
                        # print("- extract_features_from_packet")
                        field = extract_features_from_packet(pkt, self.malicious_address_array,
                                                             self.benign_address_array)  # パケットからフィールドを抽出
                        # フローボックスが空の場合 flow_id = 0
                        # フローが存在する場合 flow_id = フローが存在するところ
                        # フローが存在しなかった場合 flow_id 新しいフローの番号
                        if field is not None:
                            if not flow_exist:
                                # print("- add_new_flow")
                                self.add_new_flow(field)
                            # もしパケットが以前のフローであると判断できる場合
                            else:
                                # print("- add_new_packet")
                                self.add_new_packet(flow_id,field)
                else:
                    pass
            else:
                pass
        except IndexError:
            pass
        # print("callback process end : flow_box ↓")
        # print(self.flow_box)

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
        print(flow_manager.flow_box, file=f)
    print(f"処理時間: {elapsed_time}秒")