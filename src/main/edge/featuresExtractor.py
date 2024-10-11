import asyncio
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

def is_flow_exist(labeled_features, flow_box):
    for i in flow_box.shape(axis=0): # i番目のフローに特徴量が一致するパケットがあるかどうか
        if flow_box[i,:,0] == labeled_features[0] and \
                flow_box[i,:,1] == labeled_features[1] and \
                    flow_box[i,:,2] == labeled_features[2]:
            return True
        else: pass
    return False

async def remove_flow(delay, flow_box, flow_id,labeled_features):
    #　時間計測を開始
    await asyncio.sleep(delay)
    # 要素を削除（最後の要素を削除）どこの要素を削除するのか？
    flow_box = np.delete(flow_box,flow_id,axis=0)
    return flow_box

def feature_extract_from_flow():

    return

def callback_online(packet):

    # --- Field
    flow_box = np.array([np.newaxis,np.newaxis,np.newaxis]) # axis [0,flow_id][1,packets][2,features]
    flow_id = np.array([np.newaxis,np.newaxis])

    # --- IPv4 or not
    if packet[Ether].type == 0x0800 and \
            packet[IP].src not in ex_addr_list and \
            packet[IP].dst not in ex_addr_list and \
            (packet[IP].proto == 6 or packet[IP].proto == 17):
        print(packet.summuerize())

        # 特徴情報を抽出
        explanatory_variable_array = ["","",""]

        # パケットのラベル付けを行う(これはフロー確定後でも良いかも)
        target_variable = [0]

        # ラベル付けされた特徴量
        labeled_features = ["","","",""]

        # フロー番号，パケット番号，特徴量リストをもつ3次元配列を定義

        # もしパケットが新しいフローであると判断できる場合
        if not is_flow_exist(labeled_features,flow_box):
            flow_box = np.concatenate([flow_box, labeled_features], axis=0)
            flow_id = [-1,flow_box.shape[0]]
            remove_flow(3600,flow_box,flow_id,labeled_features)

        # もしパケットが以前のフローであると判断できる場合
        else:
        # フローリストの該当フローリストに特徴量配列を追加
            flow_box[1].add(explanatory_variable_array)
        # 指定時間後，フローが終了したと判断された場合
        # まとまった状態で特徴量を抽出する

    # flow_boxを次の処理に渡す必要がある
    return

def callback_offline(packet):
    return



def write_csv(feature_matrix_list, feature_matrix, output_dir):
    explanatory_variable = pd.concat([feature_matrix, pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)])
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
# とてつもない大きさの配列リストを保持し続ける必要があるという懸念
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
    feature_matrix_column = settings["FeatureExtract"]["EXPLANATORY_VARIABLE_COLUMN"]
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
    feature_matrix = pd.DataFrame(columns=feature_matrix_column)
    feature_matrix_list = feature_matrix.values

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
                write_csv(feature_matrix_list=feature_matrix_list,feature_matrix=feature_matrix,output_dir=outputs_dir_path)
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("file")
            offline(traffic_data_path)
            write_csv(feature_matrix_list=feature_matrix_list,feature_matrix=feature_matrix,output_dir=outputs_dir_path)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")