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
    flow_id = 0
    for flow_id in flow_box.shape[0]: # axis 0 = flow_id
        if flow_box[flow_id,:,0] == labeled_features[0] and \
                flow_box[flow_id,:,1] == labeled_features[1] and \
                    flow_box[flow_id,:,2] == labeled_features[2]:
                        return True,flow_id # if exist return flow_id
        else: pass
    return False,flow_id+1 # not exist

async def remove_flow(delay, flow_box, flow_id,labeled_features):
    #　時間計測を開始
    await asyncio.sleep(delay)
    # フローに含まれている情報から特徴量を抽出

    # 要素を削除（最後の要素を削除）どこの要素を削除するのか？
    flow_box = np.delete(flow_box,flow_id,axis=0)
    return flow_box

def labeling_features(pkt):

    #IPのプロトコル番号が6の時、プロトコルをtcp
    if pkt[IP].proto == 6:
        protocol = "tcp"   # feat: protocol(str) : "tcp"

    #IPのプロトコル番号が17の時、プロトコルをudp
    elif pkt[IP].proto == 17:
        protocol = "udp"   # feat: protocol(str) : "udp"

    length = int(pkt[IP].len)   # feat: length(int) : 64

    # src : malicious address
    #パケットの送信元IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    if str(pkt[IP].src) in arr_malicious_address_list:
        addr = str(pkt[IP].dst) # feat: addr(str) : "192.0.2.0"
        partner = str(pkt[IP].src)  # feat: partner(str) : "192.168.0.1"
        label = "1"         # feat: label(str)   : "1"
        direction = "snd"       # feat: direction(str) : "snd"
        if protocol == "tcp":
            port = pkt[TCP].sport   # feat: port(int) : 23
            tcpflag = str(pkt[TCP].flags)  # feat: tcpflag(str) : "S"
        elif protocol == "udp":
            port = pkt[UDP].sport
            tcpflag = "-1"

        #summarize関数に値を送る
        summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag)

    # dst : malicious address
    #パケットの宛先IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    elif str(pkt[IP].dst) in arr_malicious_address_list:
        addr = str(pkt[IP].src)
        partner = str(pkt[IP].dst)
        label = "1"
        direction = "rcv"
        if protocol == "tcp":
            port = pkt[TCP].dport
            tcpflag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].dport
            tcpflag = "-1"

        summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag)

    # src : benign address
    #パケットの送信元IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、宛先IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif str(pkt[IP].src) in arr_benign_address_list and str(pkt[IP].dst) not in arr_benign_address_list:
        addr = str(pkt[IP].dst)
        partner = str(pkt[IP].src)
        label = "0"
        direction = "snd"
        if protocol == "tcp":
            port = pkt[TCP].sport
            tcpflag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].sport
            tcpflag = "-1"

        summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag)

    # dst : benign address
    #パケットの宛先IPアドレスが、指定された良性のあるIPアドレスのリストに含まれていて、送信元IPアドレスが指定された良性のあるIPアドレスのリストに含まれていない時
    elif str(pkt[IP].dst) in arr_benign_address_list and str(pkt[IP].src) not in arr_benign_address_list:
        addr = str(pkt[IP].src)
        partner = str(pkt[IP].dst)
        label = "0"
        direction = "rcv"
        if protocol == "tcp":
            port = pkt[TCP].dport
            tcpflag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = pkt[UDP].dport
            tcpflag = "-1"

        summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag)

    return []

#summarize引数
#day=日付, hour=時間, minute=分, seconds=秒, protocol=プロトコル(tcp/udp), length=IPアドレス長?
#addr=宛先アドレス, partner=送信元アドレス, label = 1/0, direction=snd/rcv, port=ポート番号, tcpflag= /-1
def summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag):
    global connecting_address_list
    global connecting_object_list
    global output_list
    global timeout_count

    timeout_count += 1
    #----------------------- first packet(no connection) ---------------------------------
    #最初のパケット(接続なし)

    #変数addrが、リストconnecting_address_listに含まれていない場合
    #connecting_address_listにaddrを追加
    #フローが始まった日時date作成?
    #FlowFeaturesClassのインスタンスを作成し、変数objに格納
    #FlowFeaturesClass = フローごとに作成されるオブジェクトのクラスファイル。
    #このファイルに追記することで抽出する特徴量を増やしたりもできる。alt_val は欠損値の代わりに書き込まれる値。
    if addr not in connecting_address_list:
        connecting_address_list.append(addr)

        date = day + hour + minute
        obj = FlowFeaturesClass(date, addr, partner, label) # create instance

        #drectionがrcv(受信)のとき
        #オブジェクトの属性rcv_packet_count(受信パケット数)を1増やす
        #rcv_tmp_timeに秒を格納
        #rcv_length_list(リスト)にlengthの値を追加
        if direction == "rcv":
            obj.rcv_packet_count += 1
            obj.rcv_tmp_time = seconds
            obj.rcv_length_list.append(length)

        #drectionがsnd(送信)のとき
        #snd_packet_count(送信パケット数)を1増やす
        #snd_tmp_timeに秒を格納
        #snd_length_list(リスト)にlengthの値を追加
        elif direction == "snd":
            obj.snd_packet_count += 1
            obj.snd_tmp_time = seconds
            obj.snd_length_list.append(length)

        #protcol_listにprotocolの値追加
        #port_listにportの値追加
        obj.protocol_list.append(protocol)
        obj.port_list.append(port)

        #last_secondsにsecondsの値追加
        obj.last_seconds = seconds

        #connecting_object_listにオブジェクトobjを追加
        connecting_object_list.append(obj)

    #----------------------- already connecting ------------------------------------------
    #すでに接続しているとき
    #connecting_address_listからaddrがある行の値をaddr_indexに格納
    else:
        addr_index = connecting_address_list.index(addr)

        #rcvのとき
        #connecting_object_listのaddr_indexにあるrcv_packet_count(受信パケット数)を1増やす
        #connecting_object_list[addr_index]のメソッドcalculateTimeIntervalにsecondsとdirectionの値を入れる
        #connecting_object_list[addr_index]のrcv_length_listにlengthの値が追加される
        if direction == "rcv":
            connecting_object_list[addr_index].rcv_packet_count += 1
            connecting_object_list[addr_index].calculateTimeInterval(seconds, direction)
            connecting_object_list[addr_index].rcv_length_list.append(length)

        #sndのとき
        #connecting_object_listのaddr_indexにあるsnd_packet_count(受信パケット数)を1増やす
        #connecting_object_list[addr_index]のメソッドcalculateTimeIntervalにsecondsとdirectionの値を送る
        #connecting_object_list[addr_index]のsnd_length_listにlengthの値が追加される
        elif direction == "snd":
            connecting_object_list[addr_index].snd_packet_count += 1
            connecting_object_list[addr_index].calculateTimeInterval(seconds, direction)
            connecting_object_list[addr_index].snd_length_list.append(length)

        #connecting_object_list[addr_index]の属性protocol_listにprotocolの値追加
        #connecting_object_list[addr_index]の属性port_listにportの値追加
        connecting_object_list[addr_index].protocol_list.append(protocol)
        connecting_object_list[addr_index].port_list.append(port)

        #connecting_object_list[addr_index]の属性last_secondsにsecondsの値代入
        connecting_object_list[addr_index].last_seconds = seconds

    #------------------------ timeout check --------------------------------------
    #タイムアウトになっているか
    # timeout_check_intervalが条件を満たしたら
    if timeout_count == timeout_check_interval:

        output_addr_list = []

        #connecting_address_listの全部に対して繰り返し処理
        #current_time(sniffメソッドを終了するまでの秒数)より小さい場合にoutput_listにconnecting_object_list[l]追加
        #また、output_addr_listにconnecting_object_list[l]のaddrを追加
        for l in range(len(connecting_address_list)):
            if connecting_object_list[l].last_seconds + timeout < current_time:
                output_list.append(connecting_object_list[l])
                output_addr_list.append(connecting_object_list[l].addr)

        #output_addr_listの要素に対してconnecting_address_listからout_addrの値を削除
        for out_addr in output_addr_list:
            connecting_address_list.remove(out_addr)

            #connecting_object_listのコピーに対しての繰り返し処理(a[:]でリストaのコピーを作成するらしい)
        #更にoutput_addr_listに対しての繰り返し処理
        #ob.addrとout_addrが等しいならconnecting_object_listからobの値を削除
        for ob in connecting_object_list[:]:
            for out_addr in output_addr_list:
                if ob.addr == out_addr:
                    connecting_object_list.remove(ob)

        #timeout_countを0にする
        timeout_count = 0
    #-------------------------------------------------------------------------------------

def feature_extract(pkt):
    # パケットから各種特徴量を抽出
    explanatory_variable = []
    return explanatory_variable

def callback(pkt):

    print(pkt.show())
    flow_box = np.array([np.newaxis,np.newaxis,np.newaxis]) # axis [0,flow_id][1,packet_id][2,features]
    feature_matrix = pd.DataFrame() # mainから引っ張ってくるように修正

    # --- IPv4 or not
    if pkt[Ether].type == 0x0800 and \
        pkt[IP].src not in ex_addr_list and \
        pkt[IP].dst not in ex_addr_list and \
        (pkt[IP].proto == 6 or pkt[IP].proto == 17):

        # フローのラベリング、特徴量の抽出
        explanatory_variable = feature_extract(pkt)

        # もしパケットが新しいフローであると判断できる場合
        # 新しいflow_idにlabeled_featuresを追加
        # 追加してから60秒待機 その間他のパケットによる同一フローへの追加を許可する
        # つまりこのコールバック関数を待機状態にしたうえで他のコールバック関数が呼ばれるように処理を一時的に他のプロセスに受け渡す必要がある
        # 待機が終了したらフローから特徴量を抽出する
        flow_exist,flow_id = is_flow_exist(explanatory_variable,flow_box)
        if not flow_exist:
            # 作成したフローは別スレッドで動かす必要がある
            flow_box = np.concatenate([flow_box, explanatory_variable], axis=0)
            remove_flow(3600,flow_box,flow_id,explanatory_variable)
            # フローをラベリング
            labeled_features = labeling_features(pkt)
            # 指定時間後，フローが終了したと判断された場合，特徴量行列に書き込む
            labeled_features_matrix = [-1,labeled_features]
        # もしパケットが以前のフローであると判断できる場合
        else:
            # フローリストの該当フローリストに特徴量配列を追加
            flow_box[flow_id] = explanatory_variable

def write_csv(feature_matrix_list, feature_matrix, output_dir):
    explanatory_variable = pd.concat([feature_matrix, pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)])
    explanatory_variable.to_csv(f"{output_dir}/results_training.csv",index=False)

def online():
    print("online mode")
    while 1:
        sniff(prn = callback, timeout = captime, store = False)

def offline(file_path):
    print("offline mode")
    if os.path.getsize(file_path) == 0:
        print(file_path + ":no data")
    else:
        print(file_path + ":sniffing...")
        sniff(offline = file_path, prn = callback, store = False)
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
                feature_matrix_list = offline(pcap_file_path)
                feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
                feature_matrix.to_csv(f"{outputs_dir_path}/results_training.csv",index=False)
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("file")
            feature_matrix_list = offline(traffic_data_path)
            feature_matrix = pd.DataFrame(feature_matrix_list, columns=feature_matrix.columns)
            feature_matrix.to_csv(f"{outputs_dir_path}/results_training.csv",index=False)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")