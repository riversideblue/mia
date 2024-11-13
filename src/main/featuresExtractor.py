import time
import warnings

import pandas as pd
import pytz
from scapy.layers.inet import IP, TCP, UDP
from scapy.sendrecv import sniff

warnings.simplefilter('ignore')

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
    port = None
    tcp_flag = None

    # src : malicious address
    # パケットの送信元IPアドレスが、指定された悪意のあるIPアドレスのリストに含まれている時
    if src in malicious_addresses:
        direction = "rcv"
        external_network_address = src
        internal_network_address = dst
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
        direction = "snd"
        external_network_address = dst
        internal_network_address = src
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
        direction = "rcv"
        external_network_address = src
        internal_network_address = dst
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
        direction = "snd"
        external_network_address = dst
        internal_network_address = src
        label = "0"
        if protocol == "tcp":
            port = str(pkt[TCP].dport)
            tcp_flag = str(pkt[TCP].flags)
        elif protocol == "udp":
            port = str(pkt[UDP].dport)
            tcp_flag = "-1"

    # それ以外のパケットは関係のないものとして破棄
    else:
        return (None,None),None

    return (
        external_network_address, # 外部ネットワークのアドレス
        internal_network_address # 内部ネットワークのアドレス
    ),[
        direction, # 通信の方向が外部 => 内部なら「snd」，内部 => 外部なら「rcv」 ? ここは疑問が残る
        protocol, # IPレイヤーのアプリケーションプロトコル
        port, # IPレイヤーの使用されているポート
        tcp_flag, # TCPフラグ
        length, # IPレイヤーのパケット長 ここはEthernetレイヤーじゃなくていいのか？
        label # 悪性なら1、良性なら0
    ]

def extract_features_from_flow(packets_in_flow):

    rcv_packet_count = 0
    snd_packet_count = 0
    tcp_count = 0
    udp_count = 0
    most_port = None
    port_count = 0
    rcv_max_interval = rcv_min_interval = None
    rcv_max_length = rcv_min_length = 0
    snd_max_interval = snd_min_interval = None
    snd_max_length = snd_min_length = 0
    label = None

    port_freq = {}
    last_rcv_time = last_snd_time = None

    for key, field in packets_in_flow.items():

        if field[0] == "rcv":

            # --- rcv_count
            rcv_packet_count += 1

            # --- rcv_max/min_interval
            if last_rcv_time is not None:
                rcv_interval = key - last_rcv_time
                if rcv_max_interval is None or rcv_max_interval < rcv_interval:
                    rcv_max_interval = rcv_interval
                if rcv_min_interval is None or rcv_max_interval > rcv_interval:
                    rcv_min_interval = rcv_interval
            last_rcv_time = key

            # --- rcv_max/min_length
            rcv_length = int(field[4])
            if rcv_max_length is None or rcv_length > rcv_max_length:
                rcv_max_length = rcv_length
            if rcv_min_length is None or rcv_length < rcv_min_length:
                rcv_min_length = rcv_length

        elif field[0] == "snd":

            # --- snd_count
            snd_packet_count += 1

            # --- snd_max/min_interval
            if last_snd_time is not None:
                snd_interval = key - last_snd_time
                if snd_max_interval is None or snd_interval > snd_max_interval:
                    snd_max_interval = snd_interval
                if snd_min_interval is None or snd_interval < snd_min_interval:
                    snd_min_interval = snd_interval
            last_snd_time = key

            # --- rcv_max/min_length
            snd_length = int(field[4])
            if snd_length > snd_max_length:
                snd_max_length = snd_length
            if snd_length < snd_min_length:
                snd_min_length = snd_length

        # --- tcp/udp count
        if field[1] == "tcp":
            tcp_count += 1
        elif field[1] == "udp":
            udp_count += 1

        # --- most port / port count
        port = field[2]
        if port in port_freq:
            port_freq[port] += 1
        else:
            port_freq[port] = 1
        if port_freq[port] > port_count:
            most_port = port
            port_count = port_freq[port]

        # --- label
        if label is None:
            label = field[5]

    rcv_max_interval = rcv_max_interval if rcv_max_interval is not None else 60
    rcv_min_interval = rcv_min_interval if rcv_min_interval is not None else 60
    snd_max_interval = snd_max_interval if snd_max_interval is not None else 60
    snd_min_interval = snd_min_interval if snd_min_interval is not None else 60

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

    def __init__(self, eal, mas, bas, ft):

        self.ex_address_list = eal
        self.malicious_address_set = mas
        self.benign_address_set = bas
        self.flow_timeout = ft
        # flow_manager = {
        #     (Flow registered time(unix epoc time), "External_Address", "Internal_Address"): {
        #         1st Packet registered time(unix epoc time): [direction, protocol, port, tcp_flag, length, label],
        #         2nd Packet registered time(unix epoc time): [direction, protocol, port, tcp_flag, length, label],
        #         3rd Packet registered time(unix epoc time): [direction, protocol, port, tcp_flag, length, label],
        #         >>>
        #         Nth Packet registered time(unix epoc time): [direction, protocol, port, tcp_flag, length, label]
        #     },
        #     (1659513007.736493, '192.168.10.5', '187.35.147.87'): {
        #         1659513007.736493: ['rcv', 'tcp', '22', 'PA', '136', '1'],
        #     }
        # }
        self.flow_manager = dict()
        self.featured_flow_matrix = [[
            "ex_address",
            "in_address",
            "timestamp",
            "rcv_packet_count",
            "snd_packet_count",
            "tcp_count",
            "udp_count",
            "most_port",
            "port_count",
            "rcv_max_interval",
            "rcv_min_interval",
            "rcv_max_length",
            "rcv_min_length",
            "snd_max_interval",
            "snd_min_interval",
            "snd_max_length",
            "snd_min_length",
            "label"
        ]]

    def delete_flow(self,key): # flow_boxからフローを削除
        flow = self.flow_manager.pop(key)
        featured_list = [key[2],key[1],(datetime.fromtimestamp(key[0], tz=timezone.utc).
                          astimezone(timezone(timedelta(hours=9)))
                          + timedelta(seconds=self.flow_timeout))
                         .strftime('%Y-%m-%d %H:%M:%S')]
        featured_list.extend(extract_features_from_flow(flow))
        self.featured_flow_matrix.append(featured_list)

    def is_flow_exist(self, captured_time, src, dst):
        if not self.flow_manager:
            return None
        else:
        # --- is all flow timeout ?
            keys = list(self.flow_manager.keys())  # 明示的にキーリストを作成
            for key in keys:
                if captured_time - key[0] > float(self.flow_timeout):
                    self.delete_flow(key)

            # --- is flow pkt specified existing ?
            keys = list(self.flow_manager.keys())
            for key in keys:
                if (key[1] == src and key[2] == dst) or (key[1] == dst and key[2] == src):
                    return key  # if exist return key
            return None

    def callback(self,pkt):
        captured_time = float(pkt.time)
        src = str(pkt[IP].src)
        dst = str(pkt[IP].dst)

        addr,field = extract_features_from_packet(pkt, self.malicious_address_set,
                                                  self.benign_address_set)
        key = self.is_flow_exist(captured_time, src, dst)


        if field is not None:
            if key is None:
                new_flow_key = (captured_time,addr[0],addr[1])
                self.flow_manager[new_flow_key] = {captured_time:field}
                # print("--- newflow")
                # print(new_flow_key)
            else:
                self.flow_manager[key][captured_time] = field
            #     print(f"--- exist: [[[{captured_time},{addr[0]},{addr[1]}]]]")
            #     print(key)
            # print(field)

def online(manager, filter_online):
    print("online mode")
    while 1:
        sniff(prn=manager.callback, timeout = capture_timeout, store = False, filter=filter_online)

def offline(file_path, manager, filter_offline):
    print("offline mode sniffing ...")
    if os.path.getsize(file_path) == 0:
        print(os.path.basename(file_path) + ":no data")
    else:
        sniff(offline=file_path, prn=manager.callback, store=False, filter=filter_offline)
        print(os.path.basename(file_path) + ":finish")


# 与えられたフォルダまたはファイルに含まれるトラヒックデータ（pcapファイル）から特徴量を抽出してcsvファイルに書き込む
if __name__ == "__main__":

    start_time = time.time()  # 処理開始時刻

    # --- Load settings
    settings = json.load(open('src/main/settings.json', 'r'))

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- Field

    online_mode = settings["FeatureExtract"]["ONLINE_MODE"]
    traffic_data_path = settings["FeatureExtract"]["TRAFFIC_DATA_PATH"]

    flow_timeout = settings["FeatureExtract"]["FLOW_TIMEOUT"]  # connection timeout [seconds]
    capture_timeout = settings["FeatureExtract"]["CAPTURE_TIMEOUT"]


    malicious_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["MALICIOUS"]
    malicious_address_set = set()
    for net in malicious_network_address_list:
        for malicious in ipaddress.ip_network(net):
            malicious_address_set.add(str(malicious))

    benign_network_address_list = settings["FeatureExtract"]["NetworkAddress"]["BENIGN"]
    benign_address_set = set()
    for net in benign_network_address_list:
        for benign in ipaddress.ip_network(net):
            benign_address_set.add(str(benign))

    ex_addr_list = settings["FeatureExtract"]["NetworkAddress"]["EXCEPTION"]
    ex_addr_filter = " and ".join([f"not (ip src {ip} or ip dst {ip})" for ip in ex_addr_list])
    filter_condition = f"ether proto 0x0800 and ({ex_addr_filter}) and (tcp or udp)"

    # --- Create output directory for csv
    outputs_dir_path: str = f"src/main/outputs/extracted/{init_time}"
    os.makedirs(outputs_dir_path)

    # --- build constructor
    flow_manager = FlowManager(ex_addr_list,malicious_address_set,benign_address_set,flow_timeout)

    # --- Online mode
    if online_mode:
        online(flow_manager,filter_condition)

    # --- Offline mode
    # --- データセット内のpcapファイルごとに特徴量抽出を行う
    else:
        if os.path.isdir(traffic_data_path):
            count = 0
            for pcap_file in os.listdir(traffic_data_path):
                print("- " + pcap_file + " found")
                pcap_file_path:str = os.path.join(traffic_data_path,pcap_file)
                offline(pcap_file_path,flow_manager,filter_condition)
                feature_matrix_column = flow_manager.featured_flow_matrix[0]
                feature_matrix_row = flow_manager.featured_flow_matrix[1:]
                feature_matrix = pd.DataFrame(feature_matrix_row,columns=feature_matrix_column)
                feature_matrix.to_csv(f"{outputs_dir_path}/{count:05d}_{os.path.splitext(os.path.basename(traffic_data_path))[0]}.csv",index=False)
                flow_manager.featured_flow_matrix = [feature_matrix_column] # featured_flow_matrixの初期化
                count += 1
            print("all pcap file sniffed")
        elif os.path.isfile(traffic_data_path):
            print("- " + os.path.basename(traffic_data_path) + " found")
            offline(traffic_data_path,flow_manager,filter_condition)
            feature_matrix_column = flow_manager.featured_flow_matrix[0]
            feature_matrix_row = flow_manager.featured_flow_matrix[1:]
            feature_matrix = pd.DataFrame(feature_matrix_row,columns=feature_matrix_column)
            feature_matrix.to_csv(f"{outputs_dir_path}/{os.path.splitext(os.path.basename(traffic_data_path))[0]}.csv",index=False)
            print("all pcap file sniffed")
        else:
            print(f"traffic data path : {traffic_data_path} not found")