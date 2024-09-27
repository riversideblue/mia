import warnings

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.sendrecv import sniff
from scapy.utils import wrpcap

warnings.simplefilter('ignore')

import numpy as np

from datetime import datetime
import glob
import os
import csv
import ipaddress
import sys
import json

from FlowFeaturesClass import *



pcap_dir_name = sys.argv[1].replace("/","") # pcap files directory name

online_flag = False
if pcap_dir_name == "online":
    online_flag = True
#---------------------------------------------------------------------------------
settings = json.load(open("settings.json", "r"))

pcap_fl_id = settings["pcap_file_identifier"] # pcap file identifier(ex: ".pcap")
subdir_year_month = settings["subdirectory_year_month"]
subdir_start_d = settings["subdirectory_elapsed_day"]  # subdir start day

timeout = settings["flow_timeout"]   # connection timeout [seconds]
timeout_check_interval = settings["packet_interval_check_timeout"]    # check timeout at N packets intervals

malicious_network_address_list = settings["malicious_network_address"]
benign_network_address_list = settings["benign_network_address"]
ex_addr_list = settings["exception_address"]

captime = settings["capture_time"]  # capture time for online mode [seconds]
pcap_saved_dir_name = settings["pcap_files_saved_dir_name"]
#--------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------


def callback(pkt):
    global now_csv_file_name
    global first_packet_flag
    global current_time
    global packet_list
    
    try:
        #ipv4
        if pkt[Ether].type == 0x0800 and pkt[IP].src not in ex_addr_list and pkt[IP].dst not in ex_addr_list and (pkt[IP].proto == 6 or pkt[IP].proto == 17):

            if online_flag == True:
                packet_list.append(pkt)
            
            time_list = str(datetime.fromtimestamp(float(pkt.time))).split(" ")
            day = time_list[0].split("-")[0] + time_list[0].split("-")[1] + time_list[0].split("-")[2]  # feat: day(str) : "20220101" 
            hour = time_list[1].split(":")[0]   # feat: hour(str) : "01"
            date = day + hour        

            csv_file_name = date + ".csv"
            if csv_file_name != now_csv_file_name and first_packet_flag == False:   
                now_csv_file_name = csv_file_name
                first_packet_flag = True
            elif csv_file_name != now_csv_file_name and first_packet_flag == True:
                csv_fn = now_csv_file_name
                writeCsv(csv_fn)
                now_csv_file_name = csv_file_name
            
            minute = time_list[1].split(":")[1]            
            seconds = float(pkt.time)   # feat: seconds(float) : 1634701.123456
            
            current_time = seconds
            
            if pkt[IP].proto == 6:
                protocol = "tcp"   # feat: protocol(str) : "tcp"
            elif pkt[IP].proto == 17:
                protocol = "udp"   # feat: protocol(str) : "udp"
            
            length = int(pkt[IP].len)   # feat: length(int) : 64
            
            #----------------------------------------------------------------------------------------
            
            # src : malicious address
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
                    
                summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag)
            
            # dst : malicious address
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
            
            else:   #unrelated packet
                pass
        
        else:   #not ipv4 packet
            pass

    except IndexError:  # not found [Ether]layer
        pass
    

def summarize(day, hour, minute, seconds, protocol, length, addr, partner, label, direction, port, tcpflag):
    global connecting_address_list
    global connecting_object_list    
    global output_list
    global timeout_count
    
    timeout_count += 1     
    #----------------------- first packet(no connection) ---------------------------------
    if addr not in connecting_address_list: 
        connecting_address_list.append(addr)
        
        date = day + hour + minute
        obj = FlowFeaturesClass(date, addr, partner, label) # create instance
        
        if direction == "rcv":
            obj.rcv_packet_count += 1
            obj.rcv_tmp_time = seconds
            obj.rcv_length_list.append(length)
        
        elif direction == "snd":
            obj.snd_packet_count += 1
            obj.snd_tmp_time = seconds
            obj.snd_length_list.append(length)
        
        obj.protocol_list.append(protocol)
        obj.port_list.append(port)
        
        obj.last_seconds = seconds
        
        connecting_object_list.append(obj)
               
    #----------------------- already connecting ------------------------------------------
    else:   
        addr_index = connecting_address_list.index(addr)
        
        if direction == "rcv":
            connecting_object_list[addr_index].rcv_packet_count += 1
            connecting_object_list[addr_index].calculateTimeInterval(seconds, direction)
            connecting_object_list[addr_index].rcv_length_list.append(length)
            
        elif direction == "snd":
            connecting_object_list[addr_index].snd_packet_count += 1
            connecting_object_list[addr_index].calculateTimeInterval(seconds, direction)
            connecting_object_list[addr_index].snd_length_list.append(length)

        connecting_object_list[addr_index].protocol_list.append(protocol)
        connecting_object_list[addr_index].port_list.append(port)
        
        connecting_object_list[addr_index].last_seconds = seconds

    #------------------------ timeout check --------------------------------------
    if timeout_count == timeout_check_interval:
        
        output_addr_list = []
        for l in range(len(connecting_address_list)):
            if connecting_object_list[l].last_seconds + timeout < current_time:
                output_list.append(connecting_object_list[l])
                output_addr_list.append(connecting_object_list[l].addr)

        for out_addr in output_addr_list:
            connecting_address_list.remove(out_addr)            
        
        for ob in connecting_object_list[:]:
            for out_addr in output_addr_list:                
                if ob.addr == out_addr:
                    connecting_object_list.remove(ob)

        timeout_count = 0
    #-------------------------------------------------------------------------------------


def writeCsv(csv_file_name):
    global output_list 
    
    existing_flag = False   # for check same name csv file
    try:
        if os.path.isfile(pcap_dir_name + "_csv" + "/" + csv_file_name):
            print("same name csv file exist")
            existing_flag = True    # same csv file exist
        f = open(pcap_dir_name + "_csv" + "/" + csv_file_name, "a", newline = "")

    except Exception as e:
        print("not found save directory")
        os.makedirs(pcap_dir_name + "_csv", exist_ok = False)
        print(pcap_dir_name + "_csv:made")
        f = open(pcap_dir_name + "_csv" + "/" + csv_file_name, "a", newline = "")
        
    writer = csv.writer(f, lineterminator = "\n")

    if existing_flag == False:  # same csv file not exist
        writer.writerow(["address", "partner", "date",  
                         "rcv_packet_count", "snd_packet_count",
                         "tcp_count", "udp_count", "most_port", "port_count",
                         "rcv_max_int", "rcv_min_int", "rcv_max_len", "rcv_min_len",
                         "snd_max_int", "snd_min_int", "snd_max_len", "snd_min_len", 
                         "label"])
    else:
        pass

    if len(output_list) == 0:
        print(pcap_dir_name + "_csv" + "/" + csv_file_name + ":no data")
    
    else:
        for ob in output_list:
            tcp_count, udp_count = ob.getProtocolCount()
            most_port, port_count = ob.getMostPortAndCount()
            rcv_max_int, rcv_min_int = ob.getMaxMinTimeInterval("rcv")
            rcv_max_len, rcv_min_len = ob.getMaxMinLength("rcv")
            snd_max_int, snd_min_int = ob.getMaxMinTimeInterval("snd")
            snd_max_len, snd_min_len = ob.getMaxMinLength("snd")
            
            writer.writerow([str(ob.addr), str(ob.partner), str(ob.date), 
                            str(ob.rcv_packet_count), str(ob.snd_packet_count),
                            str(tcp_count), str(udp_count), str(most_port), str(port_count),
                            str(rcv_max_int), str(rcv_min_int), str(rcv_max_len), str(rcv_min_len),
                            str(snd_max_int), str(snd_min_int), str(snd_max_len), str(snd_min_len),
                            str(ob.label)])

    f.close()
    print(pcap_dir_name + "_csv" + "/" + csv_file_name + ":saved")
    output_list.clear()
    

def stopFilter(p):
    global termination_flag

    try:
        if p[Ether].type == 0x0800:   #IPv4
            if p[IP].proto == 1 and (p[IP].src == "192.0.2.1" or p[IP].dst == "192.0.2.1"):
                termination_flag = True
                return True
            else:    
                return False
                
    except IndexError:
        pass


def setCaptureStartDatetime():
    global start_datetime
    start_datetime = str(datetime.now()).split(" ")[0].replace("-","") + str(datetime.now()).split(" ")[1].split(".")[0].replace(":","")


def writePcap():
    global packet_list

    pcap_file_name = start_datetime + ".pcap"
    if not os.path.isdir(pcap_saved_dir_name):
        os.makedirs(pcap_saved_dir_name)
    wrpcap(pcap_saved_dir_name + "/" + pcap_file_name, packet_list)
    print(pcap_file_name + ":saved")
    packet_list = []


if __name__ == "__main__":

    #----------------------------------------- offline -----------------------------------------------------
    if not online_flag:
        file_path_list = glob.glob(pcap_dir_name + "/" + "*")

        file_list = []
        for path in file_path_list:
            file_list.append(path.split("/")[1])

        #------- pcap files in dir -------
        pcap_exist_flag = False
        for fn in file_list:
            if "pcap" in fn:
                pcap_exist_flag = True
                print("pcap_exist_flag => True")
                break
        #----------------------------------

        #-------- pcap files in subdir -------------
        if not pcap_exist_flag:
            for i in range(len(glob.glob(pcap_dir_name + "/*/"))):
                if i+1+subdir_start_d < 10:
                    subdir_name = subdir_year_month + "0" + str((i + 1) + subdir_start_d)
                else:
                    subdir_name = subdir_year_month + str((i + 1) + subdir_start_d)
                
                for l in range(len(glob.glob(pcap_dir_name + "/*" + subdir_name + "*/" + "*" + pcap_fl_id + "*"))):
                    pcap_fl_name = glob.glob(pcap_dir_name + "/*" + subdir_name + "*/" + "*" + pcap_fl_id + "*")[l]

                    if os.path.getsize(pcap_fl_name) == 0:
                        print(pcap_fl_name + ":no data")
                    else:
                        print(pcap_fl_name + ":sniffing...")
                        sniff(offline = pcap_fl_name, prn = callback, store = False)
                        print(pcap_fl_name + ":finish")
                    
                print("all pcap file sniffed")
                writeCsv(now_csv_file_name)
        #------------------------------------------

        #----------- pcap files in dir ------------
        elif pcap_exist_flag:
            for i in range(len(glob.glob(pcap_dir_name + "/" + "*" + pcap_fl_id + "*"))):
                pcap_fl_name = glob.glob(pcap_dir_name + "/" + "*" + pcap_fl_id + "*")[i]
                
                if os.path.getsize(pcap_fl_name) == 0:
                    print(pcap_fl_name + ":no data")
                else:
                    print(pcap_fl_name + ":sniffing...")
                    sniff(offline = pcap_fl_name, prn = callback, store = False)
                    print(pcap_fl_name + ":finish")
                    
            print("all pcap file sniffed")
            writeCsv(now_csv_file_name)
        #-------------------------------------------

    #-------------------------------------------- online -----------------------------------------------

    else:
        print("online mode")
        while 1:
            setCaptureStartDatetime()
            sniff(prn = callback, timeout = captime, stop_filter = stopFilter, store = False)
            if termination_flag:
                writePcap()
                break