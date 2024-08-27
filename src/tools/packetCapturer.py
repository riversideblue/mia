from scapy.all import *

import datetime
import time
import threading
import os



captime = 3600

save_dir = "pcap"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#------------------------------------------------------------
filename = ""
sTime = time.time()
p_count = 0
packets = []
termination_flag = False
#------------------------------------------------------------


def createFileName(ex):
    dt = datetime.datetime.now()
    
    year = str(dt.year)
    if dt.month < 10:
        month = "0" + str(dt.month)
    else:
        month = str(dt.month)
    if dt.day < 10:
        day = "0" + str(dt.day)
    else:
        day = str(dt.day)
    if dt.hour < 10:
        hour = "0" + str(dt.hour)
    else:
        hour = str(dt.hour)
    if dt.minute < 10:
        minute = "0" + str(dt.minute)
    else:
        minute = str(dt.minute)
    if dt.second < 10:
        second = "0" + str(dt.second)
    else:
        second = str(dt.second)
    
    return year + month + day + hour + minute + second + ex


def setFileName(fn):
    global filename
    filename = fn
    

def initialize():
    global packets    
    packets = []
 

def getTime():
    eTime = time.time() - sTime
    return eTime


def callback(pkt):
    global packets
    
    packets.append(pkt)
    printPacket(pkt)


def printPacket(pkt):
    global p_count
    
    try:
        if pkt[Ether].type == 0x0800:   # IPv4          
            length = pkt[IP].len + 14    
            eTime = getTime()

            if pkt[IP].proto == 6:    # TCP
                p_count += 1
                print('{0:7} | {1:12.5f} |TCP| {2:15} => {3:15} | {4:6} => {5:6} | {6:5}'.format(p_count, eTime, pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, length))
            elif pkt[IP].proto == 17:   # UDP
                p_count += 1
                print('{0:7} | {1:12.5f} |UDP| {2:15} => {3:15} | {4:6} => {5:6} | {6:5}'.format(p_count, eTime, pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, length))
            elif pkt[IP].proto == 1:    # ICMP
                p_count += 1
                print('{0:7} | {1:12.5f} |ICMP|{2:15} => {3:15} |                  | {4:5}'.format(p_count, eTime, pkt[IP].src, pkt[IP].dst, length))
    
    except IndexError:
        pass


def writePcap():
    wrpcap(save_dir + "/" + filename, packets)
    initialize()


def stopfilter(p):
    global termination_flag
    
    try:
        if p[Ether].type == 0x0800:
            if (p[IP].src == "192.0.2.1" or p[IP].dst == "192.0.2.1"):
                termination_flag = True
                return True
            else:    
                return False
                
    except IndexError:
        pass



if __name__ == "__main__":  #main

    interface = input("network interface =>")

    while(1):
        fn = createFileName(".pcap")
        setFileName(fn)
        
        print("-------------------------------------------------------------------------------------------")
        print(" No     | Time         |Pro| Source          => Destination     | SrcPort=> DstPort| len   ")
        print("-------------------------------------------------------------------------------------------")
        
        sniff(iface = interface, prn = callback, timeout = captime, stop_filter = stopfilter)

        print("--------------------------------------Pcap Saved-------------------------------------------")
        print("")
        
        thread = threading.Thread(target = writePcap)
        thread.start()

        if(termination_flag == True):
            initialize()
            break