import collections



alt_val = -10000    # alternative value for missing value

class FlowFeaturesClass:
    
    date = ""   # flow started datetime    
    addr = ""   # external host
    partner = "" # internal host
    label = ""  # "0" : "1" <=> benign : malicious    
    
    rcv_packet_count = 0
    rcv_tmp_time = 0
    
    snd_packet_count = 0
    snd_tmp_time = 0

    last_seconds = 0
    #--------------------------------------------------------------

    def __init__(self, date, addr, partner, label):   #constructor
        self.date = date
        self.addr = addr
        self.partner = partner
        self.label = label
                
        self.rcv_interval_time_list = []
        self.rcv_length_list = []
        self.snd_interval_time_list = []
        self.snd_length_list = []
        
        self.protocol_list = []
        self.port_list = [] # list of destination ports for incoming packets
                
    def __del__(self):    #destructor
        pass
        


    def calculateTimeInterval(self, next_time, direction):
        if direction == "rcv":
            if self.rcv_tmp_time == 0:
                self.rcv_tmp_time = next_time
            else:
                interval = round(next_time - self.rcv_tmp_time, 6)
                self.rcv_interval_time_list.append(interval)
                self.rcv_tmp_time = next_time
        
        elif direction == "snd":
            if self.snd_tmp_time == 0:
                self.snd_tmp_time = next_time
            else:
                interval = round(next_time - self.snd_tmp_time, 6)
                self.snd_interval_time_list.append(interval)
                self.snd_tmp_time = next_time

    
    def getMaxMinTimeInterval(self, direction):
        if direction == "rcv":
            if len(self.rcv_interval_time_list) == 0:
                return alt_val, alt_val
            else:
                max_int = max(self.rcv_interval_time_list)
                min_int = min(self.rcv_interval_time_list)
                return max_int, min_int
                
        elif direction == "snd":
            if len(self.snd_interval_time_list) == 0:
                return alt_val, alt_val
            else:
                max_int = max(self.snd_interval_time_list)
                min_int = min(self.snd_interval_time_list)
                return max_int, min_int
                
    
    def getMaxMinLength(self, direction):
        if direction == "rcv":
            if len(self.rcv_length_list) == 0:
                return alt_val, alt_val
            else:
                max_len = max(self.rcv_length_list)
                min_len = min(self.rcv_length_list)
                return max_len, min_len
        
        elif direction == "snd":
            if len(self.snd_length_list) == 0:
                return alt_val, alt_val
            else:
                max_len = max(self.snd_length_list)
                min_len = min(self.snd_length_list)
                return max_len, min_len

    
    def getProtocolCount(self):
        tcp_count = self.protocol_list.count("tcp")
        udp_count = self.protocol_list.count("udp")
        return tcp_count, udp_count
    
    
    def getMostPortAndCount(self):
        Counter_port_list = collections.Counter(self.port_list)
        return Counter_port_list.most_common()[0][0], len(Counter_port_list.most_common())