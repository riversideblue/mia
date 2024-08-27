from scapy.all import *



addr = "192.0.2.1"

if __name__ == "__main__":  
    
    #addr = input("destination address => ")
    
    try:            
        send(IP(src = addr, dst = addr)/ICMP(), verbose=0)
        send(IP(src = addr, dst = addr)/ICMP(), verbose=0)
        print("sent termination signal!")

    except OSError:
        pass