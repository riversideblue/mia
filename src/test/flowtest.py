import multiprocessing

flow = {
    # flow info
    ("FLOW_START_TIME","EX_ADDR","IN_ADDR"):{
        # packet info
        "CAPTURED_TIME":["direction","protocol","port","tcp_flag","label"]
    }, # ex under
    (12889393332.1288,"192.168.10.5","169.197.141.136"):{
        "20220215165007276936":["rcv","tcp","80","0","1"],
        "20220215165007277812":["snd","udp","30","0","0"],
        "20220215165007279918":["rcv","tcp","80","0","1"],
    },
    (1992883774.1222,"192.168.10.5","142.251.42.136"):{
        "20220215165007276936":["rcv","tcp","80","0","1"],
        "20220215165007277812":["snd","udp","30","0","0"],
        "20220215165007279918":["rcv","tcp","80","0","1"],
    },(1992883224.1222,"192.168.10.5","142.251.42.136"):{

    },

}

def flow_gen():

    manager = multiprocessing.Manager()
    x = manager.dict()
    x[("coffee","chocolate")] = {
        "0919" : [1,2,3,4,5]
    }
    x[("whiskey","cigarette")] = {
        "0920" : [1,3,3,4,5]
    }
    x[("beer","sausage")] = {
        "0920" : [1,3,3,4,5]
    }
    return x

# print(flow)
# print(flow_gen())
# print(len(flow_gen()))

key = (1992883774.1222,"192.168.10.5","169.197.141.136")

print(flow[key])
delete_key_list = [
    key for key in flow
    if key[0] == 1992883774.1222
]

for key in delete_key_list:
    del flow[key]
#
# print(flow)

if not flow_gen():
    print("Null")
else:
    print("Yes")