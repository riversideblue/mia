import numpy as np

test2 = [[[2,3,4,5,6],[2,3,5,6,7,8]],[[2,3,4,5,6],[2,3,5,6,7,8]]]
test3 = [
    [["dst","snd","label"],[198.23,198.45,"benign"],[172.21,172.99,"malicious"]], # flow 1
    [["dst","snd","label"],[138.13,192.43,"malicious"],[172.21,172.99,"malicious"]],   # flow 2
    [["dst","snd","label"],[121.23,121.67,"benign"],[172.21,172.99,"malicious"]] # flow 3
    ]

test = np.array([[2,3,4,5,6],[2,3,5,6,7,8]],[[2,3,4,5,6],[2,3,5,6,7,8]])

print(test)
print(test3)
print(test2)