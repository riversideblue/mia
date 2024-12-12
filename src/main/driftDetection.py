import numpy as np
from collections import deque

class Window:
    def __init__(self,present_window_size,past_window_size,threshold,row_len):
        self.present_window = deque(
            np.full((present_window_size, row_len), -threshold, dtype=object),
            maxlen=present_window_size
        )
        self.past_window = deque(
            np.full((past_window_size, row_len), -threshold, dtype=object),
            maxlen=past_window_size
        )
        self.threshold = threshold

    def update(self,row):
        self.past_window.append(self.present_window.popleft())
        self.present_window.append(np.array(row, dtype=object))

    def fnum_present(self):
        rcv_present = np.array(self.present_window)[:, 3].astype(int)
        snd_present = np.array(self.present_window)[:, 4].astype(int)
        extracted_present = rcv_present + snd_present
        return extracted_present

    def fnum_past(self):
        rcv_past = np.array(self.past_window)[:, 3].astype(int)
        snd_past = np.array(self.past_window)[:, 4].astype(int)
        extracted_past = rcv_past + snd_past
        return extracted_past

def TTest(present_window,past_window,threshold):

    # debug
    print(f"Present: {present_window}")
    print(f"Past: {past_window}")
    ave_fn_present = np.mean(present_window)
    ave_fn_past = np.mean(past_window)
    if abs(ave_fn_present - ave_fn_past) > threshold:
        return True
    return False

class KLDivergence:
    def __init__(self):
        print("KL Divergence Based Drift Detection")


class ADWIN:
    def __init__(self):
        print("AD Window Based Drift Detection")

class DDM:
    def __init__(self):
        print("DDM Based Drift Detection")

class Hoeffding:
    def __init__(self):
        print("Hoeffding Based Drift Detection")

class KSTest:
    def __init__(self):
        print("KSTest Based Drift Detection")

class LeveneTest:
    def __init__(self):
        print("Levene Test Based Drift Detection")

class Wasserstein:
    def __init__(self):
        print("Wasserstein Based Drift Detection")

class LikelihoodRatioTest:
    def __init__(self):
        print("Likelihood Ratio Test Based Drift Detection")

class Bhattacharyya:
    def __init__(self):
        print("Bhattacharyya Based Drift Detection")