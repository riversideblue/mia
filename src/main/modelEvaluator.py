import pandas as pd


def main (confusion_matrix):

    print(pd.DataFrame(confusion_matrix))
    tp = sum(confusion_matrix[:,0])
    fp = sum(confusion_matrix[:, 1])
    fn = sum(confusion_matrix[:, 2])
    tn = sum(confusion_matrix[:, 3])

    # 各指標を計算

    if tp == 0:
        tpr = 0
    elif fn == 0:
        tpr = 1
    else:
        tpr = tp / (tp + fn) # 真陽性率

    if fp == 0:
        fpr = 0
    elif tn == 0:
        fpr = 1
    else:
        fpr = fp / (fp + tn) # 偽陽性率

    if fn == 0:
        fnr = 0
    elif tp == 0:
        fnr = 1
    else:
        fnr = fn / (fn + tp) # 偽陰性率

    if tn == 0:
        tnr = 0
    elif fp == 0:
        tnr = 1
    else:
        tnr = tn / (tn + fp) # 真陰性率

    if tp+tn == 0:
        accuracy = 0
    elif fp+fn == 0:
        accuracy = 1
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn) # Accuracy

    if tp == 0:
        precision = 0
    elif fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp) # precision (陽性的中率)

    f1 = (2 * precision * tpr) / (precision + tpr) \
        if (precision + tpr) != 0 else 0
    loss = 1

    benign_count = tn + fp
    malicious_count = tp + fn
    flow_count = benign_count + malicious_count
    benign_rate = benign_count / flow_count

    return tp,fp,fn,tn,tpr,fpr,fnr,tnr,accuracy,precision,f1,loss,flow_count,benign_rate