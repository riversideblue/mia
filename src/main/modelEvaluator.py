def main (confusion_matrix):

    tp = sum(confusion_matrix[:,0])
    fp = sum(confusion_matrix[:, 1])
    fn = sum(confusion_matrix[:, 2])
    tn = sum(confusion_matrix[:, 3])

    # 各指標を計算
    tpr = tp / (tp + fn) \
        if (tp + fn) != 0 else 0 # 真陽性率 (Recall)
    fpr = fp / (fp + tn) \
        if (fp + tn) != 0 else 0 # 偽陽性率
    fnr = fn / (fn + tp) \
        if (fn + tp) != 0 else 0 # 偽陰性率
    tnr = tn / (tn + fp) \
        if (tn + fp) != 0 else 0 # 真陰性率 (Specificity)
    accuracy = (tp + tn) / (tp + tn + fp + fn) \
        if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) \
        if (tp + fp) != 0 else 0
    f1 = (2 * precision * tpr) / (precision + tpr) \
        if (precision + tpr) != 0 else 0
    loss = 1

    benign_count = tn + fp
    malicious_count = tp + fn
    flow_count = benign_count + malicious_count
    benign_rate = benign_count / flow_count

    return tp,fp,fn,tn,tpr,fpr,fnr,tnr,accuracy,precision,f1,loss,flow_count,benign_rate