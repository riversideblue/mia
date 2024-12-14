import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
import tensorflow as tf

def main (y_true,y_pred):

    y_pred_bin =  [1 if x >= 0.5 else 0 for x in y_pred]
    conf_matrix = confusion_matrix(y_true, y_pred_bin,labels=[0,1])
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真陽性率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 偽陽性率
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 偽陰性率
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真陰性率

    accuracy = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    eval_loss = bce(np.array(y_true), np.array(y_pred)).numpy()

    flow_num = tn + fp + fn + tp
    benign_count = tn + fp
    benign_rate = benign_count / flow_num if flow_num > 0 else 0

    return tp,fn,fp,tn,flow_num,tpr,fnr,fpr,tnr,accuracy,precision,f1,eval_loss,benign_rate