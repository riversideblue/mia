from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from tensorflow.keras.losses import BinaryCrossentropy

def main (y_true,y_pred):

    y_pred_bin = (y_pred >= 0.5).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = conf_matrix.ravel()

    tpr = recall_score(y_true, y_pred_bin)  # TPR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 偽陰性率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 偽陽性率
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真陰性率

    accuracy = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)

    bce = BinaryCrossentropy()
    loss = bce(y_true, y_pred).numpy()

    flow_num = tn + fp + fn + tp
    benign_count = tn + fp
    benign_rate = benign_count / flow_num if flow_num > 0 else 0

    return tp,fn,fp,tn,flow_num,tpr,fpr,fnr,tnr,accuracy,precision,f1,loss,benign_rate