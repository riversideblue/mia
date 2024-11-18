import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix


def scale_features(features, scaler, scaled_flag):

    if not scaled_flag:
        print("Feature matrix scaling for the first time")
        scaled_features = scaler.fit_transform(features)
        scaled_flag = True
    else:
        scaled_features = scaler.transform(features)
    return scaled_features, scaled_flag

def evaluate_model(targets, predictions):
    # 予測結果を2値化
    prediction_binary = (predictions >= 0.5).astype(int)

    cm = confusion_matrix(targets, prediction_binary,labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)

    # 各指標を計算
    tpr = tp / (tp + fn)  # 真陽性率 (Recall)
    fpr = fp / (fp + tn)  # 偽陽性率
    tnr = tn / (tn + fp)  # 真陰性率 (Specificity)
    fnr = fn / (fn + tp)  # 偽陰性率

    # 各指標を計算
    accuracy = accuracy_score(targets, prediction_binary)
    loss = log_loss(targets, predictions, labels=[0, 1])
    f1 = f1_score(targets, prediction_binary)

    return tpr, fpr, tnr, fnr, accuracy, loss, f1

def main(model, df, scaler, scaled_flag, evaluate_daytime):

    if not df.empty:
        features = df.iloc[:, 3:-1]
        targets = df.iloc[:, -1].astype(int)

        scaled_features, scaled_flag = scale_features(features, scaler, scaled_flag)

        # --- Prediction
        prediction_values = model.predict(scaled_features)

        # --- Evaluate
        tpr, fpr, tnr, fnr, accuracy, loss, f1 = evaluate_model(targets, prediction_values)


        flow_num = df.shape[0]
        benign_count = np.sum(targets == 0)
        malicious_count = np.sum(targets == 1)
        benign_rate = benign_count/flow_num

        return [
            evaluate_daytime,
            tpr,
            fpr,
            tnr,
            fnr,
            accuracy,
            loss,
            f1,
            benign_count,
            malicious_count,
            flow_num,
            benign_rate
        ], scaled_flag