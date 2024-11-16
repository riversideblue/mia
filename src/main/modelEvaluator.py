import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

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

    # 各指標を計算
    accuracy = accuracy_score(targets, prediction_binary)
    precision = precision_score(targets, prediction_binary)
    recall = recall_score(targets, prediction_binary)
    f1 = f1_score(targets, prediction_binary)
    loss = log_loss(targets, predictions)  # 損失値の計算

    return accuracy, precision, recall, f1, loss

def main(model, df, scaler, scaled_flag, evaluate_daytime):

    if not df.empty:
        features = df.iloc[:, 3:-1]
        targets = df.iloc[:, -1].astype(int)

        scaled_features, scaled_flag = scale_features(features, scaler, scaled_flag)

        # --- Prediction
        prediction_values = model.predict(scaled_features)

        # --- Evaluate
        accuracy, precision, recall, f1, loss = evaluate_model(targets, prediction_values)


        flow_num = df.shape[0]
        benign_count = np.sum(targets == 0)
        malicious_count = np.sum(targets == 1)
        benign_rate = benign_count/flow_num

        return [
            evaluate_daytime,
            accuracy,
            precision,
            recall,
            f1,
            loss,
            benign_count,
            malicious_count,
            benign_rate,
            flow_num
        ], scaled_flag