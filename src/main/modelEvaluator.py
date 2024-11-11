from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main(model, evaluate_dataframe, scaler, scaled_flag, evaluate_daytime):

    if not len(evaluate_dataframe) == 0:
        features = evaluate_dataframe.iloc[:, 3:-1]
        targets = evaluate_dataframe.iloc[:, -1].astype(int)
        if not scaled_flag:
            print("feature matrix scaling first time")
            scaled_feature_matrix = scaler.fit_transform(features)
            scaled_flag = True
        else:
            scaled_feature_matrix = scaler.transform(features)

        # --- Prediction
        prediction_values = model.predict(scaled_feature_matrix)
        prediction_binary_values = (prediction_values >= 0.5).astype(int)

        # --- Evaluate
        accuracy = accuracy_score(targets, prediction_binary_values)
        precision = precision_score(targets, prediction_binary_values)
        recall = recall_score(targets, prediction_binary_values)
        f1 = f1_score(targets, prediction_binary_values)

        return [evaluate_daytime, accuracy, precision, recall, f1], scaled_flag