import time
import pickle
import numpy as np

def main(model, features, targets, output_dir_path, epochs, batch_size, rtr_date):
    print("execute model_training ...")

    # モデルのトレーニング
    start_time = time.time()
    history = model.fit(features, targets, epochs=epochs, batch_size=batch_size)
    training_time = time.time() - start_time

    # 結果の記録
    accuracy = history.history["accuracy"][-1]
    loss = history.history["loss"][-1]
    str_rtr_date = rtr_date.strftime("%Y-%m-%dT%H:%M:%S")

    # モデルの重みを保存
    with open(f"{output_dir_path}/model_weights/{str_rtr_date}-weights.pickle", 'wb') as f:
        pickle.dump(model.get_weights(), f)

    # データの統計情報
    benign_count = np.sum(targets == 0)
    malicious_count = np.sum(targets == 1)
    flow_num = len(targets)

    return model, [rtr_date, accuracy, loss, training_time, benign_count, malicious_count, flow_num]

