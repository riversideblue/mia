from tensorflow.python.keras import models,layers

# ----- Define foundation model
def main():
    # Sequentialモデルを定義
    model = models.Sequential([
        # 入力層＋1層目の全結合レイヤー（ReLU活性化関数）
        layers.Input(shape=(14,)),
        layers.Dense(100, activation='relu'),
        # 2層目の全結合レイヤー（ReLU活性化関数）
        layers.Dense(50, activation='relu'),
        # 3層目の全結合レイヤー（ReLU活性化関数）
        layers.Dense(25, activation='relu'),
        # 出力層（Sigmoid活性化関数）
        layers.Dense(1, activation='sigmoid')
    ])
    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
