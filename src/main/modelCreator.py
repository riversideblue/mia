import keras

# ----- Define foundation model
def main():
    # Sequentialモデルを定義
    model = keras.models.Sequential([
        # 入力層＋1層目の全結合レイヤー（ReLU活性化関数）
        keras.layers.Input(shape=(14,)),  # 入力形状を明示的に指定
        keras.layers.Dense(100, activation='relu'),
        # 2層目の全結合レイヤー（ReLU活性化関数）
        keras.layers.Dense(50, activation='relu'),
        # 3層目の全結合レイヤー（ReLU活性化関数）
        keras.layers.Dense(25, activation='relu'),
        # 出力層（Sigmoid活性化関数）
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
