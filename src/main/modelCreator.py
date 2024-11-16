import tensorflow as tf

print(tf.__version__)  # TensorFlowのバージョン確認
print(tf.keras.__file__)  # kerasモジュールのパス確認

# ----- Define foundation model
def main():
    # Sequentialモデルを定義
    model = tf.keras.models.Sequential([
        # 入力層＋1層目の全結合レイヤー（ReLU活性化関数）
        tf.keras.layers.Dense(100, activation='relu', input_dim=14),
        # 2層目の全結合レイヤー（ReLU活性化関数）
        tf.keras.layers.Dense(50, activation='relu'),
        # 3層目の全結合レイヤー（ReLU活性化関数）
        tf.keras.layers.Dense(25, activation='relu'),
        # 出力層（Sigmoid活性化関数）
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
