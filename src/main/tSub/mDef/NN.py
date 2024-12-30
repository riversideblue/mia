# ----- Define foundation model
def main(tf):
    # Sequentialモデルを定義
    model = tf.keras.Sequential([
        # 入力層 (14次元の特徴量)
        tf.keras.layers.Input(shape=(14,), dtype=tf.float32),
        # 隠れ層1 (100ユニット, ReLU活性化)
        tf.keras.layers.Dense(100, activation='relu'),
        # 隠れ層2 (50ユニット, ReLU活性化)
        tf.keras.layers.Dense(50, activation='relu'),
        # 隠れ層3 (25ユニット, ReLU活性化)
        tf.keras.layers.Dense(25, activation='relu'),
        # 出力層 (1ユニット, Sigmoid活性化)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
