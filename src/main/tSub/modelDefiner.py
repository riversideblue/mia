import pandas as pd


def dnn(tf):
    # Deep Neural Network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,), dtype=tf.float32),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def rnn(tf):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14, 1)),  # 時系列データの入力
        tf.keras.layers.SimpleRNN(50, activation='relu', return_sequences=True),
        tf.keras.layers.SimpleRNN(25, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 出力層
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def autoencoder(tf):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),  # 入力層（14次元）
        tf.keras.layers.Dense(64, activation='relu'),  # エンコーダ層1
        tf.keras.layers.Dense(32, activation='relu'),  # エンコーダ層2 (ボトルネック)
        tf.keras.layers.Dense(64, activation='relu'),  # デコーダ層1
        tf.keras.layers.Dense(14, activation='sigmoid')  # デコーダ層2 (再構築層)
    ])
    model.compile(
        optimizer='adam',
        loss='mse',  # 再構築誤差を損失関数に
        metrics=['mae']  # 平均絶対誤差を評価指標に
    )
    return model

def svm(tf):
    def hinge_loss(y_true, y_pred):
        return tf.reduce_mean(tf.maximum(0., 1. - y_true * y_pred))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.Dense(1, activation='linear')  # 線形活性化
    ])
    model.compile(
        optimizer='adam',
        loss=hinge_loss,
        metrics=['accuracy']
    )
    return model

def logistic_regression(tf):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 出力層はSigmoid活性化
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def lstm(tf):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def random_forest(tfdf):
    class CustomRandomForestModel(tfdf.keras.RandomForestModel):
        def fit(self, x=None, y=None, **kwargs):
            # x が Pandas DataFrame の場合、自動で変換
            if isinstance(x, pd.DataFrame):
                if 'label' not in kwargs:
                    raise ValueError("Please provide the label column name as `label='your_label_column'`.")
                x = tfdf.keras.pd_dataframe_to_tf_dataset(x, label=kwargs.pop('label'))
            super().fit(x, y, **kwargs)

    # モデルの構築
    model = CustomRandomForestModel(
        task=tfdf.keras.Task.CLASSIFICATION,  # 分類タスク
        num_trees=100,  # 決定木の数
    )
    return model


def gradient_boosting(tfdf):
    class CustomGradientBoostingModel(tfdf.keras.GradientBoostedTreesModel):
        def fit(self, x=None, y=None, **kwargs):
            # x が Pandas DataFrame の場合、自動で変換
            if isinstance(x, pd.DataFrame):
                if 'label' not in kwargs:
                    raise ValueError("Please provide the label column name as `label='your_label_column'`.")
                x = tfdf.keras.pd_dataframe_to_tf_dataset(x, label=kwargs.pop('label'))
            super().fit(x, y, **kwargs)

    # モデルの構築
    model = CustomGradientBoostingModel(
        task=tfdf.keras.Task.CLASSIFICATION,  # 分類タスク
        num_trees=100,  # 決定木の数
        max_depth=6,    # 決定木の深さ
    )
    return model
