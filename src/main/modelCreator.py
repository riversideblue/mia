import tensorflow as tf

# ----- Def foundation model
def main():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100, input_dim=14))
    model.add(tf.keras.layers.Activation(tf.nn.relu))

    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Activation(tf.nn.relu))

    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Activation(tf.nn.relu))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation(tf.nn.sigmoid))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
