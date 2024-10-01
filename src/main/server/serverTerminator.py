import driftDetection
import featureExtraction
import intrutionDetection
import machineLearning
import tensorflow as tf

# ----- Def foundation model

def createModel():
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

def sendModel():

    return 0

# ----- Main

if __name__ == "__main__":

    # --- Create foundation model

    # --- Send foundation model to Gateway
    sendModel()

    # --- Start machine learning
    machineLearning.main()