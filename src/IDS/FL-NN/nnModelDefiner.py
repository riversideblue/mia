import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    #log amount
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    #gpu mem limit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #cpu : -1
import tensorflow as tf



# in => 100 => 50 => 25 => out
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