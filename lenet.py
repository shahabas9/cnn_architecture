import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings(action="ignore")


def build_lenet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6,(5,5),activation="tanh",input_shape=(32,32,3),padding="same"),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(16,(5,5),activation="tanh"),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation="tanh"),
        tf.keras.layers.Dense(84,activation="tanh"),
        tf.keras.layers.Dense(10,activation="softmax")
    ])
    return model 