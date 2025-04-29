import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings(action="ignore")


def build_alexnet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96,(3,3),strides=(1,1),activation="relu",input_shape=(32,32,3), padding='same'),
        tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(256,(5,5),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(384,(3,3),activation="relu",padding="same"),
        tf.keras.layers.Conv2D(384,(3,3),activation="relu",padding="same"),
        tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation="softmax")
    ])
    return model 


# def build_alexnet():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(96,(3,3),strides=(1,1),activation="relu",input_shape=(32,32,3), padding='same'),
#         tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)),
#         tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"),
#         tf.keras.layers.MaxPooling2D((2,2),strides=(2,2)),
#         tf.keras.layers.Conv2D(384,(3,3),activation="relu",padding="same"),
#         tf.keras.layers.Conv2D(384,(3,3),activation="relu",padding="same"),
#         tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"),
#         # Remove final pooling or adjust
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(4096,activation="relu"),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(4096,activation="relu"),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(10,activation="softmax")
#     ])
#     return model
