import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings(action="ignore")

def residual_block(x,filter):
    shortcut = x
    x = layers.Conv2D(filter,(3,3),activation="relu",padding="same")(x)
    x=layers.Conv2D(filter,(3,3),padding="same")(x)
    # Projection shortcut if filter size mismatches
    if shortcut.shape[-1] != filter:
        shortcut = layers.Conv2D(filter, (1, 1), padding="same")(shortcut)
    x=layers.add([x,shortcut])
    x=layers.Activation("relu")(x)
    return x


def build_resnet():
    input =tf.keras.Input((32,32,3))
    x = layers.Conv2D(64,(3,3),padding="same",activation="relu")(input)
    x=residual_block(x,64)
    x=layers.MaxPooling2D((2,2))(x)
    x=residual_block(x,128)
    x=layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(10,activation='softmax')(x)
    model=models.Model(input,output)
    return model
