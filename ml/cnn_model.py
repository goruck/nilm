"""
Creates a CNN model for seq2point learning.

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(window_length):
    """Specifies a 1D seq2point model using the Keras Sequential API.

    Returns a TF-Keras model that, once trained, can be optimized and
    quantized by TF-Lite for the edge.

    Using a BatchNormalization layer in each convolution and dense layer
    will increase accuracy but since its not supported by TF-Lite its
    not being used here.

    Dropout (input for dense or spatial for convolution) not needed
    because the train dataset is very large.

    Kernel L2 regularizer was tried but had little benefit for much longer
    training time. Different datasets and / or network might benefit.

    Args:
        window_length: input length of model, mains power timeseries.

    Returns:
        TF-Keras model.

    Raises:
        Nothing.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_length,)),
        tf.keras.layers.Reshape(target_shape=(window_length, 1)),

        tf.keras.layers.Convolution1D(
            filters=32, kernel_size=5, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=64, kernel_size=3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=128, kernel_size=3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=256, kernel_size=3, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=512, kernel_size=3, padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(1, activation='linear')
    ])