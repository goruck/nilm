"""
Creates a CNN model for seq2point learning.

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(input_window_length):
    """Specifies a seq2point model using the Keras Sequential API."""
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_window_length,)),
        tf.keras.layers.Reshape(target_shape=(1, input_window_length, 1)),
        tf.keras.layers.Convolution2D(
            filters=30, kernel_size=(10, 1), strides=(1, 1), padding='same',
            activation='relu'),
        tf.keras.layers.Convolution2D(
            filters=30, kernel_size=(8, 1), strides=(1, 1), padding='same',
            activation='relu'),
        tf.keras.layers.Convolution2D(
            filters=40, kernel_size=(6, 1), strides=(1, 1), padding='same',
            activation='relu'),
        tf.keras.layers.Convolution2D(
            filters=50, kernel_size=(5, 1), strides=(1, 1), padding='same',
            activation='relu'),
        tf.keras.layers.Convolution2D(
            filters=50, kernel_size=(5, 1), strides=(1, 1), padding='same',
            activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])