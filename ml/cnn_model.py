"""
Creates a CNN model for seq2point learning.

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(input_window_length, n_dense=1) -> tf.keras.Model:
    """Specifies a seq2point model using the Keras functional API."""

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))

    reshape_layer = tf.keras.layers.Reshape(
        (-1, input_window_length, 1))(input_layer)

    conv_layer_1 = tf.keras.layers.Convolution2D(
        filters=30, kernel_size=(10, 1), strides=(1, 1), padding='same',
        activation='relu')(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(
        filters=30, kernel_size=(8, 1), strides=(1, 1), padding='same',
        activation='relu')(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(
        filters=40, kernel_size=(6, 1), strides=(1, 1), padding='same',
        activation='relu')(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(
        filters=50, kernel_size=(5, 1), strides=(1, 1), padding='same',
        activation='relu')(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(
        filters=50, kernel_size=(5, 1), strides=(1, 1), padding='same',
        activation='relu')(conv_layer_4)

    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)

    d0 = tf.keras.layers.Dense(1024, activation='relu')(flatten_layer)
    
    if n_dense == 1:
        label_layer = d0
    if n_dense == 2:
        d1 = tf.keras.layers.Dense(1024, activation='relu')(d0)
        label_layer = d1
    if n_dense == 3:
        d1 = tf.keras.layers.Dense(1024, activation='relu')(d0)
        d2 = tf.keras.layers.Dense(1024, activation='relu')(d1)
        label_layer = d2

    output_layer = tf.keras.layers.Dense(1, activation='linear')(label_layer)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)