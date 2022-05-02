"""
Creates a CNN model for seq2point learning.

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(window_length):
    """Specifies a 1D seq2point model using the Keras Sequential API."""
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_length,)),
        tf.keras.layers.Reshape(target_shape=(window_length, 1)),

        tf.keras.layers.Convolution1D(
            filters=32, kernel_size=5, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=64, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=128, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=256, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=512, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=1024, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.SpatialDropout1D(0.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation='linear')
    ])