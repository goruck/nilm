"""Creates a CNN model for seq2point learning.

***DEPRECATED - USE "define_models.py" INSTEAD***

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(window_length, conv_l2=0, dense_l2=0):
    """Specifies a 1D seq2point model using the Keras Sequential API.

    Returns a TF-Keras model that, once trained, can be optimized and
    quantized by TF-Lite for the edge.

    Primary model used in training and currently gives best results.

    This uses 2D convolutions as 1D equivalents to ensure maximum
    compatibility with Tensorflow downstream processing such as 
    quantization aware training and pruning for on-device inference.

    This model is usually trained with large batch sizes (~1000) so
    batch normalization has little benefit based on results to date.

    The data sets used to train this model are large (~10's M samples)
    so kernel regularization has little or no benefit.

    Args:
        window_length: model input length based on time series window
        conv_l2: L2 regularization factor for conv layers
        dense_l2: L2 regularization factor for dense layer(s)

    Returns:
        TF-Keras model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_length,)),
        tf.keras.layers.Reshape(target_shape=(1, window_length, 1)),

        tf.keras.layers.Convolution2D(
            filters=16, kernel_size=(1, 5), padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(conv_l2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same'),

        tf.keras.layers.Convolution2D(
            filters=32, kernel_size=(1, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(conv_l2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same'),

        tf.keras.layers.Convolution2D(
            filters=64, kernel_size=(1, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(conv_l2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same'),

        tf.keras.layers.Convolution2D(
            filters=128, kernel_size=(1, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(conv_l2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same'),
        
        tf.keras.layers.Convolution2D(
            filters=256, kernel_size=(1, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.L2(conv_l2)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024,
            kernel_regularizer=tf.keras.regularizers.L2(dense_l2)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(1, activation='linear')
    ])

def create_model_1D(window_length):
    """Specifies a 1D seq2point model using the Keras Sequential API.

    This is a 1D version of the primary model.

    Args:
        window_length: model input length based on time series window

    Returns:
        TF-Keras model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_length,)),
        tf.keras.layers.Reshape(target_shape=(window_length, 1)),

        tf.keras.layers.Convolution1D(
            filters=16, kernel_size=5, padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),

        tf.keras.layers.Convolution1D(
            filters=32, kernel_size=3, padding='same'),
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

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(1, activation='linear')
    ])

def create_model_fcn(window_length):
    """Specifies a fully connected seq2point model using the Keras Sequential API.

    This is a fully connected version of the model.

    Args:
        window_length: model input length based on time series window

    Returns:
        TF-Keras model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_length,)),
        tf.keras.layers.Reshape(target_shape=(1, window_length, 1)),

        tf.keras.layers.Convolution2D(
            filters=16, kernel_size=(1, 5), padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(
            filters=32, kernel_size=(1, 3), padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(
            filters=64, kernel_size=(1, 3), padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(
            filters=128, kernel_size=(1, 3), padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(
            filters=256, kernel_size=(1, 3), padding='same'),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(1, activation='linear')
    ])

def create_model_resnet(window_length, n_feature_maps=16):
    """Specifies a resnet seq2point model using the Keras Functional API.

    This is a resnet version of the model.

    Args:
        window_length: model input length based on time series window
        n_feature_maps: number of feature maps to use in conv layers

    Returns:
        TF-Keras model.
    """

    input_layer = tf.keras.layers.Input(window_length)
    input_layer = tf.keras.layers.Reshape(target_shape=(1, window_length, 1))(input_layer)

    # Block 1.

    conv_x = tf.keras.layers.Conv2D(filters=n_feature_maps, kernel_size=(1, 5), padding='same')(input_layer)
    #conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(filters=n_feature_maps, kernel_size=(1, 3), padding='same')(conv_x)
    #conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(filters=n_feature_maps, kernel_size=(1, 3), padding='same')(conv_y)
    #conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # Expand channels for the sum.
    shortcut_y = tf.keras.layers.Conv2D(filters=n_feature_maps, kernel_size=(1, 1), padding='same')(input_layer)
    #shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

    # Block 2.

    conv_x = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 5), padding='same')(output_block_1)
    #conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_x)
    #conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # Expand channels for the sum.
    shortcut_y = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 1), padding='same')(output_block_1)
    #shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

    # Block 3.

    conv_x = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 5), padding='same')(output_block_2)
    #conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_x)
    #conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_y)
    #conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # No need to expand channels because they are equal.
    #shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

    # Final.

    gap_layer = tf.keras.layers.GlobalAveragePooling2D()(output_block_3)

    output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)