"""Creates Keras models for seq2point learning.

There are several different architectures defined in this module.
All are based on CNN backbones.

The function "create_model" generates the model that currently
gives the best results.

The models should be fitted with a subset of the training data
because the entire dataset contains redundant information that
leads to over-fitting in a single epoch. The size of this subset
varies by appliance type but 10M samples is good starting place.

Copyright (c) 2022 Lindo St. Angel
"""

import tensorflow as tf

def create_model(window_length=599, conv_l2=0, dense_l2=0, batch_norm=False):
    """Specifies a 1D seq2point model using the Keras Functional API.

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
        batch_norm: If true adds batch normalization. 

    Returns:
        TF-Keras model.
    """
    input_layer = tf.keras.layers.Input(shape=(1, window_length))
    input_layer = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    conv_1 = tf.keras.layers.Convolution2D(
        filters=16, kernel_size=(1, 5), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(input_layer)
    if batch_norm:
        conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation('relu')(conv_1)
    conv_1 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_1)

    conv_2 = tf.keras.layers.Convolution2D(
        filters=32, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_1)
    if batch_norm:
        conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation('relu')(conv_2)
    conv_2 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_2)

    conv_3 = tf.keras.layers.Convolution2D(
        filters=64, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_2)
    if batch_norm:
        conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation('relu')(conv_3)
    conv_3 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_3)

    conv_4 = tf.keras.layers.Convolution2D(
        filters=128, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_3)
    if batch_norm:
        conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Activation('relu')(conv_4)
    conv_4 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_4)

    conv_5 = tf.keras.layers.Convolution2D(
        filters=256, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_4)
    if batch_norm:
        conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Activation('relu')(conv_5)

    flatten_layer = tf.keras.layers.Flatten()(conv_5)

    label_layer = tf.keras.layers.Dense(1024,
        kernel_regularizer=tf.keras.regularizers.L2(dense_l2))(flatten_layer)
    if batch_norm:
        label_layer = tf.keras.layers.BatchNormalization()(label_layer)
    label_layer = tf.keras.layers.Activation('relu')(label_layer)

    output_layer = tf.keras.layers.Dense(1, activation='linear')(label_layer)

    return tf.keras.models.Model(
        inputs=input_layer, outputs=output_layer, name='cnn')

def create_model_fcn(window_length=599, conv_l2=0, batch_norm=False):
    """Specifies a 1D seq2point model using the Keras Functional API.

    Returns a TF-Keras model that, once trained, can be optimized and
    quantized by TF-Lite for the edge.

    This is a fully connected version of the model.

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
        batch_norm: If true adds batch normalization. 

    Returns:
        TF-Keras model.
    """
    input_layer = tf.keras.layers.Input(shape=(1, window_length))
    input_layer = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    conv_1 = tf.keras.layers.Convolution2D(
        filters=16, kernel_size=(1, 5), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(input_layer)
    if batch_norm:
        conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation('relu')(conv_1)

    conv_2 = tf.keras.layers.Convolution2D(
        filters=32, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_1)
    if batch_norm:
        conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation('relu')(conv_2)

    conv_3 = tf.keras.layers.Convolution2D(
        filters=64, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_2)
    if batch_norm:
        conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation('relu')(conv_3)

    conv_4 = tf.keras.layers.Convolution2D(
        filters=128, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_3)
    if batch_norm:
        conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Activation('relu')(conv_4)

    conv_5 = tf.keras.layers.Convolution2D(
        filters=256, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_4)
    if batch_norm:
        conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Activation('relu')(conv_5)

    gap_layer = tf.keras.layers.GlobalAveragePooling2D()(conv_5)

    output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

    return tf.keras.models.Model(
        inputs=input_layer, outputs=output_layer, name='fcn')

def create_model_resnet(window_length=599, n_feature_maps=16, batch_norm=False):
    """Specifies a resnet seq2point model using the Keras Functional API.

    This is a resnet version of the model.

    Args:
        window_length: model input length based on time series window
        n_feature_maps: number of feature maps to use in conv layers
        batch_norm: If true adds batch normalization. 

    Returns:
        TF-Keras model.
    """
    input_layer = tf.keras.layers.Input(shape=(1, window_length))
    input_layer = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    # Block 1.

    conv_x = tf.keras.layers.Conv2D(
        filters=n_feature_maps, kernel_size=(1, 5), padding='same')(input_layer)
    if batch_norm:
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(
        filters=n_feature_maps, kernel_size=(1, 3), padding='same')(conv_x)
    if batch_norm:
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(
        filters=n_feature_maps, kernel_size=(1, 3), padding='same')(conv_y)
    if batch_norm:
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # Expand channels for the sum.
    shortcut_y = tf.keras.layers.Conv2D(
        filters=n_feature_maps, kernel_size=(1, 1), padding='same')(input_layer)
    if batch_norm:
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

    # Block 2.

    conv_x = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 5), padding='same')(output_block_1)
    if batch_norm:
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_x)
    if batch_norm:
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_y)
    if batch_norm:
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # Expand channels for the sum.
    shortcut_y = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 1), padding='same')(output_block_1)
    if batch_norm:
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

    # Block 3.

    conv_x = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 5), padding='same')(output_block_2)
    if batch_norm:
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_x)
    if batch_norm:
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv2D(
        filters=n_feature_maps * 2, kernel_size=(1, 3), padding='same')(conv_y)
    if batch_norm:
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # No need to expand channels because they are equal.
    if batch_norm:
        shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

    # Final.

    gap_layer = tf.keras.layers.GlobalAveragePooling2D()(output_block_3)

    output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

    return tf.keras.models.Model(
        inputs=input_layer, outputs=output_layer, name='resnet')