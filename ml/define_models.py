"""Keras models for NILM seq2point learning.

There are several different architectures defined in this module.
One is transformer-based, all others are based on CNN backbones.

Copyright (c) 2022~2023 Lindo St. Angel
"""

import tensorflow as tf

from transformer_model import (
    NILMTransformerModel,
    NILMTransformerModelFit,
    L2NormPooling1D,
    PositionEmbedding,
    AddNormalization,
    TransformerBlock,
    RelativePositionEmbedding
)

def transformer_fun(window_length=599, dropout_rate=0.1, d_model=256) -> tf.keras.Model:
    """Specifies a transformer-based Keras Functional model."""
    latent_len = window_length // 2

    inp = tf.keras.Input(batch_shape=(None, window_length, 1))
    x = tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, padding='same')(inp)
    x = L2NormPooling1D(pool_size=2)(x)
    p = PositionEmbedding(max_length=window_length)(x)
    x = AddNormalization()(x, p)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = TransformerBlock(d_model, 2, d_model * 4, dropout_rate)(x, mask=None)
    x = TransformerBlock(d_model, 2, d_model * 4, dropout_rate)(x, mask=None)
    r = RelativePositionEmbedding(max_length=latent_len)(x)
    x = AddNormalization()(x, r)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    out = tf.keras.layers.Dense(units=1, activation='linear')(x)
    return tf.keras.Model(inp, out)

def transformer(
        window_length=599,
        drop_out=0.1,
        d_model=256,
        **kwargs
    ) -> tf.keras.Model:
    """Specifies a transformer-based model to be trained in a loop."""
    return NILMTransformerModel(
        window_length=window_length,
        drop_out=drop_out,
        hidden=d_model,
        **kwargs
    )

def transformer_fit(
        window_length=599,
        drop_out=0.1,
        threshold=0.5,
        d_model=256,
        c0=1.0,
        **kwargs
    ) -> tf.keras.Model:
    """Specifies a transformer-based model to be trained by .fit()."""
    return NILMTransformerModelFit(
        window_length=window_length,
        drop_out=drop_out,
        threshold=threshold,
        hidden=d_model,
        c0=c0,
        **kwargs
    )

def cnn() -> tf.keras.Sequential:
    """Specifies a 1D seq2point cnn model using the Keras Sequential API.

    Returns a TF-Keras model that, once trained, can be optimized and
    quantized by TF-Lite for the edge.

    The model expects inputs with shape = (batch_size, sequence_length, 1)
    and its output has shape = (batch_size, 1).

    This code results in a model that has about 2x the inference rate than
    the 2D version below ('cnn_fun'). TODO - figure out why.

    Args:
        None.

    Returns:
        TF-Keras model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Convolution1D(16, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(2, padding='same'),

            tf.keras.layers.Convolution1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(2, padding='same'),

            tf.keras.layers.Convolution1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(2, padding='same'),

            tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(2, padding='same'),

            tf.keras.layers.Convolution1D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ],
        name='cnn'
    )

def cnn_fun(window_length=599,
        conv_l2=0,
        dense_l2=0,
        batch_norm=False,
        drop_rate=0.3) -> tf.keras.Model:
    """Specifies a 1D seq2point model using the Keras Functional API.

    Returns a TF-Keras model that, once trained, can be optimized and
    quantized by TF-Lite for the edge.

    This uses 2D convolutions as 1D equivalents to ensure maximum
    compatibility with Tensorflow downstream processing such as 
    quantization aware training and pruning for on-device inference.

    Args:
        window_length: model input length based on time series window
        conv_l2: L2 regularization factor for conv layers
        dense_l2: L2 regularization factor for dense layer(s)
        batch_norm: If true adds batch normalization
        drop_rate: Drop rate for Dropout layers

    Returns:
        TF-Keras model.
    """
    input_layer = tf.keras.layers.Input(shape=(window_length,))
    input_layer_reshape = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    conv_1 = tf.keras.layers.Convolution2D(
        filters=16, kernel_size=(1, 5), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(input_layer_reshape)
    if batch_norm:
        conv_1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv_1) # TODO: synch True causes training to hang after 'tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8600'
    conv_1 = tf.keras.layers.Activation('relu')(conv_1)
    conv_1 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_1)

    conv_2 = tf.keras.layers.Convolution2D(
        filters=32, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_1)
    if batch_norm:
        conv_2 = tf.keras.layers.BatchNormalization(synchronized=True)(conv_2)
    conv_2 = tf.keras.layers.Activation('relu')(conv_2)
    conv_2 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_2)

    conv_3 = tf.keras.layers.Convolution2D(
        filters=64, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_2)
    if batch_norm:
        conv_3 = tf.keras.layers.BatchNormalization(synchronized=True)(conv_3)
    conv_3 = tf.keras.layers.Activation('relu')(conv_3)
    conv_3 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_3)

    conv_4 = tf.keras.layers.Convolution2D(
        filters=128, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_3)
    if batch_norm:
        conv_4 = tf.keras.layers.BatchNormalization(synchronized=True)(conv_4)
    conv_4 = tf.keras.layers.Activation('relu')(conv_4)
    conv_4 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same')(conv_4)

    conv_5 = tf.keras.layers.Convolution2D(
        filters=256, kernel_size=(1, 3), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(conv_4)
    if batch_norm:
        conv_5 = tf.keras.layers.BatchNormalization(synchronized=True)(conv_5)
    conv_5 = tf.keras.layers.Activation('relu')(conv_5)

    flatten_layer = tf.keras.layers.Flatten()(conv_5)

    flatten_layer = tf.keras.layers.Dropout(rate=drop_rate)(flatten_layer)

    label_layer = tf.keras.layers.Dense(1024,
        kernel_regularizer=tf.keras.regularizers.L2(dense_l2))(flatten_layer)
    if batch_norm:
        label_layer = tf.keras.layers.BatchNormalization(synchronized=True)(label_layer)
    label_layer = tf.keras.layers.Activation('relu')(label_layer)

    label_layer = tf.keras.layers.Dropout(rate=drop_rate)(label_layer)

    output_layer = tf.keras.layers.Dense(1, activation='linear')(label_layer)

    return tf.keras.models.Model(
        inputs=input_layer, outputs=output_layer, name='cnn')

def fcn(window_length=599, conv_l2=0, batch_norm=False) -> tf.keras.Model:
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
    input_layer = tf.keras.layers.Input(shape=(window_length,))
    input_layer_reshape = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    conv_1 = tf.keras.layers.Convolution2D(
        filters=16, kernel_size=(1, 5), padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(conv_l2))(input_layer_reshape)
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

def resnet(window_length, n_feature_maps=16, batch_norm=False) -> tf.keras.Model:
    """Specifies a resnet seq2point model using the Keras Functional API.

    This is a resnet version of the model.

    Args:
        window_length: model input length based on time series window
        n_feature_maps: number of feature maps to use in conv layers
        batch_norm: If true adds batch normalization. 

    Returns:
        TF-Keras model.
    """
    input_layer = tf.keras.layers.Input(shape=(window_length,))
    input_layer_reshape = tf.keras.layers.Reshape(
        target_shape=(1, window_length, 1))(input_layer)

    # Block 1.

    conv_x = tf.keras.layers.Conv2D(
        filters=n_feature_maps, kernel_size=(1, 5), padding='same')(input_layer_reshape)
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