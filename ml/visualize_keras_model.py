"""Visualize Keras Model"""

import tensorflow as tf

import define_models

model = define_models.cnn()
model.build((None, 599, 1))
tf.keras.utils.plot_model(
    model, to_file='../img/cnn_model_plot.png', show_shapes=True, show_layer_names=True
)

model = define_models.transformer_fun()
tf.keras.utils.plot_model(
    model, to_file='../img/transformer_model_plot.png', show_shapes=True, show_layer_names=True
)