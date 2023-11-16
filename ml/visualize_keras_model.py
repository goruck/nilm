"""Visualize Keras Model"""

import tensorflow as tf

import define_models
from transformer_model import (
    MultiHeadedAttention, AddNormalization, PositionwiseFeedForward
)

model = define_models.cnn()
model.build((None, 599, 1))
tf.keras.utils.plot_model(
    model, to_file='../img/cnn_model_plot.png', show_shapes=True, show_layer_names=True
)

model = define_models.transformer_fun()
tf.keras.utils.plot_model(
    model, to_file='../img/transformer_model_plot.png', show_shapes=True, show_layer_names=True
)

def transformer_block(hidden, attn_heads, feed_forward_hidden, dropout) -> tf.keras.Model:
    """Specifies a Keras Functional model of a Transformer Block."""
    inp = tf.keras.Input(batch_shape=(None, 299, 256))
    x = MultiHeadedAttention(attn_heads, hidden//2, hidden//2, hidden)(inp, inp, inp, None)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    add_norm = AddNormalization()(x, inp)
    x = PositionwiseFeedForward(hidden, feed_forward_hidden)(add_norm)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    out = AddNormalization()(add_norm, x)
    return tf.keras.Model(inp, out)

model = transformer_block(256, 2, 4 * 256, 0.1)
tf.keras.utils.plot_model(
    model, to_file='../img/transformer_block_plot.png', show_shapes=True, show_layer_names=True
)
