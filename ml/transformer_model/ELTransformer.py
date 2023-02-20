import math
import numpy as np

#import torch
#from torch import nn
#import torch.nn.functional as F

import tensorflow as tf
from tensorflow import matmul, reshape, shape, transpose, cast, float32, tanh, pow
from tensorflow import keras
from keras.layers import LayerNormalization, Layer, Dense, Flatten, Dropout, Embedding, Dropout, MultiHeadAttention, Conv1D, MaxPooling1D, Reshape
from keras.backend import softmax
from keras.models import Functional
from keras import Model


class GELU(Layer):
    def __init__(self, name='GELU', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3))))


class PositionalEmbedding(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(*kwargs)
        #print(f'max_len: {max_len}')
        #print(f'd_model: {d_model}')
        self.pe = Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        #print(f'pos x shape: {tf.shape(x)}')
        batch = tf.shape(x)[0]
        #print(f'pos x batch size: {batch}')

        #x = tf.reshape(x, [1000, 599])
        #print(f'pos x shape: {tf.shape(x)}')
        #batch = tf.shape(x)[0]
        #print(f'pos x batch size: {batch}')

        position_indices = tf.range(tf.shape(x)[1])
        #print(f'pos indices shape: {tf.shape(position_indices)}')
        #print(f'pos indices: {position_indices}')

        position_indices = tf.expand_dims(position_indices, axis=0)

        position_indices = tf.repeat(position_indices, batch, axis=0)

        #position_indices = tf.reshape(position_indices, [batch, 599])
        #print(f'pos indices shape: {tf.shape(position_indices)}')
        #print(f'pos indices: {position_indices}')

        #position_indices = tf.reshape(position_indices, shape=(1000, 599))
        #print(f'pos indices shape: {tf.shape(position_indices)}')
        #print(f'pos indices: {position_indices}')

        #position_indices = np.random.randint(599, size=(1000, 299))
        #print(f'pos indices shape: {tf.shape(position_indices)}')
        #print(f'pos indices: {position_indices}')

        return self.pe(position_indices)


class RelativePositionalEmbedding(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.rpe = Embedding(input_dim=max_len, output_dim=d_model)

    def _get_relative_position_indices(self, len:int) -> tf.Tensor:
        rel_position_indices =  [abs(i - len // 2) for i in range(len)]
        #print(rel_position_indices)
        return tf.convert_to_tensor(rel_position_indices, dtype=tf.float32)

    def call(self, x):
        #print(f'rel pos x shape: {tf.shape(x)}')
        batch = tf.shape(x)[0]
        #print(f'rel pos x batch size: {batch}')

        relative_position_indices = self._get_relative_position_indices(self.max_len)
        
        relative_position_indices = tf.expand_dims(relative_position_indices, axis=0)

        relative_position_indices = tf.repeat(relative_position_indices, batch, axis=0)
        #print(f'rel pos ind: {relative_position_indices}')
        #print(tf.shape(relative_position_indices))

        return self.rpe(relative_position_indices)

"""
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
"""

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

"""
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
"""

class MultiHeadedAttention(Layer):
    def __init__(self, h, d_model, **kwargs):
        super().__init__(**kwargs)
        assert d_model % h == 0
        self.heads = h              # Number of attention heads to use
        self.d_model = d_model      # Dimensionality of the model
        self.d_k = d_model // h     # Dimensionality of the linearly projected queries and keys
        self.d_v = d_model // h     # Dimensionality of the linearly projected values
        self.attention = DotProductAttention()# Scaled dot product attention
        self.W_q = Dense(self.d_k)  # Learned projection matrix for the queries
        self.W_k = Dense(self.d_k)  # Learned projection matrix for the keys
        self.W_v = Dense(self.d_v)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)   # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries,
        # keys, and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head
        # attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)


"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed:
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (queryprint(f'rel pos shape: {tf.shape(rel_pos)}'), key, value))]

        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
"""


class PositionwiseFeedForward(Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = GELU()

    def call(self, x):
        # The input is passed into the two fully-connected layers, with GELU activation.
        x_fc1 = self.fully_connected1(x)

        return self.activation(self.fully_connected2(x_fc1))


class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)


class TransformerBlock(Layer):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = MultiHeadedAttention(attn_heads, hidden, **kwargs)
        self.dropout1 = Dropout(rate=dropout)
        self.add_norm1 = AddNormalization()
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.dropout2 = Dropout(rate=dropout)
        self.add_norm2 = AddNormalization()

    def call(self, x, mask, training=None):
        # Multi-head attention layer
        multihead_output = self.attention(x, x, x, mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)


class ELTransformerModel(Model):
    def __init__(self, window_size, drop_out, **kwargs):
        super().__init__(**kwargs)

        self.original_len = window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = drop_out

        self.hidden = 256
        self.heads = 2
        self.n_layers = 2

        self.conv = Conv1D(filters=self.hidden, kernel_size=5, padding='same', input_shape=(window_size, 1))

        self.pool = MaxPooling1D()

        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)

        self.layer_norm = LayerNormalization()

        self.dropout = Dropout(rate=self.dropout_rate)

        self.transformer_blocks = [TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)]

        self.relative_position = RelativePositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)

        self.dense1 = Dense(units=self.hidden, activation='relu')

        self.flatten = Flatten()

        self.dense2 = Dense(units=1)

    def _reshape_input(self, x: tf.Tensor) -> tf.Tensor:
            batch_size = tf.shape(x)[0]
            return tf.reshape(x, [batch_size, self.original_len, 1])

    def call(self, sequence:tf.Tensor, mask:bool=None, training:bool=None):
        #print(f'in seq shape: {tf.shape(sequence)}')
        #print(f'seq: {sequence}')

        sequence = self._reshape_input(sequence)
        #print(f'reshaped seq shape: {tf.shape(sequence)}')
        #print(f'reshaped seq: {sequence}')

        conv = self.conv(sequence)
        #print(f'conv: {tf.shape(conv)}')
        x_token = self.pool(conv)
        #print(f'x_token shape: {tf.shape(x_token)}')
        #print(f'x_token: {x_token}')

        pos = self.position(x_token)
        # Expected output shape = (batch_size, latent_len, d_model)
        #print(f'pos shape: {tf.shape(pos)}')
        #print(f'pos: {pos}')

        embedding = x_token + pos
        #print(f'embedding shape: {tf.shape(embedding)}')

        x = self.dropout(embedding, training=training)

        # Pass on the positional encoded values to each transformer.
        for _, transformer in enumerate(self.transformer_blocks):
            x = transformer(x, mask)

        rel_pos = self.relative_position(x)

        #print(f'rel pos shape: {tf.shape(rel_pos)}')
        #print(f'x shape before adding rel pos: {tf.shape(x)}')

        x = x + rel_pos
        x = self.dropout(self.layer_norm(x), training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        #print(f'final x shape: {tf.shape(x)}')
        #print(f'final x: {x}')
        return x