import math
import numpy as np

import tensorflow as tf
from tensorflow import matmul, reshape, shape, transpose, cast, float32, tanh, pow, float16
from tensorflow import keras
from keras.layers import Activation, LayerNormalization, Layer, Dense, Flatten, Dropout, Embedding, Dropout, MultiHeadAttention, Conv1D, MaxPooling1D, Reshape, Concatenate, Conv1DTranspose, GlobalAveragePooling1D
from keras.backend import softmax
from keras.activations import gelu
from keras.initializers import TruncatedNormal
from keras import Model
from tensorflow_models import nlp
import keras_nlp

class GELU(Layer):
    def __init__(self, name='GELU', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3))))

class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.
  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```
  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.
  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "seq_axis": self._seq_axis,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    weight_sequence_length = self._max_length

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
        trainable=True)

    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]
    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)


class PositionalEmbeddingOld(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(*kwargs)
        self.max_len = max_len
        self.pe = Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        batch = tf.shape(x)[0]

        position_indices = tf.range(self.max_len)

        position_indices = tf.expand_dims(position_indices, axis=0)

        position_indices = tf.repeat(position_indices, batch, axis=0)

        return self.pe(position_indices)
    
    
class PositionalEmbedding(Layer):
    def __init__(self, max_len, d_model, initializer='glorot_uniform', seq_axis=1, **kwargs):
        super().__init__(*kwargs)
        self._initializer = tf.keras.initializers.get(initializer)
        self._seq_axis = seq_axis
        self._position_embeddings = self.add_weight(
            'embeddings',
            shape=[max_len, d_model],
            initializer=self._initializer)

    def call(self, x):
        input_shape = tf.shape(x)
        actual_seq_len = input_shape[self._seq_axis]
        position_embeddings = self._position_embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in x.get_shape().as_list()]
        new_shape[self._seq_axis] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)

class RelativePositionEmbeddingOld(tf.keras.layers.Layer):
  """Creates a relative positional embedding.
  Example:
  ```python
  position_embedding = RelativePositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```
  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.
  Reference: This layer creates a relative positional embedding as described in
  [Efficient Localness Transformer for Smart Sensor-Based Energy Disaggregation
  ](https://arxiv.org/abs/2203.16537).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "seq_axis": self._seq_axis,
    }
    base_config = super(RelativePositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    length = dimension_list[-2]
    weight_sequence_length = self._max_length
    
    self.m = length // 2

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
        trainable=True)
    
    """
    m = actual_seq_len // 2
    
    if actual_seq_len % 2 == 0:
        self._rel_pe = tf.concat([pe[:m], pe[:m][::-1]], axis=0)
    else:
        self._rel_pe = tf.concat([pe[:m], pe[m::-1]], axis=0)
    """
    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]

    m = actual_seq_len // 2 # index of sequence mid-point

    # This works only for odd length sequences - TODO Fix for even lengths
    position_embeddings = tf.concat(
       [position_embeddings[:m], position_embeddings[m::-1]], axis=0)

    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)
  

class RelativePositionEmbeddingAlt(tf.keras.layers.Layer):
  """Creates a relative positional embedding.
  Example:
  ```python
  position_embedding = RelativePositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```
  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      'glorot_uniform.
    seq_axis: The axis of the input tensor where we add the embeddings.
  Reference: This layer creates a relative positional embedding as described in
  [Efficient Localness Transformer for Smart Sensor-Based Energy Disaggregation
  ](https://arxiv.org/abs/2203.16537).
  """

  def __init__(self,
               max_length,
               initializer='glorot_uniform',
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        'max_length': self._max_length,
        'initializer': tf.keras.initializers.serialize(self._initializer),
        'seq_axis': self._seq_axis,
    }
    base_config = super(RelativePositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]

    # Embeddings need only to be half the max sequence length
    # because they designed to be symmetric.
    weight_sequence_length = tf.math.ceil(self._max_length / 2)
    self._position_embeddings = self.add_weight(
        'embeddings',
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
        trainable=True)
    
    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]

    # Actual sequence mid-point.
    m = actual_seq_len // 2

    # Embeddings are identical when elements have same distance
    # to the input sequence midpoint.
    position_embeddings = tf.concat(
       [position_embeddings[:m], position_embeddings[m::-1]], axis=0)

    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)

class RelativePositionalEmbeddingOld(Layer):
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
    
class RelativePositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.
  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```
  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.
  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "seq_axis": self._seq_axis,
    }
    base_config = super(RelativePositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    weight_sequence_length = self._max_length
    
    self.m = dimension_list[-2] // 2

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
        trainable=True)
    
    """
    m = actual_seq_len // 2
    
    if actual_seq_len % 2 == 0:
        self._rel_pe = tf.concat([pe[:m], pe[:m][::-1]], axis=0)
    else:
        self._rel_pe = tf.concat([pe[:m], pe[m::-1]], axis=0)
    """
    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]

    m = actual_seq_len // 2

    position_embeddings = tf.concat([position_embeddings[:m], position_embeddings[m::-1]], axis=0)

    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(cast(d_k, queries.dtype))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation.
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
    
    
class LinearDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask):

        rho_q = softmax(queries, axis=1)    # softmax wrt query rows
        rho_k = softmax(keys)               # softmax wrt key columns

        return matmul(rho_q, matmul(rho_k, values, transpose_a=True))


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
        self.W_o = Dense(self.d_model)   # Learned projection matrix for the multi-head output

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


class LocalAttention(Layer):
    def __init__(self, l_win, d_loc, d_model, **kwargs):
        super().__init__(**kwargs)

        self.l_win = l_win
        self.d_loc = d_loc
        self.d_model = d_model

        self.W_q = Dense(self.d_loc)  # Learned projection matrix for the queries
        self.W_k = Dense(self.d_loc)  # Learned projection matrix for the keys
        self.W_v = Dense(self.d_loc)  # Learned projection matrix for the values
        self.W_o = Dense(self.d_model)   # Learned projection matrix for the multi-head output

        self.attention = DotProductAttention()# Scaled dot product attention

    def build(self, input_shape):
        batch = input_shape[0]
        len = input_shape[1]

        self.batch = batch
        self.len = len
        self.pad = self.l_win - (len % self.l_win)
        self.padded_len = len + self.pad
        self.num_l_win = self.padded_len // self.l_win

        super().build(input_shape)

    def call(self, queries, keys, values, mask=None):
        #print(f' q: {queries}')
        paddings = tf.constant([[0, 0], [self.pad, 0], [0, 0]])

        padded_q = tf.pad(queries, paddings)
        padded_k = tf.pad(keys, paddings)
        padded_v = tf.pad(values, paddings)
        #print(f'padded q shape: {tf.shape(padded_q)}')

        #print(f'batch: {self.batch}, l_win: {self.l_win}, num l_win: {self.num_l_win}')

        padded_q_reshape = reshape(padded_q, shape=[self.batch, self.num_l_win, self.l_win, -1])
        #print(f'padded q reshape: {tf.shape(padded_q_reshape)}')
        projection_q = self.W_q(padded_q_reshape)
        #print(f'projection q shape: {tf.shape(projection_q)}')

        padded_k_reshape = reshape(padded_k, shape=[self.batch, self.num_l_win // 3, self.l_win * 3, -1])
        #print(f'padded k reshape: {tf.shape(padded_k_reshape)}')
        padded_k_reshape = tf.repeat(padded_k_reshape, repeats=[3], axis=1)
        #print(f'padded k reshape repeat: {tf.shape(padded_k_reshape)}')
        projection_k = self.W_k(padded_k_reshape)
        #print(f'projection k shape: {tf.shape(projection_k)}')

        padded_v_reshape = reshape(padded_v, shape=[self.batch, self.num_l_win // 3, self.l_win * 3, -1])
        #print(f'padded v reshape: {tf.shape(padded_v_reshape)}')
        padded_v_reshape = tf.repeat(padded_v_reshape, repeats=[3], axis=1)
        #print(f'padded v reshape repeat: {tf.shape(padded_v_reshape)}')
        projection_v = self.W_v(padded_v_reshape)
        #print(f'projection v shape: {tf.shape(projection_v)}')

        #q_kt = matmul(padded_q_reshape, padded_k_reshape, transpose_b=True)
        #print(q_kt)

        #q_kt_v = matmul(q_kt, padded_v_reshape)
        #print(q_kt_v)

        #q_kt_v_reshape = reshape(q_kt_v, shape=[self.batch, self.padded_len, self.d_loc])
        #print(q_kt_v_reshape)

        o = self.attention(projection_q, projection_k, projection_v, self.d_loc, mask)
        #print(f'o shape: {tf.shape(o)}')
        o_reshape = reshape(o, shape=[self.batch, self.padded_len, self.d_loc])
        o_reshape = o_reshape[:, self.pad:, :]
        #print(f'o rehape: {tf.shape(o_reshape)}')
        
        projection_o = self.W_o(o_reshape)
        #print(f'projection o: {tf.shape(projection_o)}')

        return projection_o


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
        super().__init__(**kwargs)
        self.attention = MultiHeadedAttention(attn_heads, hidden, **kwargs)
        #self.local_attention = LocalAttention(l_win=10, d_loc=32, d_model=hidden//2, **kwargs)
        self.dropout1 = Dropout(rate=dropout)
        self.add_norm1 = AddNormalization()
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.dropout2 = Dropout(rate=dropout)
        self.add_norm2 = AddNormalization()

        #self.W_o = Dense(hidden)

    def call(self, x, mask, training=None):
        # Multi-head attention layer
        multihead_output = self.attention(x, x, x, mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        #local_attention_output = self.local_attention(x, x, x, mask)
        #print(local_attention_output)
        #exit()

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)
        #multihead_output = self.dropout1(tf.concat([multihead_output, local_attention_output], axis=-1), training=training)
        #multihead_output = self.dropout1(local_attention_output, training=training)
        #print(tf.shape(multihead_output))

        #global_local_atten = self.dropout1(self.W_o(tf.concat([multihead_output, local_attention_output], axis=-1)), training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        #addnorm_output = self.add_norm1(x, global_local_atten)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)


class ELTransformerModel(Model):
    def __init__(self, window_length, drop_out, **kwargs):
        super().__init__(**kwargs)

        self.original_len = window_length
        self.latent_len = self.original_len // 2
        self.dropout_rate = drop_out

        self.hidden = 256
        self.decoder_hidden = 1024
        self.heads = 2
        self.n_layers = 2
        self.pool_size = 2

        # Deconvolution (aka Transposed Convolution) parameters.
        # Note restriction on choice of parameters checked below.
        # See "A guide to convolution arithmetic for deep learning"
        # https://arxiv.org/abs/1603.07285
        #self.deconv_k = 3
        #self.deconv_s = 2
        #if (window_length + 2 - self.deconv_k) % self.deconv_s != 0:
            #raise ValueError('Bad deconvolution parameters.')

        self.conv = Conv1D(filters=self.hidden, kernel_size=5, padding='same')

        self.pool = MaxPooling1D(pool_size=self.pool_size)

        #self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.position = PositionEmbedding(max_length=self.original_len)

        self.layer_norm = LayerNormalization()

        self.dropout1 = Dropout(rate=self.dropout_rate)
        self.dropout2 = Dropout(rate=self.dropout_rate)
        #self.dropout3 = Dropout(rate=self.dropout_rate)
        #self.dropout4 = Dropout(rate=self.dropout_rate)

        self.transformer_layers = [TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)]
        
        """
        self.transformer_layers = [nlp.layers.TransformerEncoderBlock(
            num_attention_heads=self.heads,
            inner_dim=self.hidden,
            inner_activation=(lambda x: gelu(x, approximate=True)),
            output_dropout=self.dropout_rate,
            attention_dropout=self.dropout_rate,
            inner_dropout=self.dropout_rate,
            attention_initializer=TruncatedNormal(stddev=0.02)) for _ in range(self.n_layers)]
        """

        #self.deconv = Conv1DTranspose(filters=self.hidden, kernel_size=self.deconv_k, strides=self.deconv_s)

        #self.relative_position = RelativePositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.relative_position = RelativePositionEmbedding(max_length=self.latent_len)

        self.dense1 = Dense(units=1024, activation='relu')

        #self.dense2 = Dense(units=256, activation='relu')

        #self.dense3 = Dense(units=64, activation='relu')

        self.flatten = Flatten()

        #self.GlobalAveragePool = GlobalAveragePooling1D()

        self.layer_norm2 = LayerNormalization()

        self.dense4 = Dense(units=1)

        # If training with mixed-precision, ensure model output is float32.
        # This helps to avoids numerical instability.
        # See https://www.tensorflow.org/guide/mixed_precision#building_the_model
        self.output_activation = Activation('linear', dtype='float32')

    def call(self, sequence:tf.Tensor, training:bool=None):
        # Expected input sequence shape = (batch_size, original_len)

        # Add sequence length axis.
        sequence = sequence[:, :, np.newaxis]
        #print(f'sequence: {sequence}')
        # Expected output shape = (batch_size, original_len, 1)

        ### Encoder Layers ###

        # Sequence positional encoding layer.
        # Some position indices will have same encoding according to the
        # "div" partition strategy since sequence length > embedding weight length.
        # 'pool_size' entries in the embedding matrix rows are skipped to match
        # upstream layer shape. This results in no loss of information since the
        # partition strategy results in multiple indices having the same encoding.
        # See tf.nn.embedding_lookup documentation for details on partition strategy.
        #position_embeddings = self.position(sequence)[:, ::self.pool_size, :]

        # Sequence positional encoding layer.
        # Skip 'pool_size' positional embedding weights to match upstream processing.
        # This causes every 'pool_size 'position indices to have same encoding.
        #position_embeddings = self.position(sequence)#[:, ::self.pool_size, :]
        # Expected output shape = (batch_size, latent_len, hidden)

        # Sequence positional encoding layer.
        extracted_features = self.pool(self.conv(sequence))
        # Expected output shape = (batch_size, latent_len, hidden)

        embeddings = extracted_features + self.position(extracted_features, training=training)
        # Expected output shape = (batch_size, latent_len, hidden)

        x = self.dropout1(self.layer_norm(embeddings), training=training)

        ### Transformer Layers ###

        # Assign importance weights to embeddings using back-to-back transformers.
        for _, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x, mask=None, training=training)
        # Expected output shape = {batch_size, latent_len, hidden}

        ### Decoder Layers ###

        rel_pos = self.relative_position(x)

        x = self.dense1(self.layer_norm2(x + rel_pos))

        # Expand xformer output to original length with transposed convolution.
        #x = self.deconv(x)
        # Expected output shape = (batch_size, original_len, hidden)

        #x = self.layer_norm2(x)

        x = self.flatten(x)

        # Generate normalized power predictions over sequence length.
        #x = self.dense1(x)
        # Expected output_size = (batch_size, original_len, decoder_hidden)

        x = self.dropout2(x, training=training)

        #x = self.dense2(x)

        #x = self.dropout3(x, training=training)

        #x = self.dense3(x)

        #x = self.dropout4(x, training=training)

        #x = self.flatten(x)
        # Expected output size = (batch_size, original_len * decoder_hidden)

        #x = self.GlobalAveragePool(x)

        # Apply sequence-to-point transformation.
        x = self.dense4(x)
        # Expected output_size = (batch_size, 1)
        return self.output_activation(x)