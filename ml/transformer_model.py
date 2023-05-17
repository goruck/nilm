"""NILM transformer-based model.

Implemented by Keras subclassed layers and model.

Copyright (c) 2023 Lindo St. Angel
"""
import math

import tensorflow as tf

class GELU(tf.keras.layers.Layer):
    """Applies the Gaussian error linear unit (GELU) activation function."""

    def __init__(self, name='GELU', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(
           tf.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    

class L2NormPooling1D(tf.keras.layers.Layer):
    """Applies L2 norm average pooling over an input signal.

    If the L2 norm pooling is zero, the gradient of this function is not defined.
    This implementation adds `epsilon` to that quantity to maintain numerical stability.
    
    Args:
        Same as tf.keras.layers.AveragePooling1D except:
        epsilon: A small number added to the pooled output for numerical stability.
    """

    def __init__(self,
                 pool_size=2,
                 strides=2,
                 padding='valid',
                 data_format='channels_last',
                 epsilon=1e-10,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.epsilon = epsilon
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=self.pool_size,
                                                         strides=self.strides,
                                                         padding=self.padding,
                                                         data_format=self.data_format)

    def call(self, x):
        avg_pooled_squares = self.avg_pool(tf.square(x)) #* tf.cast(self.pool_size, dtype=x.dtype)
        return tf.sqrt(avg_pooled_squares + self.epsilon)
    

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
      'glorot_uniform'.
    seq_axis: The axis of the input tensor where we add the embeddings.
  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
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
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    weight_sequence_length = self._max_length

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
    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)
  

class RelativePositionEmbedding(tf.keras.layers.Layer):
  """Creates a relative positional embedding.
  Example:
  ```pythontensorflow loss nan square root
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
    weight_sequence_length = self._max_length // 2
    self._embeddings = self.add_weight(
        'embeddings',
        shape=[weight_sequence_length, width],
        initializer=self._initializer,
        trainable=True)
    
    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    embeddings = self._embeddings[:actual_seq_len, :]

    seq_indices = tf.range(actual_seq_len)

    # Make embeddings to have the same value when its elements have same
    # distance to the sequence midpoint.
    mid = actual_seq_len // 2
    dist_from_mid = tf.abs(seq_indices - mid)
    position_embeddings = tf.gather(embeddings, dist_from_mid)

    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)
  

class DotProductAttention(tf.keras.layers.Layer):
    """Dot-product attention layer, a.k.a. Luong-style attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = tf.matmul(
           queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, queries.dtype))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation.
        weights = tf.nn.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return tf.matmul(weights, values)


class MultiHeadedAttention(tf.keras.layers.Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).

    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector."""

    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.heads = h # Number of attention heads to use
        self.d_model = d_model # Dimensionality of the model
        self.d_k = d_k # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v # Dimensionality of the linearly projected values
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.W_q = tf.keras.layers.Dense(self.d_k)  # Learned projection matrix for the queries
        self.W_k = tf.keras.layers.Dense(self.d_k)  # Learned projection matrix for the keys
        self.W_v = tf.keras.layers.Dense(self.d_v)  # Learned projection matrix for the values
        self.W_o = tf.keras.layers.Dense(self.d_model)   # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))
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


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """PositionwiseFeedForward layer.

    This is an implementation of position wise feed-forward as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017)."""

    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = tf.keras.layers.Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = tf.keras.layers.Dense(d_model)  # Second fully connected layer
        self.activation = GELU()

    def call(self, x):
        # The input is passed into the two fully-connected layers, with GELU activation.
        x_fc1 = self.fully_connected1(x)

        return self.activation(self.fully_connected2(x_fc1))


class AddNormalization(tf.keras.layers.Layer):
    """AddNormaliztion Layer"""

    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)


class TransformerBlock(tf.keras.layers.Layer):
    """A Bert-style transformer encoder."""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, **kwargs):
        super().__init__(**kwargs)
        d_k = hidden // 2 # d_k must be an even multiple of hidden
        d_v = hidden // 2 # d_v must be an even multiple of "hidden
        self.attention = MultiHeadedAttention(attn_heads, d_k, d_v, hidden, **kwargs)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.add_norm1 = AddNormalization()
        self.feed_forward = PositionwiseFeedForward(hidden, feed_forward_hidden)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
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


class NILMTransformerModel(tf.keras.Model):
    """NILM model based on a BERT-style transformer-based encoder.

    Implements Non-Intrusive Load Monitoring (aka load disaggregation) by a network
    surrounding a BERT-style encoder. Inspired by the following references.

    1. "Efficient Localness Transformer for Smart Sensor-Based Energy Disaggregation"
    (https://arxiv.org/abs/2203.16537).

    2. "BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring"
    (http://nilmworkshop.org/2020/proceedings/nilm20-final88.pdf).

    3. "Building Transformer Models with Attention".
    (https://machinelearningmastery.com/transformer-models-with-attention/).

    This is implemented as a Keras subclassed Model and is meant to be trained by fit().

    Args:
        window_length: The length of the input sequence of aggregate power samples.
        drop_out: Drop out rate used in all drop out layers.
        sequence: The input sequence (used by the call method).
        training: Flag indicating training or inference (used by the call method).
    """

    def __init__(self, window_length:int, drop_out:float, **kwargs) -> None:
        super().__init__(**kwargs)

        self.original_len = window_length
        self.latent_len = self.original_len // 2
        self.dropout_rate = drop_out

        self.hidden = 256
        self.decoder_hidden = 1024
        self.heads = 2
        self.n_layers = 2
        self.pool_size = 2

        self.conv = tf.keras.layers.Conv1D(
           filters=self.hidden, kernel_size=5, padding='same')
        self.pool = L2NormPooling1D(pool_size=self.pool_size)
        self.position = PositionEmbedding(max_length=self.original_len)
        self.add_norm1 = AddNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.transformer_layers = [TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate)
            for _ in range(self.n_layers)]
        self.relative_position = RelativePositionEmbedding(max_length=self.latent_len)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        #self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(units=self.decoder_hidden, activation='relu')
        #self.flatten = tf.keras.layers.Flatten()
        self.dropout2 = tf.keras.layers.Dropout(rate=0.25)
        self.add_norm2 = AddNormalization()
        self.dense2 = tf.keras.layers.Dense(units=1)
        # If training with mixed-precision, ensure model output is float32.
        # This helps to avoids numerical instability.
        # See https://www.tensorflow.org/guide/mixed_precision#building_the_model
        self.output_activation = tf.keras.layers.Activation('linear', dtype=tf.float32)

    def train_step(self, data):
        """Function called by fit() that trains on every batch of data."""

        # Unpack the data. Its structure depends on what is passed to `fit()`.
        data_len = len(data)
        if data_len == 3:
            x, y, mask = data
        else:
            x, y = data
        # x expected shape = (batch_size, original_length)
        # y expected shape = (batch_size,)
        # masks expected shape = (batch_size,)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value using only outputs from the
            # randomly masked input sequence elements. 
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y[mask] if data_len==3 else y,
                y_pred[mask] if data_len==3 else y_pred,
                sample_weight=None,
                regularization_losses=self.losses
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, sequence:tf.Tensor, training:bool=None) -> tf.Tensor:
        # Expected input sequence shape = (batch_size, original_len)

        # Add sequence length axis.
        sequence = tf.expand_dims(sequence, axis=-1)
        # Expected output shape = (batch_size, original_len, 1)

        ### Encoder Layers ###

        features = self.pool(self.conv(sequence))
        # Expected output shape = (batch_size, latent_len, hidden)
        positional_embeddings = self.position(features, training=training)
        # Expected output shape = (batch_size, latent_len, hidden)
        x = self.add_norm1(features, positional_embeddings)
        # Expected output shape = (batch_size, latent_len, hidden)
        x = self.dropout1(x, training=training)
        # Expected output shape = (batch_size, latent_len, hidden)

        ### Transformer Layers ###

        # Assign importance weights to "x" using back-to-back transformers.
        for _, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x, mask=None, training=training)
        # Expected output shape = (batch_size, latent_len, hidden)

        ### Decoder Layers ###

        relative_positional_embeddings = self.relative_position(x)
        # Expected output shape = (batch_size, latent_len, hidden)
        x = self.add_norm2(x, relative_positional_embeddings)
        # Expected output shape = (batch_size, latent_len, hidden)
        x = self.avg_pool(x) #self.max_pool(x)
        # Expected output shape = (batch_size, hidden)
        x = self.dense1(x)
        # Expected output shape = (batch_size, decoder_hidden)
        x = self.dropout2(x, training=training)
        # Expected output size = (batch_size, decoder_hidden)
        # Apply sequence-to-point transformation.
        x = self.dense2(x)
        # Expected output_size = (batch_size, 1)
        return self.output_activation(x)