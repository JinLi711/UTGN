"""Encoder portion of the transformer.

Input tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES]
Output tensor of the same shape.

# TODO: Take a look at _higher_recurrence and perform similar
functions (like updating config, etc)
# TODO: add trainable parameters to collection 
(check _recurrence)
"""

import numpy as np
import tensorflow as tf


cast32 = lambda x: tf.dtypes.cast(x, tf.float32)
none_to1 = lambda x: -1 if x == None else x


def _get_mean_std(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    squared = tf.square(x - mean)
    variance = tf.reduce_mean(squared, axis=-1, keepdims=True)
    std = tf.sqrt(variance)
    return mean, std

def _layer_norm(layer):
    """Perform layer normalization.

    Not the same as batch normalization.

    Args:
        layer: Tensor

    Returns:
        Tensor
    """

    with tf.variable_scope("norm"):
        scale = tf.get_variable(
            "scale", 
            shape=layer.shape[-1], 
            dtype=tf.float32)
        base = tf.get_variable(
            "base", 
            shape=layer.shape[-1], 
            dtype=tf.float32)
        mean, std = _get_mean_std(layer)
        norm = (layer - mean) / (std + 1e-6)
        return norm * scale + base
    
    
def _attention(query, key, value, mask, keep_prob):
    """Calculates scaled dot-product attention.
    
    softmax(Q K^{T} / sqrt(d_{k}))V
    
    Args:
        query: A query tensor of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES].
        key:  The key tensor.
        value: The value tensor.
        mask: Mask of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
        keep_prob: The drop out probability.
    
    Returns:
        The scaled dot-product attention.
        Shape: [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
    """

    d_k = query.shape[-1].value
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
    scores = scores / tf.constant(np.sqrt(d_k), dtype=tf.float32)
    mask_add = ((scores * 0) - cast32(1e9)) * (tf.constant(1.) - cast32(mask))
    scores = scores * cast32(mask) + mask_add
    attn = tf.nn.softmax(scores, axis=-1)
    attn = tf.nn.dropout(attn, keep_prob)
    return tf.matmul(attn, value)


def _prepare_multi_head_attention(x, heads, name):
    """Prepares for multihead attention.
    
    Prepares query, key, value that have form [BATCH_SIZE, SEQ_LEN, FEATURES].
    
    Args:
        x: Tensor input.
        heads: Number of heads.
        name: Either query, key, or value.
    
    Returns:
        A prepared Q, K, or V of form [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
        
    Raises:
        AssertionError: Dimension of features must be divisible by the number of heads.
    """

    n_batches, seq_len, d_model = x.get_shape().as_list()
    seq_len = none_to1(seq_len)
    assert d_model % heads == 0, "Dimension of features needs to be divisible by the number of heads."
    d_k = d_model // heads
    x = tf.layers.dense(x, units=d_model, name=name)
    x = tf.reshape(x, shape=(n_batches, seq_len, heads, d_k))
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    return x


def _multi_head_attention(query, key, value, mask, heads, keep_prob):
    """Calculates the multihead attention.
    
    Args:
        query: query tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        key: key tensor.
        value: value tensor.
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        heads: number of heads.
        keep_prob: The drop out probability.
    
    Returns:
        Tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES]
    """

    with tf.variable_scope("multi_head"):
        n_batches, seq_len, d_model = query.get_shape().as_list()
        query = _prepare_multi_head_attention(query, heads, "query")
        key = _prepare_multi_head_attention(key, heads, "key")
        value = _prepare_multi_head_attention(value, heads, "value")
        mask = tf.expand_dims(mask, axis=1)
        out = _attention(
            query, 
            key, 
            value, 
            mask=mask, 
            keep_prob=keep_prob)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        seq_len = none_to1(seq_len)
        out = tf.reshape(out, shape=[n_batches, seq_len, d_model])
        return tf.layers.dense(out, units=d_model, name="attention")


def _feed_forward(x, d_model, d_ff, keep_prob):
    """Feed forward layer along with of relu and dropout.
    
    FFN(x) = max(0,xW1+b1)W2+b2
    
    Args:
        x: Input tensor.
        d_model: dimension of W2.
        d_ff: dimension of W1.
        keep_prob: The drop out probability.
        
    Returns:
        Tensor
    """

    with tf.variable_scope("feed_forward"):
        hidden = tf.layers.dense(x, units=d_ff, name="hidden")
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.layers.dense(hidden, units=d_model, name="out")


def _encoder_layer(x, mask, layer_num, heads, keep_prob, d_ff):
    """Create a single encoder layer.
    
    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        layer_num: The number label of an encoder layer.
        heads: Number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        
    Returns:
        Tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    d_model = x.shape[-1]
    # with tf.variable_scope(f"attention_{layer_num}"):
    with tf.variable_scope("attention_" + str(layer_num)):
        attention_out = _multi_head_attention(
            x,
            x,
            x,
            mask=mask,
            heads=heads,
            keep_prob=keep_prob)
        added = x + tf.nn.dropout(attention_out, keep_prob)
        x = _layer_norm(added)

    # with tf.variable_scope(f"ff_{layer_num}"):
    with tf.variable_scope("ff_" + str(layer_num)):
        ff_out = _feed_forward(x, d_model, d_ff, keep_prob)
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return _layer_norm(added)


def _encoder(x, mask, n_layers, heads, keep_prob, d_ff):
    """Create the encoder architecture
    
    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        n_layers: number of layers of the encoder model.
        heads: number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        
    Returns:
        Tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    with tf.variable_scope("encoder"):
        for i in range(n_layers):
            x = _encoder_layer(
                x,
                mask=mask,
                layer_num=i,
                heads=heads,
                keep_prob=keep_prob,
                d_ff=d_ff)
        return x


def _generate_positional_encodings(d_model, seq_len=5000):
    """Create positional encoding.
    
    Args:
        d_model: dimension of input embeddings
        seq_len: maximum sequence length of batch
        
    Returns:
        Constant tensor of shape [1, seq_len, d_model]
    """

    encodings = np.zeros((seq_len, d_model), dtype=float)
    position = np.arange(0, seq_len).reshape((seq_len, 1))
    two_i = np.arange(0, d_model, 2)
    div_term = np.exp(-np.log(10000.0) * two_i / d_model)
    encodings[:, 0::2] = np.sin(position * div_term)
    encodings[:, 1::2] = np.cos(position * div_term)

    pos_encodings = tf.constant(
        encodings.reshape((1, seq_len, d_model)),
        dtype=tf.float32,
        name="positional_encodings")

    return pos_encodings


def _prepare_embeddings(x, positional_encodings, keep_prob):
    """Add positional encoding and normalize embeddings.
    
    Args:
        x: input embeddings of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        positional_encodings: encoding tensor of shape [1, SEQ_LEN, FEATURES].
        keep_prob: The drop out probability.
        
    Returns:
        Tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    with tf.variable_scope("prepare_input"):
        _, seq_len, _ = x.shape
        # TODO: put positional encoding back in
        # x = x + positional_encodings[:, :seq_len, :]
        x = tf.nn.dropout(x, keep_prob)
        return _layer_norm(x)