import math

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
def get_mean_std(x: tf.Tensor):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    squared = tf.square(x - mean)
    variance = tf.reduce_mean(squared, axis=-1, keepdims=True)
    std = tf.sqrt(variance)

    return mean, std

def layer_norm(layer: tf.Tensor):
    with tf.variable_scope("norm"):
        scale = tf.get_variable("scale", shape=layer.shape[-1], dtype=tf.float32)
        base = tf.get_variable("base", shape=layer.shape[-1], dtype=tf.float32)
        mean, std = get_mean_std(layer)
        norm = (layer - mean) / (std + 1e-6)
        return norm * scale + base

def attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, *,
              mask: tf.Tensor,
              keep_prob: float):
    d_k = query.shape[-1].value
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
    scores = scores / tf.constant(math.sqrt(d_k))
    mask_add = ((scores * 0) - 1e9) * (tf.constant(1.) - mask)
    scores = scores * mask + mask_add
    attn = tf.nn.softmax(scores, axis=-1)
    attn = tf.nn.dropout(attn, keep_prob)
    return tf.matmul(attn, value), attn

def prepare_for_multi_head_attention(x: tf.Tensor, heads: int, name: str):
    n_batches, seq_len, d_model = x.shape
    assert d_model % heads == 0
    d_k = d_model // heads
    x = tf.layers.dense(x, units=d_model, name=name)
    x = tf.reshape(x, shape=[n_batches, seq_len, heads, d_k])
    x = tf.transpose(x, perm=[0, 2, 1, 3])

    return x

def multi_head_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, *,
                         mask: tf.Tensor,
                         heads: int,
                         keep_prob: float):
    with tf.variable_scope("multi_head"):
        n_batches, seq_len, d_model = query.shape
        query = prepare_for_multi_head_attention(query, heads, "query")
        key = prepare_for_multi_head_attention(key, heads, "key")
        value = prepare_for_multi_head_attention(value, heads, "value")
        mask = tf.expand_dims(mask, axis=1)
        out, _ = attention(query, key, value, mask=mask, keep_prob=keep_prob)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, shape=[n_batches, seq_len, d_model])
        return tf.layers.dense(out, units=d_model, name="attention")
    
def feed_forward(x: tf.Tensor,
                 d_model: int, d_ff: int, keep_prob: float):
    
    with tf.variable_scope("feed_forward"):
        hidden = tf.layers.dense(x, units=d_ff, name="hidden")
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.layers.dense(hidden, units=d_model, name="out")
    
def encoder_layer(x: tf.Tensor, *,
                  mask: tf.Tensor, index: int, heads: int,
                  keep_prob: float, d_ff: int):
    d_model = x.shape[-1]

    with tf.variable_scope(f"attention_{index}"):
        attention_out = multi_head_attention(x, x, x,
                                             mask=mask, heads=heads, keep_prob=keep_prob)
        added = x + tf.nn.dropout(attention_out, keep_prob)
        x = layer_norm(added)
    with tf.variable_scope(f"ff_{index}"):
        ff_out = feed_forward(x, d_model, d_ff, keep_prob)
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return layer_norm(added)

def encoder(x: tf.Tensor, *,
            mask: tf.Tensor,
            n_layers: int,
            heads: int, keep_prob: float, d_ff: int):
    with tf.variable_scope("encoder"):
        for i in range(n_layers):
            x = encoder_layer(x,
                              mask=mask, index=i,
                              heads=heads, keep_prob=keep_prob, d_ff=d_ff)

        return x
    
def decoder_layer(encoding: tf.Tensor, x: tf.Tensor, *,
                  enc_mask: tf.Tensor, mask: tf.Tensor,
                  index: int, heads: int, keep_prob: float, d_ff: int):
    d_model = encoding.shape[-1]
    
    with tf.variable_scope(f"{index}_self_attention"):
        attention_out = multi_head_attention(x, x, x,
                                             mask=mask, heads=heads, keep_prob=keep_prob)
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = layer_norm(added)
    with tf.variable_scope(f"{index}_encoding_attention"):
        attention_out = multi_head_attention(x, encoding, encoding,
                                             mask=enc_mask, heads=heads, keep_prob=keep_prob)
        
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = layer_norm(added)
    with tf.variable_scope(f"{index}_ff"):
        ff_out = feed_forward(x, d_model, d_ff, keep_prob)
        
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return layer_norm(added)

def decoder(encoding: tf.Tensor, x: tf.Tensor, *,
            enc_mask: tf.Tensor, mask: tf.Tensor,
            n_layers: int,
            heads: int, keep_prob: float, d_ff: int):
    with tf.variable_scope("decoder"):
        for i in range(n_layers):
            x = decoder_layer(encoding, x,
                              enc_mask=enc_mask, mask=mask, index=i,
                              heads=heads, keep_prob=keep_prob, d_ff=d_ff)

        return x
    
def get_embeddings(input_ids: tf.Tensor, output_ids: tf.Tensor,
                   vocab_size: int, d_model: int):

    word_embeddings = tf.get_variable("word_embeddings",
                                      shape=[vocab_size, d_model],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.random_normal())
    in_emb = tf.nn.embedding_lookup(word_embeddings, input_ids)
    out_emb = tf.nn.embedding_lookup(word_embeddings, output_ids)
    return word_embeddings, in_emb, out_emb

def generate_positional_encodings(d_model: int, max_len: int = 5000):
    encodings = np.zeros((max_len, d_model), dtype=float)
    position = np.arange(0, max_len).reshape((max_len, 1))
    two_i = np.arange(0, d_model, 2)
    div_term = np.exp(-math.log(10000.0) * two_i / d_model)
    encodings[:, 0::2] = np.sin(position * div_term)
    encodings[:, 1::2] = np.cos(position * div_term)
    return tf.constant(encodings.reshape((1, max_len, d_model)),
                       dtype=tf.float32, name="positional_encodings")

def prepare_embeddings(x: tf.Tensor, *,
                       positional_encodings: tf.Tensor,
                       keep_prob: float, is_input: bool):
    name = "prepare_input" if is_input else "prepare_output"
    with tf.variable_scope(name):
        _, seq_len, _ = x.shape
        x = x + positional_encodings[:, :seq_len, :]
        x = tf.nn.dropout(x, keep_prob)
        return layer_norm(x)

def generator(x: tf.Tensor, *, vocab_size: int):
#
    res = tf.layers.dense(x, units=vocab_size, name="generator")
    return tf.nn.log_softmax(res, axis=-1)


def label_smoothing_loss(results: tf.Tensor, expected: tf.Tensor, *,
                         vocab_size: int, smoothing: float):
    results = tf.reshape(results, shape=(-1, vocab_size))
    expected = tf.reshape(expected, shape=[-1])

    confidence = 1 - smoothing
    smoothing = smoothing / (vocab_size - 1)
    expected = tf.one_hot(expected, depth=vocab_size) * (confidence - smoothing)
    expected += smoothing

    results = tf.distributions.Categorical(logits=results)
    expected = tf.distributions.Categorical(logits=expected)
    return tf.reduce_mean(tf.distributions.kl_divergence(results, expected))

def generate_data(batch_size: int, seq_len: int, vocab_size: int):
    start_token = vocab_size - 1
    repeat_token = vocab_size - 2
    vocab_size -= 2

    inputs = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    outputs = np.zeros((batch_size, seq_len + 1), dtype=int)
    outputs[:, 1:] = np.flip(inputs, 1)
    outputs[:, 0] = start_token

    for i in range(batch_size):
        v = np.zeros(vocab_size, dtype=bool)
        for j in range(seq_len):
            word = inputs[i, j]
            if v[word]:
                v[word] = False
                outputs[i][seq_len - j] = repeat_token
            else:
                v[word] = True

    return inputs, outputs


def noam_learning_rate(step: int, warm_up: float, d_model: int):
    return (d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

def output_subsequent_mask(seq_len: int):
    mask = np.zeros((seq_len, seq_len), dtype=float)
    for i in range(seq_len):
        for j in range(i + 1):
            mask[i, j] = 1.

    return mask

def train():
    seq_length = 10
    vocab_size = 10 + 1 + 1
    vocab_str = [f"{i}" for i in range(10)]
    vocab_str += ['X', 'S']
    
    batch_size = 32  # 12000
#     d_model = 128  # 512
    d_model = 512
    heads = 8
    keep_prob = 0.9
#     n_layers = 2  # 6
    n_layers = 6
#     d_ff = 256  # 2048
    d_ff = 2048
    positional_encodings = generate_positional_encodings(d_model)
    inputs = tf.placeholder(dtype=tf.int32,
                            shape=(batch_size, seq_length), name="input")
    outputs = tf.placeholder(dtype=tf.int32,
                             shape=(batch_size, seq_length), name="output")
    expected = tf.placeholder(dtype=tf.int32,
                              shape=(batch_size, seq_length), name="expected")
    inputs_mask = tf.placeholder(dtype=tf.float32,
                                 shape=(1, 1, seq_length),
                                 name="input_mask")
    output_mask = tf.placeholder(dtype=tf.float32,
                                 shape=(1, seq_length, seq_length),
                                 name="output_mask")
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    w_embed, input_embeddings, output_embeddings = get_embeddings(inputs, outputs, vocab_size,
                                                                  d_model)
    input_embeddings = prepare_embeddings(input_embeddings,
                                          positional_encodings=positional_encodings,
                                          keep_prob=keep_prob,
                                          is_input=True)
    output_embeddings = prepare_embeddings(output_embeddings,
                                           positional_encodings=positional_encodings,
                                           keep_prob=keep_prob,
                                           is_input=False)

    encoding = encoder(input_embeddings, mask=inputs_mask, n_layers=n_layers, heads=heads,
                       keep_prob=keep_prob, d_ff=d_ff)
    decoding = decoder(encoding, output_embeddings,
                       enc_mask=inputs_mask, mask=output_mask,
                       n_layers=n_layers, heads=heads, keep_prob=keep_prob, d_ff=d_ff)
    log_results = generator(decoding, vocab_size=vocab_size)
    results = tf.exp(log_results)
    loss = label_smoothing_loss(log_results, expected, vocab_size=vocab_size, smoothing=0.0)
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
    params = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.)
    grads_and_vars = list(zip(grads, params))
    train_op = adam.apply_gradients(grads_and_vars, name="apply_gradients")

    warm_up = 400
    batch_in_mask = np.ones((1, 1, seq_length), dtype=float)
    batch_out_mask = output_subsequent_mask(seq_length)
    batch_out_mask = batch_out_mask.reshape(1, seq_length, seq_length)
    def __print_seq(seq):
        return ' '.join([vocab_str[i] for i in seq])

    # return
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(100_000):
            lr = noam_learning_rate(i + 1, warm_up, d_model)
            
            batch_in, batch_out = generate_data(batch_size, seq_length, vocab_size)
            _, batch_loss, batch_res = session.run([train_op, loss, results],
                                                   feed_dict={
                                                       learning_rate: lr,
                                                       inputs: batch_in,
                                                       outputs: batch_out[:, :-1],
                                                       expected: batch_out[:, 1:],
                                                       inputs_mask: batch_in_mask,
                                                       output_mask: batch_out_mask
                                                   })
            if i % 100 == 0:
                print(f"step={i}\tloss={batch_loss: .6f}")
                print(f"inp=  {__print_seq(batch_in[0])}")
                print(f"exp={__print_seq(batch_out[0])}")
                print(f"res=  {__print_seq(np.argmax(batch_res[0], -1))}")

train()