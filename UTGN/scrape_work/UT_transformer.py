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

threshold = 0.5
act_max_steps = 5

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
    
    
def _attention(query, key, value, mask, keep_prob, train=True):
    """Calculates scaled dot-product attention.
    
    softmax(Q K^{T} / sqrt(d_{k}))V
    
    Args:
        query: A query tensor of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES].
        key:  The key tensor.
        value: The value tensor.
        mask: Mask of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
        keep_prob: The drop out probability.
        train: train or predict
    
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
    if train:
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


def _multi_head_attention(query, key, value, mask, heads, keep_prob, train=True):
    """Calculates the multihead attention.
    
    Args:
        query: query tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        key: key tensor.
        value: value tensor.
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        heads: number of heads.
        keep_prob: The drop out probability.
        train: train or predict
    
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
            keep_prob=keep_prob,
            train=train)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        seq_len = none_to1(seq_len)
        out = tf.reshape(out, shape=[n_batches, seq_len, d_model])
        return tf.layers.dense(out, units=d_model, name="attention")


def _feed_forward(x, d_model, d_ff, keep_prob, train=True):
    """Feed forward layer along with of relu and dropout.
    
    FFN(x) = max(0,xW1+b1)W2+b2
    
    Args:
        x: Input tensor.
        d_model: dimension of W2.
        d_ff: dimension of W1.
        keep_prob: The drop out probability.
        train: train or predict
        
    Returns:
        Tensor
    """

    with tf.variable_scope("feed_forward"):
        hidden = tf.layers.dense(x, units=d_ff, name="hidden")
        hidden = tf.nn.relu(hidden)
        if train:
            hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.layers.dense(hidden, units=d_model, name="out")


def _encoder_layer(x, mask, layer_num, 
                   heads, keep_prob, d_ff, 
                   train=True):
    """Create a single encoder layer.
    
    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        layer_num: The number label of an encoder layer.
        heads: Number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        train: train or predict
        
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
            keep_prob=keep_prob,
            train=train)

        if train:
            attention_out = tf.nn.dropout(
                attention_out, 
                keep_prob)
        added = x + attention_out
        x = _layer_norm(added)

    # with tf.variable_scope(f"ff_{layer_num}"):
    with tf.variable_scope("ff_" + str(layer_num)):
        ff_out = _feed_forward(x, d_model, d_ff, keep_prob, train=train)
        if train:
            ff_out = tf.nn.dropout(ff_out, keep_prob)
        added = x + ff_out
        return _layer_norm(added)


def _encoder(x, mask, n_layers, heads, keep_prob, d_ff, train=True):
    """Create the encoder architecture
    
    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        n_layers: number of layers of the encoder model.
        heads: number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        train: train or predict
        
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
                d_ff=d_ff,
                train=train)
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


def _prepare_embeddings(x, positional_encodings, 
                        keep_prob, train=True):
    """Add positional encoding and normalize embeddings.
    
    Args:
        x: input embeddings of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        positional_encodings: encoding tensor of shape [1, SEQ_LEN, FEATURES].
        keep_prob: The drop out probability.
        train: train or predict
        
    Returns:
        Tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    with tf.variable_scope("prepare_input"):
        _, seq_len, _ = x.shape
        # TODO: put positional encoding back in
        # x = x + positional_encodings[:, :seq_len, :]

        if train:
            x = tf.nn.dropout(x, keep_prob)
        return _layer_norm(x)



import math
def decoder_layer(encoding: tf.Tensor, x: tf.Tensor, *,
                  enc_mask: tf.Tensor, mask: tf.Tensor,
                  index: int, heads: int, keep_prob: float, d_ff: int):
    d_model = encoding.shape[-1]
    
    with tf.variable_scope(f"{index}_self_attention"):
        attention_out = _multi_head_attention(x, x, x,
                                             mask=mask, heads=heads, keep_prob=keep_prob)
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = _layer_norm(added)
    with tf.variable_scope(f"{index}_encoding_attention"):
        attention_out = _multi_head_attention(x, encoding, encoding,
                                             mask=enc_mask, heads=heads, keep_prob=keep_prob)
        
        added = x + tf.nn.dropout(attention_out, keep_prob=keep_prob)
        x = _layer_norm(added)
    with tf.variable_scope(f"{index}_ff"):
        ff_out = _feed_forward(x, d_model, d_ff, keep_prob)
        
        added = x + tf.nn.dropout(ff_out, keep_prob)
        return _layer_norm(added)

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
        return _layer_norm(x)

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
#     return inputs, inputs

def noam_learning_rate(step: int, warm_up: float, d_model: int):
    return (d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

def output_subsequent_mask(seq_len: int):
    mask = np.zeros((seq_len, seq_len), dtype=float)
    for i in range(seq_len):
        for j in range(i + 1):
            mask[i, j] = 1.

    return mask



def ut_function(state,
                step,
                halting_probability,
                remainders,
                n_updates,
                previous_state,
                encoder_layer,
                config={}):
    """Implements ACT (position-wise halting).
    
    Args:
        state: Tensor of shape [batch_size, length, input_dim]
        step: indicates number of steps taken so far
        halting_probability: halting probability
        remainders: ACT remainders
        n_updates: ACT n_updates
        previous_state: previous state
        encoder_layer: encoder layer function
        config: configuration dict
      
    Returns:
        transformed_state: transformed state
        step: step + 1
        halting_probability: halting probability
        remainders: act remainders
        n_updates: act n_updates
        new_state: new state
        
    TODO: include positional encodings
    """

    with tf.variable_scope("sigmoid_activation_for_pondering"):
        p = tf.layers.dense(state, 1, activation=tf.nn.sigmoid, use_bias=True)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = p * still_running + new_halted * remainders

    transformed_state = state

    # TODO: change 3 to take hyperparameter
    for i in range(3):
        with tf.variable_scope("rec_layer_%d" % i):
            transformed_state = encoder_layer(state, i)

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) + (previous_state *
                                                         (1 - update_weights)))

    step += 1
    return (transformed_state, step, halting_probability, remainders,
            n_updates, new_state)


def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    """While loop stops when this predicate is FALSE.
    
    I.e. all (probability < 1-eps AND counter < N) are false.
    
    Args:
        u0: Not used
        u1: Not used
        halting_probability: halting probability
        u2: Not used
        n_updates: ACT n_updates
        u3: Not used
        
    Returns:
        bool
    """

    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(tf.less(halting_probability, threshold),
                       tf.less(n_updates, act_max_steps)))



def train():
    seq_length = 10
    vocab_size = 10 + 1 + 1
    vocab_str = [f"{i}" for i in range(10)]
    vocab_str += ['X', 'S']

    batch_size = 32  # 12000
    d_model = 128  # 512
    #     d_model = 512
    heads = 2
    keep_prob = 1.0
    n_layers = 2  # 6
    #     n_layers = 6
    d_ff = 256  # 2048
    #     d_ff = 2048
    
    
    
    
    
    step = 0
    halting_probability = tf.zeros([batch_size, seq_length, 1])
    remainders = tf.zeros([batch_size, seq_length, 1])
    n_updates = tf.zeros([batch_size, seq_length, 1])
    previous_state = tf.zeros([batch_size, seq_length, d_model])

    
    
    

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
    
    
    
    
    
    
    
    

#     encoding = _encoder(input_embeddings, mask=inputs_mask, n_layers=n_layers, heads=heads,
#                        keep_prob=keep_prob, d_ff=d_ff)

    def encoder_layer(x, layer_num):
#         return _encoder(x, inputs_mask, n_layers, heads, keep_prob, d_ff)

        return _encoder_layer(x, inputs_mask, layer_num, 
                   heads, keep_prob, d_ff)

#     def transition_function(x):
#         return x
    
    def ut_function2(state, step, halting_probability, remainders, n_updates,
                previous_state):
        

        return ut_function(state, step, halting_probability, remainders, n_updates,
                previous_state, encoder_layer)
    
    
    (_, _, _, remainder, n_updates, encoding) = tf.while_loop(
        should_continue,
        ut_function2,
        (input_embeddings, step, halting_probability, remainders, n_updates, previous_state),
        maximum_iterations=act_max_steps + 1)
    
    
    
    
    
    
    
    
    
    
    
    decoding = decoder(encoding, output_embeddings,
                       enc_mask=inputs_mask, mask=output_mask,
                       n_layers=n_layers, heads=heads, keep_prob=keep_prob, d_ff=d_ff)
    log_results = generator(decoding, vocab_size=vocab_size)
    results = tf.exp(log_results)
    loss = label_smoothing_loss(log_results, expected, vocab_size=vocab_size, smoothing=0.0)
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    print(grads)
    grads, _ = tf.clip_by_global_norm(grads, 5.)
    
    grads_and_vars = list(zip(grads, params))
    train_op = adam.apply_gradients(grads_and_vars, name="apply_gradients")

    warm_up = 400
    batch_in_mask = np.ones((1, 1, seq_length), dtype=float)
    batch_out_mask = output_subsequent_mask(seq_length)
    batch_out_mask = batch_out_mask.reshape(1, seq_length, seq_length)
    def __print_seq(seq):
        return ' '.join([vocab_str[i] for i in seq])

#     return
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(10000):
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