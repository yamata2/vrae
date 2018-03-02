import tensorflow as tf
lstm = tf.contrib.rnn.LSTMCell
fc = tf.contrib.layers.fully_connected

from IPython import embed

def make_cells(n_layers, n_units):
    cells = []
    for j in range(n_layers):
        with tf.variable_scope("layer_{}".format(j), reuse=False):
            cell = lstm(num_units=n_units,
                        use_peepholes=True,
                        forget_bias=0.8)
            cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)

def inference(x, n_units=128, n_layers=1, n_out=10, scope="Inference"):
    """Infer a posterior approximately. 

    Returns:
        mean: [batchsize, n_out]
               means of Gaussian distribution.
        log_var: [batchsize, n_out]
               std_dev of Gaussian distribution.
    """
    batch_len = x.get_shape().as_list()[0]
    batchsize = x.get_shape().as_list()[1]
    x = x[::-1]
    with tf.variable_scope(scope, reuse=False):
        cell = make_cells(n_layers, n_units)
        state = cell.zero_state(batchsize, tf.float32)
        for i in range(batch_len):
            h, state = cell(x[i], state)
        state = tf.stack(state)
        state = tf.transpose(state, [2,0,1,3])
        state = tf.reshape(state, [batchsize, -1])
        mean = fc(state, n_out,
                  activation_fn=None,
                  reuse=False,
                  scope="mean")
        log_var= fc(state, n_out,
                    activation_fn=None,
                    reuse=False,
                    scope="log_var")
    return mean, log_var

def generation(x, z_mean, z_log_var,
               n_units, n_layers, n_out, scope="Generation"):
    """Generate frames.

    Return:
        outputs: [data_len, batchsize, n_out]
        n_out is the data dimension.
    """
    batch_len = x.get_shape().as_list()[0]
    batchsize = x.get_shape().as_list()[1]
    eps = tf.random_normal(z_mean.get_shape().as_list(),
                           0, 1, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), eps))
    outputs = []

    state = fc(z, n_units*n_layers*2,
               activation_fn=None,
               scope="decoder_init")
    state_shape = (batchsize, n_layers, 2, n_units)
    state = tf.reshape(state, state_shape)
    state = tf.transpose(state, (1,2,0,3))
    state = tuple([(layer[0], tf.nn.tanh(layer[1])) for layer in tf.unstack(state)]) 

    with tf.variable_scope(scope, reuse=False):
        cell = make_cells(n_layers, n_units)
        for i in range(x.shape[0]):
            if i == 0:
                fc_reuse = False
                current_x = x[0]
            else:
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                #current_x = x[i-1] #teacher forcing
                current_x = y # student forcing
            h, state = cell(current_x, state)
            y = fc(h, n_out,
                   activation_fn=tf.nn.softmax,
                   reuse=fc_reuse,
                   scope="out")
            outputs.append(y)
    return tf.stack(outputs)
