import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def fc(x, n_neurons, activation=None):
    rows, cols = x.shape
    W = weight_variable([int(cols), n_neurons])  # data type returned by tensor.shape() is `tf.Dimension`
    b = bias_variable([n_neurons])
    if activation is None:
        y = tf.matmul(x, W) + b
    else:
        y = activation(tf.matmul(x, W) + b)
    return y, W, b


def conv2d_nhwc(x, weight_shape):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv = weight_variable(weight_shape)
    b_conv = bias_variable([weight_shape[-1]])
    y_conv = tf.nn.relu(tf.nn.bias_add(conv2d(x, W_conv), b_conv))
    y_pool = max_pool_2x2(y_conv)
    return y_pool, W_conv, b_conv


def flat(x, in_size, out_size):
    W_fc = weight_variable([in_size, out_size])
    b_fc = bias_variable([out_size])
    y_flat = tf.reshape(x, [-1, in_size])
    y_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(y_flat, W_fc), b_fc))
    return y_fc, W_fc, b_fc
