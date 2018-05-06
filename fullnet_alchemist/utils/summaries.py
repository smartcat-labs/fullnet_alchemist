import tensorflow as tf


def variable_summaries(var, scope, family):
    with tf.variable_scope(scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, family=family)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, family)
        tf.summary.scalar('max', tf.reduce_max(var), family=family)
        tf.summary.scalar('min', tf.reduce_min(var), family=family)
        tf.summary.histogram('histogram', var, family=family)


def scalar_summary(scalar, tag_name, family=None, node_name='summary'):
    with tf.variable_scope(node_name):
        tf.summary.scalar(tag_name, scalar, family=family)


def histogram_summary(hist, tag_name, family=None, node_name='summary'):
    with tf.variable_scope(node_name):
        tf.summary.histogram(tag_name, hist, family=family)
