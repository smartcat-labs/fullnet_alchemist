import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
from fullnet_alchemist.utils.summaries import variable_summaries


class SuperResolutionCNN(object):

    def __init__(self, layers, kernels, use_batch_norm, activation_fn=tf.nn.relu):
        self.layers = layers
        self.use_batch_norm = use_batch_norm
        self.kernels = kernels
        self.activation_fn = activation_fn

        self._set_batch_norm()
        self.normalizer_params = {
            'fused': None,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'decay': 0.95,
            'epsilon': 0.001,
            'center': True,
            'scale': True,

        }
        self.reuse = False

    def _set_batch_norm(self):
        if self.use_batch_norm:
            self.normalizer_fn = None
        else:
            self.normalizer_fn = slim.batch_norm

    def loss(self, lowres, highres):
        lowres_enhanced = self.enhance(lowres, True, 'lowres_enhance')

        loss = tf.reduce_mean(tf.square(lowres_enhanced - highres))
        variable_summaries(loss, 'mse', 'loss')

        return loss, lowres_enhanced

    def enhance(self, input_tensor, is_training, name):
        output = input_tensor
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])

        with tf.name_scope(name), \
            tf.variable_scope("enhance", reuse=self.reuse),\
            slim.arg_scope([slim.conv2d],
                           stride=1,
                           activation_fn=self.activation_fn,
                           normalizer_fn=self.normalizer_fn,
                           normalizer_params=norm_params,
                           padding='SAME'
                           ):
            for layer_i, layer in enumerate(self.layers):
                kernel = self.kernels[layer_i]
                with tf.variable_scope("layer_{}".format(layer_i)):
                    output = slim.conv2d(output, num_outputs=layer, kernel_size=kernel, scope="conv2d_{}x{}".format(kernel, kernel))

                    logging.debug("Output of {} layer is {}".format(layer_i, layer))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'enhance')
        return output

    def train_op(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=self.variables)

        return train_op