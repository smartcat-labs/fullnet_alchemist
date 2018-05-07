import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim

from fullnet_alchemist.utils.summaries import variable_summaries


class SimpleGenerator(object):

    def __init__(self, conv_layers, upsampling_layers, kernel, activation_fn=tf.nn.relu):
        self.conv_layers = conv_layers
        self.upsampling_layers = upsampling_layers
        self.kernel = kernel
        self.activation_fn = activation_fn
        self.reuse = False

        self.normalizer_fn = slim.batch_norm
        self.normalizer_params = {
            'fused': None,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'decay': 0.95,
            'epsilon': 0.001,
            'center': True,
            'scale': True,

        }

    def generate(self, input_tensor, name, is_training=True):

        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])
        outputs = input_tensor
        with tf.name_scope(name=name),\
             tf.variable_scope("generator", reuse=self.reuse):

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                kernel_size=self.kernel,
                                stride=1,
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=norm_params,
                                padding='SAME'
                                ):
                for conv_i, conv_layer in enumerate(self.conv_layers):
                    with tf.variable_scope("conv_layer_{}".format(conv_i)):
                        outputs = slim.conv2d(outputs, num_outputs=conv_layer, scope="conv2d_{}x{}".format(self.kernel[0], self.kernel[1]))

                        logging.debug("Output of conv layer {} is {}".format(conv_i, conv_layer))

                for upsampling_i, upsampling in enumerate(self.upsampling_layers):
                    with tf.variable_scope("upsampling_{}".format(upsampling_i)):
                        outputs = slim.conv2d_transpose(outputs, num_outputs=upsampling, stride=2, scope="deconv2d_{}x{}".format(self.kernel[0], self.kernel[1]))

                        logging.debug("Output of upsampling layer {} is {}".format(upsampling_i, upsampling))

            with tf.variable_scope("generator_output"):
                outputs = slim.conv2d(outputs,
                                      num_outputs=3,
                                      kernel_size=self.kernel,
                                      stride=1,
                                      padding='SAME',
                                      activation_fn=tf.nn.tanh,
                                      normalizer_fn=None,
                                      normalizer_params=None,
                                      scope="tanh_conv2d_{}x{}".format(self.kernel[0], self.kernel[1]))

                logging.debug("Output of upsampling generator is {}".format(outputs))
                tf.summary.image("upsampled", outputs, 5)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        return outputs


class Discriminator(object):

    def __init__(self, layer_depths, kernel, activation_fn=tf.nn.relu):
        # TODO: Extract properties into configuraion
        self.layer_depths = layer_depths
        self.reuse = False
        self.kernel = kernel

        self.normalizer_fn = slim.batch_norm
        self.normalizer_params = {
            'fused': None,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'decay': 0.95,
            'epsilon': 0.001,
            'center': True,
            'scale': True,

        }
        self.activation_fn = activation_fn

    def discriminate(self, input_tensor, name, is_training=True):
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])
        outputs = input_tensor

        with tf.name_scope('discriminator_' + name),\
            tf.variable_scope('discriminator', reuse=self.reuse),\
            slim.arg_scope([slim.conv2d],
                           kernel_size=self.kernel,
                           stride=2,
                           activation_fn=self.activation_fn,
                           normalizer_fn=self.normalizer_fn,
                           normalizer_params=norm_params,
                           padding='SAME'):

            for conv_layer_i, conv_layer in enumerate(self.layer_depths):

                with tf.variable_scope("conv_{}".format(conv_layer_i)):

                    # Don't apply batch norm on first layer (from paper)
                    if conv_layer_i == 0:
                        outputs = slim.conv2d(inputs=outputs,
                                              num_outputs=conv_layer,
                                              normalizer_fn=None,
                                              normalizer_params=None,
                                              stride=1)
                    else:
                        outputs = slim.conv2d(inputs=outputs, num_outputs=conv_layer)
                    logging.debug("Layer {}: {}".format(conv_layer_i, outputs))

            with tf.variable_scope("logits"):
                outputs = slim.flatten(outputs, scope="flatten_output")
                logits = slim.fully_connected(inputs=outputs,
                                               num_outputs=1,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               normalizer_params=None,
                                               scope='logits')
                logging.debug("Logits: {}".format(logits))

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            return logits


class SuperResolution(object):

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def loss(self, lowres_input, highres_input):
        enhanced = self.enhance(lowres_input, name='enhance', is_training=True)

        logits_lowres = self.discriminator.discriminate(enhanced, 'lowres_discriminator')
        logits_highres = self.discriminator.discriminate(highres_input, 'highres_discriminator')

        highres_discriminator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_highres,
                                                    labels=tf.ones_like(logits_highres, dtype=tf.float32))
        )
        variable_summaries(highres_discriminator_loss, 'discriminator_train/summary', 'loss')

        lowres_discriminator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_lowres,
                                                    labels=tf.zeros_like(logits_lowres, dtype=tf.float32))
        )
        variable_summaries(lowres_discriminator_loss, 'discriminator_generated/summary', 'loss')

        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_lowres,
                                                    labels=tf.ones_like(logits_lowres, dtype=tf.float32))
        )
        variable_summaries(generator_loss, 'generator/summary', 'loss')

        total_discriminator_loss = highres_discriminator_loss + lowres_discriminator_loss

        variable_summaries(total_discriminator_loss, 'discriminator_total/summary', 'loss')

        return total_discriminator_loss, generator_loss

    def train_op(self, discriminator_loss, generator_loss, learning_rate, beta1):
        discrimiator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        discriminator_train_step = slim.learning.create_train_op(discriminator_loss, discrimiator_optimizer, variables_to_train=self.discriminator.variables)
        generator_train_step = slim.learning.create_train_op(generator_loss, generator_optimizer, variables_to_train=self.generator.variables)

        with tf.control_dependencies([generator_train_step, discriminator_train_step]):
            train_op = tf.no_op(name='train_step')

        return train_op

    def enhance(self, input_tensor, name, is_training=True):
        return self.generator.generate(input_tensor, name, is_training)


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s')

    lowres_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
    highres_placeholder = tf.placeholder(tf.float32, [None, 128, 128, 3])

    generator = SimpleGenerator(conv_layers=[64, 128],
                                upsampling_layers=[64],
                                kernel=[5, 5],
                                activation_fn=tf.nn.relu)
    discriminator = Discriminator(layer_depths=[64, 96, 128, 160, 256],
                                  kernel=[5, 5],
                                  activation_fn=tf.nn.relu)

    super_resolution = SuperResolution(generator, discriminator)
    d_loss, g_loss = super_resolution.loss(lowres_placeholder, highres_placeholder)
    train_op = super_resolution.train_op(d_loss, g_loss, 0.0002, 0.6)

    logging.debug("VARIABLES")
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        logging.debug(var)
