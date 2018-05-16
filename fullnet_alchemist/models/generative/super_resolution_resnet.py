import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

from fullnet_alchemist.utils.summaries import variable_summaries


def resnet_block(input_tensor, num_outputs, name,  layers_num=2):
    output_tensor = input_tensor
    with tf.variable_scope(name):
        for conv_i in range(layers_num):
            output_tensor = slim.conv2d(output_tensor, num_outputs=num_outputs, scope="{}_{}_conv2d".format(name, conv_i))

    return output_tensor


def parametric_relu_fn():
    def prelu(input_tensor):
        with tf.variable_scope("prelu"):
            alpha = tf.get_variable("alpha",
                                    shape=input_tensor.get_shape()[-1],
                                    dtype=input_tensor.dtype,
                                    initializer=tf.constant_initializer(0.1))
            return tf.maximum(0.0, input_tensor) + alpha * tf.minimum(0.0, input_tensor)

    return prelu


class Generator(object):

    def __init__(self, activation_fn, resblocks_num=3, resnet_output_num=64, upsampling_output_num=256,
                 external_kernel_size=9, internal_kernel_size=3, upsampling_layers=2):
        self.reuse = False
        self.activation_fn = activation_fn
        self.resnet_output_num = resnet_output_num
        self.external_kernel_size = external_kernel_size
        self.internal_kernel_size = internal_kernel_size
        self.resblocks_num = resblocks_num
        self.upsampling_layers_num = upsampling_layers
        self.upsampling_output_num = upsampling_output_num

        self.normalizer_fn = slim.batch_norm
        self.normalizer_params = {
            'fused': None,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'decay': 0.95,
            'epsilon': 0.001,
            'center': True,
            'scale': True,

        }

    def _resblocks(self, input_tensor, is_training=True):
        block_input = input_tensor
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])

        with tf.variable_scope("res_blocks"),\
                     slim.arg_scope([slim.conv2d],
                                    kernel_size=self.internal_kernel_size,
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=norm_params,
                                    stride=1,
                                    padding='SAME'):
            for res_block_i in range(self.resblocks_num):
                with tf.variable_scope("resblock_{}".format(res_block_i)):
                    block_output = block_input + resnet_block(block_input, self.resnet_output_num, str(res_block_i), layers_num=2)
                    block_input = block_output

            with tf.variable_scope("last_conv"):
                block_output = slim.conv2d(block_input,
                                           num_outputs=self.resnet_output_num,
                                           activation_fn=None,
                                           scope="conv2d_{}x{}".format(self.internal_kernel_size, self.internal_kernel_size))

        return block_output

    def _upsampling_layers(self, input_tensor):
        outputs = input_tensor
        with tf.variable_scope("upsampling_blocks"), slim.arg_scope([slim.conv2d],
                                                                    kernel_size=3,
                                                                    num_outputs=self.upsampling_output_num,
                                                                    activation_fn=None,
                                                                    normalizer_fn=None,
                                                                    stride=1
                                                                    ):
            for upsampling_i in range(self.upsampling_layers_num):
                with tf.variable_scope("upsampling_{}".format(upsampling_i)):
                    outputs = slim.conv2d(outputs,
                                          scope="conv2d_{}x{}".format(self.internal_kernel_size, self.internal_kernel_size))
                    outputs = tf.depth_to_space(outputs, 2, name="subpixel_{}".format(upsampling_i))
                    outputs = self.activation_fn(outputs)

        return outputs

    def generate(self, input_tensor, name, is_training=True):

        output_tensor = input_tensor

        with tf.name_scope(name), tf.variable_scope("generator", reuse=self.reuse):

            with tf.variable_scope("first_conv_layer"):
                first_conv_layer = slim.conv2d(output_tensor,
                                                stride=1,
                                                padding="SAME",
                                                num_outputs=self.resnet_output_num,
                                                kernel_size=self.external_kernel_size,
                                                activation_fn=self.activation_fn,
                                                normalizer_fn=None,
                                                scope="conv2d_{}x{}".format(self.external_kernel_size, self.external_kernel_size)
                                                )
                logging.debug("Output of first convolutional layer: {}".format(first_conv_layer))

            after_resblocks = self._resblocks(first_conv_layer, is_training=is_training)
            logging.debug("Output of resblocks: {}".format(after_resblocks))
            output_tensor = after_resblocks + first_conv_layer
            logging.debug("Output of sum: {}".format(output_tensor))

            output_tensor = self._upsampling_layers(output_tensor)
            logging.debug("Output after upsampling: {}".format(output_tensor))

            with tf.variable_scope("projection_to_image"):
                outputs = slim.conv2d(output_tensor,
                                      num_outputs=3,
                                      kernel_size=self.external_kernel_size,
                                      stride=1,
                                      padding='SAME',
                                      activation_fn=tf.nn.tanh,
                                      normalizer_fn=None)

            tf.summary.image("enhanced", outputs, max_outputs=5)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        return outputs


class Discriminator(object):

    def __init__(self, layer_depths, kernel, activation_fn=tf.nn.leaky_relu):
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
                           activation_fn=self.activation_fn,
                           normalizer_fn=None,
                           padding='SAME'):

            with tf.variable_scope("first_conv"):
                outputs = slim.conv2d(outputs,
                                      num_outputs=self.layer_depths[0],
                                      stride=1)
                outputs = slim.conv2d(outputs,
                                      num_outputs=self.layer_depths[0],
                                      stride=2,
                                      padding='SAME',
                                      normalizer_fn=self.normalizer_fn,
                                      normalizer_params=norm_params)

            for conv_layer_i, conv_layer in enumerate(self.layer_depths[1:]):

                with tf.variable_scope("conv_{}".format(conv_layer_i)):
                    outputs = slim.conv2d(inputs=outputs, num_outputs=conv_layer, normalizer_fn=self.normalizer_fn, normalizer_params=norm_params, stride=1)
                    outputs = slim.conv2d(inputs=outputs, num_outputs=conv_layer, normalizer_fn=self.normalizer_fn, normalizer_params=norm_params, stride=2)
                    logging.debug("Layer {}: {}".format(conv_layer_i, outputs))

            features = slim.flatten(outputs)

            with tf.variable_scope("full_connected"):
                dense = slim.fully_connected(inputs=features,
                                             num_outputs=1024,
                                             normalizer_fn=None,
                                             activation_fn=self.activation_fn)

                logits = slim.fully_connected(inputs=dense,
                                               num_outputs=1,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               normalizer_params=None,
                                               scope='logits')
                logging.debug("Logits: {}".format(logits))

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            return logits, features


class SuperResolution(object):

    def __init__(self, generator, discriminator, feature_extractor=None):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor

    def loss(self, lowres_input, highres_input, gama):
        enhanced = self.enhance(lowres_input, name='enhance', is_training=True)

        logits_lowres, lowres_features = self.discriminator.discriminate(enhanced, 'lowres_discriminator')
        logits_highres, highres_features = self.discriminator.discriminate(highres_input, 'highres_discriminator')

        if self.feature_extractor:
            lowres_features = self.feature_extractor.extract(enhanced, "lowres", 4)
            highres_features = self.feature_extractor.extract(highres_input, "highres", 4)

        highres_discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_highres,
                                                    labels=tf.ones_like(logits_highres, dtype=tf.float32))

        variable_summaries(highres_discriminator_loss, 'discriminator_train/summary', 'loss')

        lowres_discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_lowres,
                                                    labels=tf.zeros_like(logits_lowres, dtype=tf.float32))

        variable_summaries(lowres_discriminator_loss, 'discriminator_generated/summary', 'loss')

        content_loss = tf.reduce_mean(tf.square(highres_features - lowres_features), 1, name="content_loss")
        variable_summaries(content_loss, "content_loss", "loss")

        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_lowres,
                                                    labels=tf.ones_like(logits_lowres, dtype=tf.float32))

        variable_summaries(generator_loss, 'generator/summary', 'loss')

        reconstruction_loss = tf.reduce_mean(tf.square(highres_input - enhanced))
        variable_summaries(reconstruction_loss, 'reconstruction/summary', 'loss')

        total_generator_loss = tf.reduce_mean(0.006 * content_loss + gama*generator_loss, name="total_generator_loss")
        variable_summaries(total_generator_loss, 'generator_total/summary', 'loss')

        total_discriminator_loss = tf.reduce_mean(highres_discriminator_loss + lowres_discriminator_loss, name="total_discriminator_loss")

        variable_summaries(total_discriminator_loss, 'discriminator_total/summary', 'loss')

        return total_discriminator_loss, total_generator_loss

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


class FeatureExtractorVGG(object):

    def __init__(self):
        self.reuse = False
        self.layers = list()
        self.mean_const = tf.constant([123.68, 116.779, 103.939], name="vgg_mean")

    def extract(self, input_tensor, name, layer_i):
        input_tensor = tf.div(tf.add(input_tensor, 1.0), 2.0) * 255.0
        input_tensor = input_tensor - self.mean_const
        self.layers = list()
        with tf.name_scope(name), tf.device("CPU:0"), tf.variable_scope("vgg_19", reuse=self.reuse):
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
                    net = slim.repeat(input_tensor, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    self.layers.append(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    self.layers.append(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                    self.layers.append(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                    self.layers.append(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
                    self.layers.append(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vgg_19")

        return slim.flatten(self.layers[layer_i])


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s')

    lowres_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    highres_placeholder = tf.placeholder(tf.float32, [None, 128, 128, 3])

    generator = Generator(parametric_relu_fn())
    discriminator = Discriminator([64, 128, 256, 512], 3, tf.nn.leaky_relu)

    vgg = FeatureExtractorVGG()
    superresolution = SuperResolution(generator, discriminator, vgg)

    discriminator_loss, generator_loss = superresolution.loss(lowres_placeholder, highres_placeholder, 1e-3)

    logging.debug(discriminator_loss)
    logging.debug(generator_loss)

    train_op = superresolution.train_op(discriminator_loss, generator_loss, 0.0002, 0.7)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(vgg.variables)

    saver.restore(sess, "/media/bigdisk/facelyzr_models/vgg_19.ckpt")
    sess.close()
