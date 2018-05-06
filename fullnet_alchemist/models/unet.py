import logging
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

from fullnet_alchemist.utils import summaries

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s')

class UNet(object):

    def __init__(self, input_tensor,
                 downsample_number,
                 start_filter_num,
                 activation_fn=tf.nn.relu,
                 normalization=None,
                 normalization_params=None,
                 dropout_keep_prob=1.0,
                 weights_decay=0.00004,
                 is_training=False, *args, **kwargs):
        self.input_tensor = input_tensor
        self.layers_number = downsample_number
        self.start_filter_num = start_filter_num
        self.activation_fn = activation_fn
        self.normalization = normalization
        self.normalization_params = normalization_params
        self.logits = None
        self.layers = []
        self.dropout_keep_prob = dropout_keep_prob
        self.weights_decay = weights_decay
        self.is_training = is_training

        with tf.variable_scope("contractive"):
            self.contracted_output = slim.dropout(self._contractive_part(), keep_prob=self.dropout_keep_prob)
            logging.debug("Tensor affter contractive part is: {}".format(self.contracted_output))

        with tf.variable_scope("dilation"):
            # TODO
            pass

        with tf.variable_scope("expansive"):
            self.expansive_output = self._expansive_part()
            logging.debug("Tensor affter expansive part is: {}".format(self.expansive_output))

        with tf.variable_scope("logits"):
            self.logits = slim.conv2d(inputs=self.expansive_output,
                                      num_outputs=1,
                                      kernel_size=[1, 1],
                                      activation_fn=None,
                                      weights_regularizer=slim.l2_regularizer(self.weights_decay))
            logging.debug("Tensor of logits: {}".format(self.logits))
            summaries.variable_summaries(self.logits, 'summary', 'logits')

    def _contractive_block(self, layer_input, filters_number, padding='SAME'):
        with slim.arg_scope([slim.conv2d],
                    num_outputs=filters_number,
                    kernel_size=[3, 3],
                    padding=padding,
                    activation_fn=self.activation_fn,
                    normalizer_fn=self.normalization,
                    normalizer_params=self.normalization_params,
                    weights_regularizer=slim.l2_regularizer(self.weights_decay)):
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                conv_output = slim.conv2d(inputs=layer_input)
                summaries.variable_summaries(conv_output, 'summary/conv', 'conv_output')

                conv_output = slim.conv2d(inputs=conv_output)
                summaries.variable_summaries(conv_output, 'summary/conv', 'conv_output')

        return conv_output

    def _expansive_block(self, layer_input, layer_from_downsample, filters_number, padding='SAME'):
        with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                    num_outputs=filters_number,
                    padding=padding,
                    activation_fn=self.activation_fn,
                    normalizer_fn=self.normalization,
                    normalizer_params=self.normalization_params,
                    weights_regularizer=slim.l2_regularizer(self.weights_decay)):
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                deconv_output = slim.conv2d_transpose(inputs=layer_input, kernel_size=[2, 2], stride=2)

                summaries.variable_summaries(deconv_output, 'summary/deconv', 'conv_output')
                concatenate = tf.concat(axis=3, values=[layer_from_downsample, deconv_output])
                conv_output = slim.conv2d(inputs=concatenate, kernel_size=[3, 3])
                summaries.variable_summaries(deconv_output, 'summary/conv', 'conv_output')
                conv_output = slim.conv2d(inputs=conv_output, kernel_size=[3, 3])
                summaries.variable_summaries(deconv_output, 'summary/conv', 'conv_output')

        return conv_output

    def _maxpool2d(self, layer_input):
        return slim.max_pool2d(inputs=layer_input,
                                 kernel_size=[2, 2],
                                 stride=2,
                                 padding='SAME')

    def _contractive_part(self):
        filter_nums = self.start_filter_num
        current_output = self.input_tensor
        for layer_i in range(self.layers_number - 1):
            with tf.variable_scope("downsample_layer_{}".format(layer_i)):
                conv_output = self._contractive_block(current_output, filter_nums)
                pooled = self._maxpool2d(conv_output)
                filter_nums *= 2
                current_output = pooled
                self.layers.append(conv_output)
                logging.debug("Output of contractive layer {} is {}".format(layer_i, current_output))

        with tf.variable_scope("downsample_final"):
            conv_output = self._contractive_block(current_output, filter_nums)
            logging.debug("Output of final contractive layer is {}".format(conv_output))
        self.layers.append(conv_output)

        return conv_output

    def _expansive_part(self):
        filter_nums = self.start_filter_num * (self.layers_number ** 2)
        current_outuput = self.contracted_output
        for layer_i in range(self.layers_number-2, -1, -1):
            with tf.variable_scope("upsample_layer_{}".format(layer_i)):
                output = self._expansive_block(current_outuput, self.layers[layer_i], filter_nums)
                filter_nums = int(filter_nums/2)
                current_outuput = output
                logging.debug("Output of expansive layer {} is {}".format(layer_i, current_outuput))

        return current_outuput


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 256, 256, 3])
    is_training = True
    unet = UNet(input_tensor=input_tensor,
                downsample_number=6,
                start_filter_num=16,
                activation_fn=tf.nn.relu,
                normalization=slim.batch_norm,
                normalization_params={
                    'fused': None,
                    'is_training': is_training,
                    'updates_collections': tf.GraphKeys.UPDATE_OPS,
                    'decay': 0.96,
                    'epsilon': 0.001
                },
                dropout_keep_prob=0.75,
                weights_decay=0.0004,
                is_training=is_training)

    logging.info("Logits: {}".format(unet.logits))

    logging.debug("Summaries")
    for summary in tf.get_collection(tf.GraphKeys.SUMMARIES):
        logging.debug(summary)

    logging.debug("Trainable variables")
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        logging.debug(var)