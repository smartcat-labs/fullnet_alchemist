import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.summaries import variable_summaries
import logging


class VariationalConvolutionalEncoder(object):

    def __init__(self, z_dim, layers, kernel, activation_fn=tf.nn.relu):
        """
        Creates variational autoencoder
        :param z_dim: int dimension of latent vector (code)
        :param layers: list representing number of filters in layers
        :param kernel: list kernel size
        :param activation_fn: activation function
        """
        self.z_dim = z_dim
        self.layers = layers
        self.kernel = kernel

        self.reuse = False

        # TODO: pull this outside (in config)
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

    def encode(self, input_tensor, is_training=True):
        """
        Actually define graph of variational autoencoder
        :param input_tensor: tensor with shape [None, height, width, channels]
        :param is_training: boolean used for batch normalization
        :return: mean, sigma tensors for calculate latent vector - z
        """
        output = input_tensor
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])
        with tf.variable_scope("variational_encoder", reuse=self.reuse), slim.arg_scope([slim.conv2d],
                                                                      kernel_size=self.kernel,
                                                                      stride=2,
                                                                      activation_fn=self.activation_fn,
                                                                      normalizer_fn=self.normalizer_fn,
                                                                      normalizer_params=norm_params,
                                                                      padding='SAME'
                                                                      ):
            for layer_i, layer in enumerate(self.layers):
                with tf.variable_scope("conv_{}".format(layer_i)):
                    output = slim.conv2d(inputs=output, num_outputs=layer)
                    logging.debug("Layer {}: {}".format(layer_i, output))

            flatten = slim.flatten(output, scope="flatten")

            mean = slim.fully_connected(inputs=flatten,
                                      num_outputs=self.z_dim,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      normalizer_params=None,
                                      scope='encoder_mean')
            logging.debug("Mean: {}".format(mean))

            sigma = slim.fully_connected(inputs=flatten,
                                      num_outputs=self.z_dim,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      normalizer_params=None,
                                      scope='encoder_sigma')

            logging.debug("Sigma: {}".format(sigma))
            variable_summaries(sigma, 'variational_encoder_sigma', 'z_sigma')
            variable_summaries(mean, 'variational_encoder_mean', 'z_mean')
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'variational_encoder')

        return mean, sigma


class Generator(object):

    def __init__(self, start_size, channel_depts, activation_fn=tf.nn.relu):
        """
        Creates generator network
        :param start_size: int that represent width and height of tensor after projection layer
        :param channel_depts: list upsampling layers, each element represent number of filters in layer
        :param activation_fn: function
        """
        # TODO: Extract properties into configuraion
        self.start_size = start_size
        self.reuse = False

        self.channel_depths =  channel_depts
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

    def generate(self, inputs, is_training=False, name=''):
        """
        Defines graph for generate network
        :param inputs: tensor with shape [None, z_dim]
        :param is_training: boolean flag for batch normalization
        :param name: name of graph (apply same weights for different inputs)
        :return: generated image
        """
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])
        outputs = inputs
        with tf.name_scope(name=name),\
             tf.variable_scope("generator", reuse=self.reuse),\
             slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[5, 5],
                            stride=2,
                            activation_fn=self.activation_fn,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=norm_params,
                            padding='SAME'):

            with tf.variable_scope("projection"):
                outputs = slim.fully_connected(inputs=outputs,
                                               num_outputs=self.start_size*self.start_size*self.channel_depths[0],
                                               activation_fn=self.activation_fn,
                                               normalizer_fn=self.normalizer_fn,
                                               normalizer_params=norm_params)
                outputs = tf.reshape(outputs, [-1, self.start_size, self.start_size, self.channel_depths[0]], name="projection_reshape")
                logging.debug("Projection: {}".format(outputs))

            for deconv_layer_i, deconv_layer in enumerate(self.channel_depths[:-1]):

                with tf.variable_scope("deconv_{}".format(deconv_layer_i)):
                    outputs = slim.conv2d_transpose(inputs=outputs, num_outputs=deconv_layer, padding='SAME')
                    logging.debug("Deconv layer {}: {}".format(deconv_layer_i, outputs))

            with tf.variable_scope("output_deconv"):
                outputs = slim.conv2d(inputs=outputs,
                                                num_outputs=self.channel_depths[-1],
                                                activation_fn=tf.nn.tanh,
                                                normalizer_fn=None,
                                                normalizer_params=None,
                                                stride=1,
                                                kernel_size=[5, 5],
                                                padding='SAME'
                                                )
                logging.debug("Generator output: {}".format(outputs))

            tf.summary.image('generated_images', tf.div(tf.add(outputs, 1.0), 2.0), max_outputs=5)
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            return outputs


class Discriminator(object):

    def __init__(self, layer_depths, activation_fn=tf.nn.relu):
        """
        Creates discriminator network
        :param layer_depths: list number of filters in each layer
        :param activation_fn:
        """
        # TODO: Extract properties into configuraion
        self.layer_depths = layer_depths
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
        self.activation_fn = activation_fn

    def discriminate(self, inputs, name, is_training=False):
        """
        Defines graph for discriminator network. If it already exists it reuses it
        :param inputs: tensor images
        :param name: str name for network, so it can apply same weights
        :param is_training: boolean used for batch normalization
        :return: logits - is it image real or fake; features - Lth layer features
        """
        norm_params = dict(self.normalizer_params.items() + [('is_training', is_training)])
        outputs = inputs

        with tf.name_scope('discriminator_' + name),\
            tf.variable_scope('discriminator', reuse=self.reuse),\
            slim.arg_scope([slim.conv2d],
                           kernel_size=[5, 5],
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
                logging.debug("Features: {}".format(outputs))
                logging.debug("Logits: {}".format(logits))

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            return logits, outputs


class VAE_DCGAN(object):

    def __init__(self, start_size, latent_size, batch_size,
                 generator_layers, discriminator_layers, variational_layers,
                 variational_kernel):
        """
        Combines all three network into single graph
        :param start_size: int that represent width and height of tensor after projection layer
        :param latent_size: int dimension of latent vector (code)
        :param batch_size: int
        :param generator_layers: list upsampling layers, each element represent number of filters in layer
        :param discriminator_layers: list number of filters in each layer in discriminator
        :param variational_layers: list representing number of filters in layers in variational encoder
        :param variational_kernel: list of kernel size in variational encoder
        """
        self.start_size = start_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.variational_layers = variational_layers
        self.variational_kernel = variational_kernel

        self.variational_encoder = VariationalConvolutionalEncoder(self.latent_size, self.variational_layers, self.variational_kernel)
        self.generator = Generator(self.start_size, self.generator_layers)
        self.discriminator = Discriminator(self.discriminator_layers)

        self.random_vae_epsilon = tf.random_normal([self.batch_size, self.latent_size], dtype=tf.float32)
        self.random_gan = tf.random_normal(shape=[self.batch_size, self.latent_size])

    def losses(self, input_tensor, gama=1e-2):
        """
        Do inference and create loss function for each network
        :param input_tensor: tensor of images
        :param gama: double weight for perceptual loss
        :return: discriminator loss, generator loss, encoder loss
        """
        mean, sigma = self.variational_encoder.encode(input_tensor, is_training=True)
        stddev = tf.sqrt(tf.exp(sigma))
        z = mean + stddev*self.random_vae_epsilon
        variable_summaries(z, 'vae', 'z')

        x_generated_z = self.generator.generate(z, is_training=True, name='generated_z')
        logits_z, features_generated_z = self.discriminator.discriminate(x_generated_z, is_training=True, name='discriminator_z_fake')

        x_generated_rand = self.generator.generate(self.random_gan, is_training=True, name='generated_on_random')
        logits_real, features_real = self.discriminator.discriminate(input_tensor, is_training=True, name='discrimiantor_positive')
        logits_fake, _ = self.discriminator.discriminate(x_generated_rand, is_training=True, name='discriminator_random_negative')

        sse_loss = tf.reduce_mean(tf.square(input_tensor - x_generated_z))
        variable_summaries(sse_loss, 'sse_loss', 'losses')

        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + sigma - tf.pow(mean, 2) - tf.exp(sigma), 1))
        variable_summaries(kl_loss, 'kl_loss', 'losses')

        LGAN = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real) + \
               tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake) + \
               tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_z), logits=logits_z)

        LL_loss = tf.reduce_sum(tf.square(features_real - features_generated_z), 1, name='ll_loss')
        variable_summaries(LL_loss, "ll_loss", "losses")

        E_loss = tf.reduce_mean(kl_loss + LL_loss, name='encoder_loss')
        variable_summaries(E_loss, "encoder_loss", "losses")

        # TODO:  + or -
        G_loss = tf.reduce_mean(gama*LL_loss - LGAN, name='generator_loss')
        variable_summaries(G_loss, "generator_loss", "losses")

        D_loss = tf.reduce_mean(LGAN, name='discrimiantor_loss')
        variable_summaries(D_loss, "discriminator_loss", "losses")

        return D_loss, G_loss, E_loss

    def training_step(self, D_loss, G_loss, E_loss, learning_rate=0.0002, beta1=0.5):
        """
        Create training step operation for each of 3 network
        :param D_loss: discriminator loss operation
        :param G_loss: generator loss operation
        :param E_loss: encoder loss operation
        :param learning_rate: float learning rate
        :param beta1: float
        :return: single train step operation
        """
        discrimiator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        discriminator_train_step = slim.learning.create_train_op(D_loss, discrimiator_optimizer, variables_to_train=self.discriminator.variables)
        generator_train_step = slim.learning.create_train_op(G_loss, generator_optimizer, variables_to_train=self.generator.variables)
        encoder_train_step = slim.learning.create_train_op(E_loss, encoder_optimizer, variables_to_train=self.variational_encoder.variables)

        with tf.control_dependencies([encoder_train_step, generator_train_step, discriminator_train_step]):
            train_op = tf.no_op(name='train_step')

        return train_op

if __name__ == '__main__':
    # Example
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s')
    vae_gan = VAE_DCGAN(start_size=8,
                        latent_size=512,
                        batch_size=64,
                        generator_layers=[256, 128, 64, 3],
                        discriminator_layers=[64, 128, 256, 256],
                        variational_layers=[64, 128, 256],
                        variational_kernel=[5, 5])

    input = tf.placeholder(tf.float32, [None, 64, 64, 3])
    d_loss, g_loss, e_loss = vae_gan.losses(input_tensor=input)

    train_op = vae_gan.training_step(d_loss, g_loss, e_loss)

    print train_op

    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print variable
