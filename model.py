from ops import *

class Estimator(object):
    def __init__(self,x, name, reuse=False):
        self.input=x
        self.name=name
        self.reuse= reuse

        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.net = conv2d(self.input, 64, [3, 3], strides=1, dilation=1, scope='conv_est1', activation='ReLU')
            self.net = conv2d(self.net, 3, [3, 3], dilation=1, scope='conv_est_out', activation=None)
            self.net=tf.image.resize_bilinear(self.net, tf.shape(self.input)[1:-1]*4)

            self.output=self.net

class Encoder(object):
    def __init__(self, x, name, feat=4, reuse=False):
        self.input = x
        self.name = name
        self.reuse = reuse
        self.feat = feat

        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.net = conv2d(self.input, 64, [3, 3], strides=1, dilation=1, scope='conv1', activation=None)
            self.net = tf.nn.max_pool(self.net, [1,2,2,1], [1,2,2,1], padding='SAME')
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1,scope='conv2', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1, scope='conv3', activation=None)
            self.net = tf.nn.max_pool(self.net, [1,2,2,1], [1,2,2,1], padding='SAME')
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3], strides=1,  dilation=1, scope='conv4', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1, scope='conv5', activation=None)

            self.mu = conv2d(self.net, self.feat, [3,3], scope='mu')
            self.sigma = conv2d(self.net, self.feat, [3,3], scope='sigma')

class Decoder(object):
    def __init__(self, x, name, reuse=False):
        self.input = x
        self.name = name
        self.reuse = reuse

        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.net= conv2d(self.input, 64, [3,3], strides=1, scope='conv_in')
            self.net= tf.image.resize_nearest_neighbor(self.net,size=tf.shape(self.net)[1:-1]*2)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3], strides=1, dilation=1, scope='conv1', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1,scope='conv2', activation=None)
            self.net=  tf.image.resize_nearest_neighbor(self.net,size=tf.shape(self.net)[1:-1]*2)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1, scope='conv3', activation=None)
            self.net = tf.nn.relu(self.net)

            self.output = conv2d(self.net, 3, [3, 3],  strides=1, dilation=1, scope='conv5', activation=None)

class Denoiser(object):
    def __init__(self, x, condition, name, reuse=False):
        self.input = x
        self.condition= tf.image.resize_bilinear(condition, tf.shape(x)[1:-1])
        self.name = name
        self.reuse = reuse
        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))
        with tf.variable_scope(self.name, reuse=self.reuse):
            input_c=tf.concat([self.input, self.condition], axis=-1)
            self.conv1 = conv2d(input_c, 64, [3, 3], scope='conv1', activation=None)
            self.head = self.conv1
            for idx in range(5):
                self.head = self.RIRblock(self.head, 5, 'RIRBlock' + repr(idx))

            self.conv2 = conv2d(self.head, 64, [3, 3], scope='conv2', activation=None)
            self.residual = tf.add(self.conv1, self.conv2)

            self.conv3= conv2d(self.residual, 3, [3, 3], scope='conv3', activation=None)

            self.output = tf.add(self.conv3, self.input)

        tf.add_to_collection('InNOut', self.input)
        tf.add_to_collection('InNOut', self.output)

    def RIRblock(self, x, num, scope):
        with tf.variable_scope(scope):
            head = x
            for idx in range(num):
                head = self.resblock(head, 'RBlock' + repr(idx))
            out = conv2d(head, 64, [3, 3], scope='conv_out')

        return tf.add(out, x)

    def resblock(self, x, scope):
        with tf.variable_scope(scope):
            net1 = conv2d(x, 64, [3, 3], dilation=1, scope='conv1', activation='ReLU')
            out = conv2d(net1, 64, [3, 3], dilation=1, scope='conv2', activation=None)

        return tf.add(out, x)

class Discriminator(object):
    def __init__(self, input, reuse=False):
        self.input = input
        self.reuse=reuse
        self.build_model()

    def build_model(self):
        print('Build Model Discriminator')
        with tf.variable_scope("DIS", reuse=self.reuse):

            self.conv1_1 = SNconv(self.input, 64, [3, 3], scope='conv1_1', activation='leakyReLU')
            self.conv1_2 = SNconv(self.conv1_1, 64, [3, 3],strides=2, scope='conv1_2', activation='leakyReLU')

            self.conv2_1 = SNconv(self.conv1_2, 128, [3, 3], scope='conv2_1', activation='leakyReLU')
            self.conv2_2 = SNconv(self.conv2_1, 128, [3, 3], strides=2, scope='conv2_2', activation='leakyReLU')

            self.conv3_1 = SNconv(self.conv2_2, 256, [3, 3], scope='conv3_1', activation='leakyReLU')
            self.conv3_2 = SNconv(self.conv3_1, 256, [3, 3],strides=2, scope='conv3_2', activation='leakyReLU')

            self.conv4_1 = SNconv(self.conv3_2, 512, [3, 3], scope='conv4_1', activation='leakyReLU')
            self.conv4_2 = SNconv(self.conv4_1, 512, [3, 3],strides=2, scope='conv4_2', activation='leakyReLU')

            self.conv5_1 = SNconv(self.conv4_2, 512, [3, 3], scope='conv5_1', activation='leakyReLU')
            self.conv5_2 = SNconv(self.conv5_1, 512, [3, 3], strides=2, scope='conv5_2', activation='leakyReLU')

            self.logit = SNconv(self.conv5_2, 1, [3, 3], scope='conv6_2')

            self.FEAT=[self.conv3_2, self.conv4_2, self.conv5_2]

