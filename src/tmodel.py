import tensorflow as tf
import numpy as np
from IPython import embed
import util

class Model():
    def __init__(self):
        self.weights = {}
        self.bnwts = {}

    def new_conv_layer( self, bottom, filter_shape, stride=1, rate=1, activation=tf.identity, padding='SAME', name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.02))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            if w.name in self.weights.keys():
                w = self.weights[w.name]
            else:
                self.weights[w.name] = w

            if b.name in self.weights.keys():
                b = self.weights[b.name]

            else:
                self.weights[b.name] = b
            
            if padding == 'VALID':
                p = int(rate * (filter_shape[0] - 1)/2)
                bottom = tf.pad(bottom, [[0,0], [p, p], [p, p], [0,0]], mode="CONSTANT")
                padding = 'VALID'

            if padding == 'SYMMETRIC' or padding == 'REFELECT':
                p = int(rate * (filter_shape[0] - 1)/2)
                bottom = tf.pad(bottom, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
                padding = 'VALID'


            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], dilations=[1, rate, rate, 1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer( self, bottom, output_shape, filter_shape, stride=1, rate=1, activation=tf.identity, padding='SAME', name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))

            if w.name in self.weights.keys():
                w = self.weights[w.name]
            else:
                self.weights[w.name] = w

            if b.name in self.weights.keys():
                b = self.weights[b.name]

            else:
                self.weights[b.name] = b

            conv = tf.nn.conv2d_transpose( bottom, w, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    # def new_deconv_layer(self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    #     deconv_stride = stride
    #     with tf.variable_scope(name):
    #         x = util.resize(bottom, func=tf.image.resize_nearest_neighbor)
    #         x = self.new_conv_layer(x, filter_shape, stride=deconv_stride, name=name+'_deconv', padding=padding)
    #     return x

    def new_fc_layer( self, bottom, output_size=None, name=None ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        if output_size is None:
            output_size = input_size

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            
            if w.name in self.weights.keys():
                w = self.weights[w.name]
            else:
                self.weights[w.name] = w

            if b.name in self.weights.keys():
                b = self.weights[b.name]

            else:
                self.weights[b.name] = b

            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.2):
        return tf.maximum(leak*bottom, bottom)

    def elu(self, bottom):
        return tf.nn.elu(bottom)

    def relu(self, bottom):
        return tf.nn.relu(bottom)

    def batchnorm(self, bottom, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)

        with tf.variable_scope(name):
            mean_nm, var_nm = name+'_mu', name+'_v'
            # batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            batch_mean, batch_var = self.bnwts[mean_nm], self.bnwts[var_nm]
            normed = tf.nn.batch_normalization(bottom, batch_mean, batch_var, None, None, epsilon)
        return normed

    def build_contour_representation(self, contour, reuse=None):
        input_cnum = contour.get_shape().as_list()[3]
        with tf.variable_scope('REP', reuse=reuse):
            dil_conv1 = self.leaky_relu(self.batchnorm(self.new_conv_layer(contour, [3,3,input_cnum,32], stride=1, rate=4, name="dil_conv1", padding='SAME'), name="bn1"))
            conv1 = self.relu(self.batchnorm(self.new_conv_layer(dil_conv1, [3, 3, 32, 3], stride=1, name='conv1'), name="bn2"))
            dil_conv2 = self.leaky_relu(self.batchnorm(self.new_conv_layer(contour, [3,3,input_cnum,256], stride=1, rate=8, name="dil_conv2", padding='SAME'), name="bn3"))
            conv2 = self.relu(self.batchnorm(self.new_conv_layer(dil_conv2, [3, 3, 256, 3], stride=1, name='conv2'), name="bn4"))
            dil_conv3 = self.leaky_relu(self.batchnorm(self.new_conv_layer(contour, [3,3,input_cnum,256], stride=1, rate=12, name="dil_conv3", padding='SAME'), name="bn5"))
            conv3 = self.relu(self.batchnorm(self.new_conv_layer(dil_conv3, [3, 3, 256, 3], stride=1, name='conv3'), name="bn6"))
            dil_conv4 = self.leaky_relu(self.batchnorm(self.new_conv_layer(contour, [3,3,input_cnum,256], stride=1, rate=16, name="dil_conv4", padding='SAME'), name="bn7"))
            conv4 = self.relu(self.batchnorm(self.new_conv_layer(dil_conv4, [3, 3, 256, 3], stride=1, name='conv4'), name="bn8"))
        
        return conv1 + conv2 + conv3 + conv4

    def build_LFN( self, contour, reuse=None):
        batch_size = contour.get_shape().as_list()[0]

        #TODO make sure whether this mask is useful in attention
        # offset_flow = None
        # ones_images = tf.ones_like(images)[:, :, :, 0:1]
        # x = tf.concat([images, ones_images, ones_images * mask], axis=3)
        # embed()
        cnum = 64
        x = contour
        input_cnum = x.get_shape().as_list()[3]
        with tf.variable_scope('LFN', reuse=reuse):

            # LFN
            # conv1 = self.leaky_relu(self.batchnorm(self.new_conv_layer(x, [4,4,input_cnum,cnum], stride=2, name="conv1", padding='VALID'), name="bn1"))
            conv1 = self.new_conv_layer(x, [4,4,input_cnum,cnum], stride=2, name="conv1", padding='SAME')
            print(conv1.shape)
            conv2 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv1), [4,4,cnum,cnum*2], stride=2, name="conv2", padding='SAME'), name="LFN_bn2")
            print(conv2.shape)
            conv3 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv2), [4,4,cnum*2,cnum*4], stride=2, name="conv3", padding='SAME'), name="LFN_bn3")
            print(conv3.shape)
            conv4 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv3), [4,4,cnum*4,cnum*8], stride=2, name="conv4", padding='SAME'), name="LFN_bn4")
            print(conv4.shape)
            conv5 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv4), [4,4,cnum*8,cnum*8], stride=2, name="conv5", padding='SAME'), name="LFN_bn5") 
            print(conv5.shape)
            conv6 = self.new_conv_layer(self.leaky_relu(conv5), [4,4,cnum*8,cnum*8], stride=2, name="conv6", padding='SAME')
            print(conv6.shape)
            deconv1 = self.batchnorm(self.new_deconv_layer(self.relu(conv6), conv5.get_shape().as_list(), [4,4,cnum*8,cnum*8], stride=2, name='deconv1'), name="LFN_bn7")
            print(deconv1.shape)
            deconv2 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv1, conv5], axis=3)), conv4.get_shape().as_list(), [4,4,cnum*8,cnum*16], stride=2, name='deconv2'), name="LFN_bn8")
            print(deconv2.shape)
            deconv3 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv2, conv4], axis=3)), conv3.get_shape().as_list(), [4,4,cnum*4,cnum*16], stride=2, name='deconv3'), name="LFN_bn9")
            print(deconv3.shape)
            deconv4 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv3, conv3], axis=3)), conv2.get_shape().as_list(), [4,4,cnum*2,cnum*8], stride=2, name='deconv4'), name="LFN_bn10")
            print(deconv4.shape)
            deconv5 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv4, conv2], axis=3)), conv1.get_shape().as_list(), [4,4,cnum,cnum*4], stride=2, name='deconv5'), name="LFN_bn11")
            print(deconv5.shape)
            deconv6 = self.new_deconv_layer(self.relu(tf.concat([deconv5, conv1], axis=3)), x.get_shape().as_list()[:-1] + [3], [4,4,3,cnum*2], stride=2, name='deconv6')
            print(deconv6.shape)
            x1_stage = tf.nn.tanh(deconv6)

        return x1_stage

    def build_HFN(self, contour, x1_stage, reuse=None):
        x = tf.concat([x1_stage, contour], axis=3)
        input_cnum = x.get_shape().as_list()[3]
        cnum = 64
        with tf.variable_scope('HFN', reuse=reuse):

            # HFN
            conv1 = self.new_conv_layer(x, [4,4,input_cnum,cnum], stride=2, name="conv1", padding='SAME')
            conv2 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv1), [4,4,cnum,cnum*2], stride=2, name="conv2", padding='SAME'), name="HFN_bn2")
            conv3 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv2), [4,4,cnum*2,cnum*4], stride=2, name="conv3", padding='SAME'), name="HFN_bn3")
            conv4 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv3), [4,4,cnum*4,cnum*8], stride=2, name="conv4", padding='SAME'), name="HFN_bn4")
            conv5 = self.batchnorm(self.new_conv_layer(self.leaky_relu(conv4), [4,4,cnum*8,cnum*8], stride=2, name="conv5", padding='SAME'), name="HFN_bn5") 
            conv6 = self.new_conv_layer(self.leaky_relu(conv5), [4,4,cnum*8,cnum*8], stride=2, name="conv6", padding='SAME')

            deconv1 = self.batchnorm(self.new_deconv_layer(self.relu(conv6), conv5.get_shape().as_list(), [4,4,cnum*8,cnum*8], stride=2, name='deconv1'), name="HFN_bn7")
            deconv2 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv1, conv5], axis=3)), conv4.get_shape().as_list(), [4,4,cnum*8,cnum*16], stride=2, name='deconv2'), name="HFN_bn8")
            deconv3 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv2, conv4], axis=3)), conv3.get_shape().as_list(), [4,4,cnum*4,cnum*16], stride=2, name='deconv3'), name="HFN_bn9")
            deconv4 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv3, conv3], axis=3)), conv2.get_shape().as_list(), [4,4,cnum*2,cnum*8], stride=2, name='deconv4'), name="HFN_bn10")
            deconv5 = self.batchnorm(self.new_deconv_layer(self.relu(tf.concat([deconv4, conv2], axis=3)), conv1.get_shape().as_list(), [4,4,cnum,cnum*4], stride=2, name='deconv5'), name="HFN_bn11")
            deconv6 = self.new_deconv_layer(self.relu(tf.concat([deconv5, conv1], axis=3)), x.get_shape().as_list()[:-1] + [3], [4,4,3,cnum*2], stride=2, name='deconv6')
            
            x2_stage = tf.nn.tanh(deconv6)
        return x2_stage#, offset_flow

    def build_conditional_adversarial(self, images, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            cnum = 64
            x = images
            input_cnum = x.get_shape().as_list()[3]
            adv_conv1 = self.leaky_relu(self.batchnorm(self.new_conv_layer(x, [4,4,input_cnum,cnum], stride=2, name="adv_conv1", padding='SAME'), name="bn1"))
            adv_conv2 = self.leaky_relu(self.batchnorm(self.new_conv_layer(adv_conv1, [4,4,cnum,cnum*2], stride=2, name="adv_conv2", padding='SAME'), name="bn2"))
            adv_conv3 = self.leaky_relu(self.batchnorm(self.new_conv_layer(adv_conv2, [4,4,cnum*2,cnum*4], stride=2, name="adv_conv3", padding='SAME'), name="bn3"))
            
            dil_conv1 = self.leaky_relu(self.new_conv_layer(adv_conv3, [4,4,cnum*4,cnum*4], stride=2, rate=2, name="dil_conv1", padding='SAME'))
            dil_conv2 = self.leaky_relu(self.new_conv_layer(adv_conv3, [4,4,cnum*4,cnum*4], stride=2, rate=4, name="dil_conv2", padding='SAME'))
            dil_conv3 = self.leaky_relu(self.new_conv_layer(adv_conv3, [4,4,cnum*4,cnum*4], stride=2, rate=8, name="dil_conv3", padding='SAME'))
            dil_conv4 = self.leaky_relu(self.new_conv_layer(adv_conv3, [4,4,cnum*4,cnum*4], stride=2, rate=12, name="dil_conv4", padding='SAME'))

            combine_x = tf.concat([dil_conv1, dil_conv2, dil_conv3, dil_conv4], axis=3)
            adv_conv4 = self.leaky_relu(self.batchnorm(self.new_conv_layer(combine_x, [4,4,cnum*16,1], stride=2, name="adv_conv4", padding='SAME'), name="bn4"))
            out = self.new_fc_layer(tf.contrib.layers.flatten(adv_conv4), output_size=1, name='output_conditional')
            out_dense = tf.nn.sigmoid(out, name='out_dense')

        return out, out_dense

    def build_local_adversarial(self, images, reuse=None):
        with tf.variable_scope('DIS/local', reuse=reuse):
            cnum = 64
            x = self.leaky_relu(self.new_conv_layer(images, [5,5,3,cnum], stride=2, name="conv1" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum,cnum*2], stride=2, name="conv2" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum*2,cnum*4], stride=2, name="conv3" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum*4,cnum*8], stride=2, name="conv4" ))
            x = tf.contrib.layers.flatten(x) # shape batch_size, :
            out = self.new_fc_layer(x, output_size=1, name='output_local')
        return out

    def build_global_adversarial(self, images, reuse=None):
        with tf.variable_scope('DIS/global', reuse=reuse):
            cnum = 64
            x = self.leaky_relu(self.new_conv_layer(images, [5,5,3,cnum], stride=2, name="conv1" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum,cnum*2], stride=2, name="conv2" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum*2,cnum*4], stride=2, name="conv3" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum*4,cnum*4], stride=2, name="conv4" ))
            x = tf.contrib.layers.flatten(x) # shape batch_size, :
            out = self.new_fc_layer(x, output_size=1, name='output_global')
        return out

    def build_adversarial(self, batch_local, batch_global, mask, reuse=False, training=True):
        
        # d_local = self.build_local_adversarial(batch_local, reuse=reuse, is_train=training)
        d_local, mask_local = self.build_contextual_discriminator(batch_local, mask, reuse=reuse)
        d_global = self.build_global_adversarial(batch_global, reuse=reuse)
        # adv_local = self.new_fc_layer(dlocal, output_size=1, name='output_local')
        # adv_global = self.new_fc_layer(dglobal, output_size=1, name='output_global')
        # return d_local[:, 0], d_global[:, 0]
        return d_local, d_global[:, 0]

    def build_contextual_discriminator(self, images, mask=None, reuse=False):
        with tf.variable_scope('context', reuse=reuse):
            h, w = images.get_shape().as_list()[1:3]
            cnum = 64
            x = self.leaky_relu(self.new_conv_layer(images, [5,5,3,cnum], stride=2, name="context_conv1" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum,cnum*2], stride=2, name="context_conv2" ))
            x = self.leaky_relu(self.new_conv_layer(x, [5,5,cnum*2,cnum*4], stride=2, name="context_conv3" ))
            x = self.new_conv_layer(x, [3, 3, cnum*4, 1], stride=1, name='context_conv4')
            if mask == None:
                mask = tf.ones_like(x)
            else:
                mask = util.max_downsampling(mask, ratio=8)
            x = x * mask
            x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
            mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
            return x, mask_local

