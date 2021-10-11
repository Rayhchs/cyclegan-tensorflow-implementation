"""
Created on Fri Sep 17 2021

model: tf_units

@author: Ray
"""
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

class tf_units():

    def __init__(self):
        self.k_initilizer = tf.random_normal_initializer(0, 0.02)

    def c7s1_k(self, inputs, filters, reuse=False, name='c7s1_k', do_relu=True, do_norm=True):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            h = tf.layers.conv2d(h,
                                kernel_size=7,
                                filters=filters,
                                strides=1,
                                padding='valid',
                                kernel_initializer=self.k_initilizer,
                                reuse=reuse)
            if do_norm:
                h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                    center=False,
                                                    scale=False)(h)
            if do_relu:
                h = tf.nn.relu(h)
            else:
                h = tf.nn.tanh(h)
            return h

    def dk(self, inputs, filters, reuse=False, name='downsample_dk'):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = tf.layers.conv2d(inputs,
                                kernel_size=3,
                                filters=filters,
                                strides=2,
                                padding='same',
                                kernel_initializer=self.k_initilizer,
                                reuse=reuse)
            h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                center=False,
                                                scale=False)(h)
            h = tf.nn.relu(h)
            return h

    def Rk(self, inputs, filters, reuse=False, name='residual_Rk'):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            h = tf.layers.conv2d(h, kernel_size=3,
                                filters=filters,
                                strides=1,
                                padding='valid',
                                kernel_initializer=self.k_initilizer,
                                reuse=reuse)
            h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                center=False,
                                                scale=False)(h)
            h = tf.nn.relu(h)
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            h = tf.layers.conv2d(h,
                                kernel_size=3,
                                filters=filters,
                                strides=1,
                                padding='valid',
                                kernel_initializer=self.k_initilizer,
                                reuse=reuse)
            h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                center=False,
                                                scale=False)(h)
            return tf.add(h, inputs)

    def uk(self, inputs, filters, reuse=False, name='upsample_uk'):
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            #h = tf.keras.layers.UpSampling2D(interpolation='bilinear')(inputs)
            #h = tf.layers.conv2d(h, kernel_size=3,
            #                    filters=filters,
            #                    strides=1,
            #                    padding='same',
            #                    kernel_initializer=self.k_initilizer,
            #                    reuse=reuse)
            h = tf.layers.conv2d_transpose(inputs,
                                           kernel_size=3,
                                           filters=filters,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=self.k_initilizer,
                                           reuse=reuse)
            h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                center=False,
                                                scale=False)(h)
            h = tf.nn.relu(h)
            return h

    def d_block(self, inputs, filters, strides, padding='same', reuse=False, do_relu=True, do_norm=True, name='discriminator_block'):
        """
        par inputs: layer
        par filters: filters
        par strides: strides
        """
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = tf.layers.conv2d(inputs,
                                 kernel_size=4,
                                 filters=filters,
                                 strides=strides,
                                 padding=padding,
                                 kernel_initializer=self.k_initilizer,
                                 reuse=reuse)
            if do_norm:
                h = tfa.layers.InstanceNormalization(epsilon=1e-5,
                                                    center=False,
                                                    scale=False)(h)

            if do_relu:
                h = tf.nn.leaky_relu(h, alpha=0.2)
                
            return h

    def discriminator_loss(self, real, fake):
        loss_d_real = tf.reduce_mean(tf.squared_difference(real, tf.ones_like(real)))
        loss_d_fake = tf.reduce_mean(tf.square(fake))
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        return loss_d

    def generator_loss(self, fake):
        loss_g = tf.reduce_mean(tf.squared_difference(fake, tf.ones_like(fake)))
        return loss_g

    def CC_loss(self, real, recon, L1_lambda=10):
        loss_cc = tf.reduce_mean(tf.abs(real - recon))
        return loss_cc * L1_lambda

    def idt_loss(self, real, same, L1_lambda=10):
        loss = tf.reduce_mean(tf.abs(real - same))
        return 0.5 * L1_lambda * loss