"""
Created on Moon Festival + 1 2021

model: cyclegan

@author: Ray
"""
import os, sys, random, time
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tf_units import tf_units
from utils import *

class cyclegan():

    def __init__(self, sess, args):
        self.sess = sess
        self.L1_lambda = args.lambda_c
        self.do_idt = args.do_idt
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.decay = args.decay
        self.do_resize = args.do_resize
        self.units = tf_units()
        self.bulid_model()

    def bulid_model(self):
        """
        init model
        """
        # init variable
        self.A_real = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='x')
        self.B_real = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='y')
        
        # generator
        self.g_B = self.generator(self.A_real, name='generator_G') # A2B
        self.g_A = self.generator(self.B_real, name='generator_F') # B2A
        self.DA_fake = self.discriminator(self.g_A, name="discriminatorA")
        self.DB_fake = self.discriminator(self.g_B, name="discriminatorB")

        self.A_recon = self.generator(self.g_B, name='generator_F', reuse=True) # B2A
        self.B_recon = self.generator(self.g_A, name='generator_G', reuse=True) # A2B

        # identity
        if self.do_idt:
            self.same_A = self.generator(self.A_real, name='generator_F', reuse=True)
            self.same_B = self.generator(self.B_real, name='generator_G', reuse=True)

        # Use second fake image for training discriminator
        self.A_fake_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='A_fake')
        self.B_fake_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='B_fake')

        self.DA_real = self.discriminator(self.A_real, name="discriminatorA", reuse=True)
        self.DB_real = self.discriminator(self.B_real, name="discriminatorB", reuse=True)

        self.DA_fake_input = self.discriminator(self.A_fake_input, name="discriminatorA", reuse=True)
        self.DB_fake_input = self.discriminator(self.B_fake_input, name="discriminatorB", reuse=True)

        # loss
        if self.do_idt:
            self.loss_ga2b, self.loss_gb2a = self.G_loss(self.A_real, self.B_real,
                                                 self.DA_fake, self.DB_fake,
                                                 self.DA_real, self.DB_real,
                                                 self.A_recon, self.B_recon,
                                                 self.same_A, self.same_B)
        else:
            self.loss_ga2b, self.loss_gb2a = self.G_loss(self.A_real, self.B_real,
                                                 self.DA_fake, self.DB_fake,
                                                 self.DA_real, self.DB_real,
                                                 self.A_recon, self.B_recon,
                                                 None, None)

        self.loss_da, self.loss_db = self.D_loss(self.DA_fake_input, self.DB_fake_input, self.DA_real, self.DB_real)

        # summary
        tf.summary.scalar("loss_ga2b", self.loss_ga2b)
        tf.summary.scalar("loss_gb2a", self.loss_gb2a)
        tf.summary.scalar("loss_da", self.loss_da)
        tf.summary.scalar("loss_db", self.loss_db)
        self.merged = tf.summary.merge_all()

        # vars
        self.vars_g_a2b = [var for var in tf.trainable_variables() if 'generator_G' in var.name]
        self.vars_g_b2a = [var for var in tf.trainable_variables() if 'generator_F' in var.name]
        self.vars_d_a = [var for var in tf.trainable_variables() if 'discriminatorA' in var.name]
        self.vars_d_b = [var for var in tf.trainable_variables() if 'discriminatorB' in var.name]

        # saver
        self.saver = tf.train.Saver()

    def generator(self, x, reuse=False, name="generator", n_block=9):
        """
        Network with residual block
        """
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = self.units.c7s1_k(x, filters=64, name='c7s1_64', reuse=reuse)
            h = self.units.dk(h, filters=128, name='d128', reuse=reuse)
            h = self.units.dk(h, filters=256, name='d256', reuse=reuse)
            for i in range(n_block):
                h = self.units.Rk(h, filters=256, name=f'R256_{i}', reuse=reuse)
            h = self.units.uk(h, filters=128, name='u128', reuse=reuse)
            h = self.units.uk(h, filters=64, name='u64', reuse=reuse)
            h = self.units.c7s1_k(h, filters=3, name='c7s1_3', do_relu=False, do_norm=False, reuse=reuse)

        return h

    def discriminator(self, x, reuse=False, name="discriminator"):
        """
        PatchGAN discriminator
        """
        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = self.units.d_block(x, filters=64, strides=2, do_norm=False, name='c64', reuse=reuse)
            h = self.units.d_block(h, filters=128, strides=2, name='c128', reuse=reuse)
            h = self.units.d_block(h, filters=256, strides=2, name='c256', reuse=reuse)
            h = self.units.d_block(h, filters=512, strides=1, padding='same', name='c512', reuse=reuse)
            h = self.units.d_block(h, filters=1, strides=1, padding='same', do_norm=False, do_relu=False, name='output', reuse=reuse)

        return h

    def D_loss(self, DA_fake, DB_fake, DA_real, DB_real):
        
        # discriminator
        loss_db = self.units.discriminator_loss(DB_real, DB_fake)
        loss_da = self.units.discriminator_loss(DA_real, DA_fake)

        return loss_da, loss_db

    def G_loss(self, A_real, B_real, DA_fake, DB_fake, DA_real, DB_real, A_recon, B_recon, same_A, same_B):

        # GAN loss
        loss_g_a2b = self.units.generator_loss(DB_fake)
        loss_g_b2a = self.units.generator_loss(DA_fake)

        # cycle consistency loss
        loss_cc_a2b2a = self.units.CC_loss(A_real, A_recon, self.L1_lambda)
        loss_cc_b2a2b = self.units.CC_loss(B_real, B_recon, self.L1_lambda)
        CC_loss = loss_cc_a2b2a + loss_cc_b2a2b

        if self.do_idt:

            # identity loss
            idt_loss_a = self.units.idt_loss(A_real, same_A, self.L1_lambda)
            idt_loss_b = self.units.idt_loss(B_real, same_B, self.L1_lambda)
            loss_ga2b = loss_g_a2b + CC_loss + idt_loss_b
            loss_gb2a = loss_g_b2a + CC_loss + idt_loss_a
        else:
            loss_ga2b = loss_g_a2b + CC_loss
            loss_gb2a = loss_g_b2a + CC_loss

        return loss_ga2b, loss_gb2a

    def train(self, A_paths, B_paths):
        """
        par A_paths, B_paths: list of image paths
        par epochs: epochs
        par batch_size: batch size
        """
        # Optimizer
        train_step_da = tf.train.AdamOptimizer(beta1=0.5).minimize(self.loss_da, var_list=self.vars_d_a)
        train_step_db = tf.train.AdamOptimizer(beta1=0.5).minimize(self.loss_db, var_list=self.vars_d_b)
        train_step_ga2b = tf.train.AdamOptimizer(learning_rate=0.0002,
                                              beta1=0.5).minimize(self.loss_ga2b, var_list=self.vars_g_a2b)
        train_step_gb2a = tf.train.AdamOptimizer(learning_rate=0.0002,
                                              beta1=0.5).minimize(self.loss_gb2a, var_list=self.vars_g_b2a)

        # init variable
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./log", self.sess.graph)
        #self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))

        # Training
        for i in range(self.epochs):
            batch_num = int(np.ceil(len(A_paths) / self.batch_size))
            batch_list_A = np.array_split(A_paths, batch_num)
            batch_list_B = np.array_split(B_paths, batch_num)
            random.shuffle(batch_list_A)
            random.shuffle(batch_list_B)

            if(i < self.decay) :
                train_step_da.learning_rate = 0.0002
                train_step_db.learning_rate = 0.0002
            else:
                train_step_da.learning_rate = 0.0002 - 0.0002*(i-self.decay)/self.decay
                train_step_db.learning_rate = 0.0002 - 0.0002*(i-self.decay)/self.decay

            for j in range(len(batch_list_A)):
                t1 = time.time()

                batch_x, batch_y = load_data(batch_list_A[j], batch_list_B[j])
                fake_A_input, fake_B_input = load_data(batch_list_A[j-1], batch_list_B[j-1])

                _, loss_ga2b, _, loss_gb2a = self.sess.run([train_step_ga2b, self.loss_ga2b, train_step_gb2a, self.loss_gb2a],
                                            feed_dict={self.A_real: batch_x, self.B_real: batch_y})

                gen_A, gen_B = self.sess.run([self.g_A, self.g_B], feed_dict={self.A_real: fake_A_input, self.B_real: fake_B_input})

                _, loss_db, _, loss_da = self.sess.run([train_step_db, self.loss_db, train_step_da, self.loss_da],
                                            feed_dict={self.A_real: batch_x, self.B_real: batch_y, self.A_fake_input: gen_A, self.B_fake_input: gen_B})

                print("%d/%d -loss_da: %.4f -loss_db: %.4f -loss_ga2b: %.4f -loss_gb2a: %.4f time: %.1fs" %
                    ((j + 1), len(batch_list_A), loss_da, loss_db, loss_ga2b, loss_gb2a, time.time()-t1))

                # Test 
                if j % 100 == 0:
                    g = self.sess.run(self.g_B, feed_dict={self.A_real: batch_x})
                    g_2 = self.sess.run(self.g_A, feed_dict={self.B_real: batch_y})
                    g = g[0] * 127.5 + 127.5
                    g_2 = g_2[0] * 127.5 + 127.5
                    g = Image.fromarray(np.uint8(g))
                    g_2 = Image.fromarray(np.uint8(g_2))
                    o1 = batch_x[0] * 127.5 + 127.5
                    o1 = Image.fromarray(np.uint8(o1))
                    g.save('testA2B.jpg')
                    g_2.save('testB2A.jpg')
                    o1.save('o.jpg')

            # Save loss
            summary = self.sess.run(self.merged, feed_dict={self.A_real: batch_x, self.B_real: batch_y, self.A_fake_input: gen_A, self.B_fake_input: gen_B})
            self.writer.add_summary(summary, global_step=i)

            # Save model
            if (i + 1) % 5 == 0:
                self.saver.save(self.sess, './checkpoint/epoch_%d.ckpt' % (i + 1))

    def test(self, paths, style):
        """
        par paths: list of image path
        par style: A2B or B2A
        """
        # init variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load model
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))

        im = []
        for j in range(len(paths)):
            batch_x = load_test_data((paths[j]))

            if style.lower() == "a2b":
                g = self.sess.run(self.g_B, feed_dict={self.A_real: batch_x})
            elif style.lower() == "b2a":
                g = self.sess.run(self.g_A, feed_dict={self.B_real: batch_x})

            g = (np.array(g[0]) + 1) * 127.5
            
            if self.do_resize:
                g = resize(paths[j], g).eval()
            im.append(Image.fromarray(np.uint8(g)))

        return im