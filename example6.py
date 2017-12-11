# -*- coding: utf-8 -*-
# example6.py

from keras.datasets import cifar10
from keras import backend as K
from dcgan import train
import numpy as np
import tensorflow as tf


def exe():
    """
    Discriminator を独自に定義するための実装例

    Keras で Generator を定義し、CIFAR 10 の犬の画像を用いて学習を行う

    :return: None
    """
    # cifar10のデータのロード
    (x_train, y_train), (_, _) = cifar10.load_data()
    # 犬画像の抽出
    dataset = np.array([x for (x, y) in zip(x_train, y_train) if y == 5])
    # -1.0 ～ 1.0 に正規化
    dataset = (dataset - 127.5) / 127.5

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        K.set_session(sess)

        # Generator の生成（Kerasで定義すること）
        generator = _create_keras_generator()

        # 学習
        train.fit(generator, dataset, custom_discriminator=_create_discriminator)
        # Generator の保存
        generator.save("generator.h5")


def _create_keras_generator():
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Activation
    from keras.layers.convolutional import Conv2DTranspose
    from keras import initializers

    w = 2
    h = 2
    size = 64

    model = Sequential()

    model.add(Dense(units=w * h * size * 8, input_dim=100))  # 1
    model.add(Reshape((w, h, size * 8)))  # 2
    model.add(Activation('relu'))  # 3
    # shape : (2, 2, 512)

    model.add(Conv2DTranspose(
        size * 8,
        (5, 5),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializers.random_normal(stddev=0.02)))  # 4
    model.add(Activation('relu'))  # 6
    # shape : (4, 4, 512)

    model.add(Conv2DTranspose(
        size * 4,
        (5, 5),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializers.random_normal(stddev=0.02)))  # 7
    model.add(Activation('relu'))  # 8
    # shape : (8, 8, 256)

    model.add(Conv2DTranspose(
        size * 2,
        (5, 5),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializers.random_normal(stddev=0.02)))  # 9
    model.add(Activation('relu'))  # 10
    # shape : (16, 16, 128)

    model.add(Conv2DTranspose(
        3,
        (5, 5),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializers.random_normal(stddev=0.02)))  # 11
    model.add(Activation('tanh'))  # 12
    # shape : (32, 32, 3)

    # model.summary()

    return model

def _create_discriminator(input, is_training, reuse=False):
    """
    Discriminator を生成する関数
    :param input: プレースホルダー
    :param is_training: プレースホルダー
    :param reuse: bool NNの重みを新規に作成する場合はTrue, 既存の重みを再利用する場合はFalse
    :return: NNの出力, logit, 学習パラメータ
    """
    import tensorflow as tf

    name = "custum_discriminator"

    # 参考URL
    # http://bamos.github.io/2016/08/09/deep-completion/#ml-heavy-dcgans-in-tensorflow
    # https://github.com/bamos/dcgan-completion.tensorflow/blob/master/model.py

    with tf.variable_scope(name, reuse=reuse) as scope:
        def conv2d(input, output_dim, name):
            with tf.variable_scope(name):
                k_h = 5
                k_w = 5
                d_h = 2
                d_w = 2
                stddev = 0.02
                w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding='SAME')
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

                return tf.nn.bias_add(conv, biases)

        def lrelu(x):
            leak = 0.2

            with tf.variable_scope("lrelu"):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)

        def batch_normalization(x, is_train, name):
            momentum = 0.9
            epsilon = 1e-5

            return tf.contrib.layers.batch_norm(
                x,
                decay=momentum,
                updates_collections=None,
                epsilon=epsilon,
                center=True,
                scale=True,
                is_training=is_train,
                scope=name)

        def linear(input_, output_size, scope):
            stddev = 0.02
            bias_start = 0.0
            with_w = False
            rand_seed = None
            shape = input_.get_shape().as_list()

            with tf.variable_scope(scope):
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev, seed=rand_seed))
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias

        output_dim = 64

        h0 = lrelu(conv2d(input, output_dim, 'd_h0_conv'))
        h1 = lrelu(batch_normalization(conv2d(h0, output_dim*2, 'd_h1_conv'), is_training, 'd_bch_h1'))
        h2 = lrelu(batch_normalization(conv2d(h1, output_dim*4, 'd_h2_conv'), is_training, 'd_bch_h2'))
        h3 = lrelu(batch_normalization(conv2d(h2, output_dim*8, 'd_h3_conv'), is_training, 'd_bch_h3'))
        h4 = linear(tf.reshape(h3, [-1, 2 * 2 * (output_dim * 8)]), 1, 'd_h4_lin')

    t_vars = tf.trainable_variables()

    return tf.nn.sigmoid(h4), h4, [var for var in t_vars if var.name.startswith(name)]
