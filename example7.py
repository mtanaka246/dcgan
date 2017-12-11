# -*- coding: utf-8 -*-
# example7.py

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
        train.fit(generator, dataset, custom_discriminator=custom_discriminator())
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


class custom_discriminator:
    def __init__(self):
        from keras.layers import Input, Dense, Conv2D
        from keras.layers.core import Flatten
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model

        inputs = Input((32, 32, 3), name="input")
        x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding="same", name="conv2d_0")(inputs)
        x = Conv2D(128, (5, 5), strides=(2, 2), activation='relu', padding="same", name="conv2d_1")(x)
        x = BatchNormalization(name="bn_1")(x)
        x = Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding="same", name="conv2d_2")(x)
        x = BatchNormalization(name="bn_2")(x)
        x = Conv2D(512, (5, 5), strides=(2, 2), activation='relu', padding="same", name="conv2d_3")(x)
        x = BatchNormalization(name="bn_3")(x)
        x = Flatten()(x)
        x = Dense(1, name="dense_4")(x)
        self.model = Model(inputs=inputs, outputs=x, name="custom_discriminator")

    def __call__(self, *args, **kwargs):
        from keras.layers import Activation

        input = args[0]
        # is_training = args[1]
        # reuse = kwargs["reuse"]

        d_logits = self.model(input)

        # print model.trainable_weights

        return Activation("sigmoid")(d_logits), d_logits, self.model.trainable_weights

