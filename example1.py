# -*- coding: utf-8 -*-
# example1.py

from keras.datasets import cifar10
from dcgan import train
import numpy as np
import tensorflow as tf
from keras import backend as K

def exe():
    """
    最も簡単な実装例 
    
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
        # （注）システムは内部でカレントディレクトリ以下に専用のディレクトリを生成し、そこに一時ファイルを生成する
        # fit()が正常に終了する場合、このディレクトリは削除される
        # しかし外部から強制終了した場合、その動作は未定義である
        # また、このディレクトリがすでに存在していた場合（システム内で定義された名前のディレクトリがすでに存在している場合）
        # 一時ファイルおよびそのディレクトリは削除されない
        train.fit(generator, dataset)

        # Generator の保存
        generator.save("generator.h5")

def _create_keras_generator():
    from keras.models import Sequential
    from keras.layers import Dense, Reshape, Activation
    from keras.layers.convolutional import Conv2DTranspose
    from keras import initializers

    w = 2
    h = 2
    size =  64

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
