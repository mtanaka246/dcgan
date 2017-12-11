# -*- coding: utf-8 -*-
# example4.py

from dcgan import train
from dcgan.BaseObserver import BaseObserver
from keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
# import scipy.misc


def exe():
    """
    Observerを使用した実装例②

    Keras で Generator を定義し、
    エポック数、バッチサイズ、学習率、β、作業ディレクトを設定して、
    CIFAR 10 の犬の画像を用いて学習を行う

    Observer は学習時に Generator の破損を検出し、復元ポイントにロールバックするために用いる
    （_CustumObserver.on_completed_batch_train()を参照）
    
    :return: None
    """

    # cifar10のデータのロード
    (x_train, y_train), (_, _) = cifar10.load_data()
    # 犬画像の抽出
    dataset = np.array([x for (x, y) in zip(x_train, y_train) if y == 5])
    # -1.0 ～ 1.0 に正規化
    dataset = (dataset - 127.5) / 127.5

    epochs = 25
    data_size_per_batch = 64
    working_dir = "./temp"

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        K.set_session(sess)

        # Generator の生成（Kerasで定義すること）
        generator = _create_keras_generator()

        # 学習
        # （注）システムは内部で "working_dir" 以下に専用のディレクトリを生成し、そこに一時ファイルを生成する
        # fit()が正常に終了する場合、このディレクトリは削除される
        # しかし外部から強制終了した場合、その動作は未定義である
        # また、このディレクトリがすでに存在していた場合（システム内で定義された名前のディレクトリがすでに存在している場合）
        # 一時ファイルおよびそのディレクトリは削除されない
        train.fit(
            generator,
            dataset,
            epochs=epochs,  # エポック数
            data_size_per_batch=data_size_per_batch,  # ミニバッチのデータ数
            d_learning_rate=2.0e-4,  # Discriminator の学習率係数
            d_beta1=0.5,  # Discriminator の beta
            g_learning_rate=2.0e-4,  # Generator の学習率係数
            g_beta1=0.5,  # Generator の beta
            working_dir=working_dir,  # 作業ディレクトリ（存在しない場合は、自動的に生成される）
            rollback_check_point_cycle=200, # 200バッチ毎にロールバック用のチェックポイントを作成
            observer=_CustumObserver(epochs, len(dataset) // data_size_per_batch, working_dir)
        )
        # Generator の保存
        generator.save(os.path.join(working_dir, "generator.h5"))


class _CustumObserver(BaseObserver):
    def __init__(self, epochs, batches, working_dir):
        self.epochs = epochs
        self.batches = batches
        self.working_dir = working_dir

        self.g_loss_moving_avg = 0.0
        self.g_loss_moving_avg_counter = 0
        self.last_check_point = None

    def on_completed_batch_train(self, proxy, epoch_id, batch_id, counter, g_loss, d_loss, elapsed_time):
        """
        バッチ単位の学習完了後にコールされる
        :param proxy: 仲介者
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param g_loss: Generator のロス
        :param d_loss: Discriminator のロス
        :param elapsed_time: 経過時間
        :return: True : 学習を継続, False : 学習を中止
        """
        self.g_loss_moving_avg_counter += 1
        n = min(self.g_loss_moving_avg_counter, 10)
        self.g_loss_moving_avg = ((n - 1) * self.g_loss_moving_avg + g_loss) / n

        # 3回目のエポック以降に Generator ロスの10バッチ移動平均が4.0を超えた場合にロールバックを実行
        if (epoch_id > 2) and (n == 10) and (self.g_loss_moving_avg > 4.0):
            # ロールバックを実行
            check_point = proxy.rollback()

            if check_point == None:
                print "{0} : {1}, {2}".format(
                    "チェックポイントが存在しないことによりロールバックに失敗したため、学習を終了",
                    "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                    "counter = {0}".format(counter)
                )
                return False
            elif check_point == self.last_check_point:
                print "{0} : {1}, {2}".format(
                    "前回と同じチェックポイントにロールバックしたため、学習を終了",
                    "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                    "counter = {0}".format(counter)
                )
                return False
            else:
                self.last_check_point = check_point
                print "{0} : {1}, {2}".format(
                    "ロールバックを実行",
                    "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                    "counter = {0}".format(counter)
                )
                self.g_loss_moving_avg = 0.0
                self.g_loss_moving_avg_counter = 0

        return True

    def on_completed_epoch_train(self, proxy, epoch_id, batch_id, counter, elapsed_time):
        """
        バッチ単位の学習完了後にコールされる
        :param proxy: 仲介者
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param elapsed_time: 経過時間
        :return: True : 学習を継続, False : 学習を中止
        """
        return True


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
