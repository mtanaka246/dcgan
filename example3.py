# -*- coding: utf-8 -*-
# example3.py

from dcgan import train
from dcgan.BaseObserver import BaseObserver
from keras.datasets import cifar10
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from keras import backend as K

def exe():
    """
    Observerを使用した実装例①

    Keras で Generator を定義し、
    エポック数、バッチサイズ、学習率、β、作業ディレクトを設定して、
    CIFAR 10 の犬の画像を用いて学習を行う
    
    Observer は学習の進捗を確認するために使用する（_CustumObserver.on_completed_batch_train()を参照）
    　・ミニバッチ処理後にlossの値をコンソールに出力
    　　（例）Epoch: [  1/ 25] [  50 /  50] time: 00:00:32.986, d_loss: 2.80770445, g_loss: 0.46044752, counter: 50
    　・作業ディレクトリにサンプル画像を出力する

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
    sample_img_row_col = [8, 8]

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
            observer=_CustumObserver(
                epochs,  # エポック数
                len(dataset) // data_size_per_batch,  # バッチ数
                dataset[0:sample_img_row_col[0] * sample_img_row_col[1]],
                sample_img_row_col,
                working_dir
            )
        )

        # Generator の保存
        generator.save(os.path.join(working_dir, "generator.h5"))

class _CustumObserver(BaseObserver):
    def __init__(self, epochs, batches, sample_images, sample_img_row_col, working_dir):
        self.epochs = epochs
        self.batches = batches
        self.working_dir = working_dir

        self.sample_z = np.random.uniform(-1, 1, size=(len(sample_images), 100))
        self.sample_images = sample_images
        self.sample_img_row_col = sample_img_row_col

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
        # コンソールに出力
        # （例）Epoch: [  1/ 25] [  50 /  50] time: 00:00:32.986, d_loss: 2.80770445, g_loss: 0.46044752, counter: 50
        print "Epoch: [{0}] [{1}] time: {2}, {3}, {4}, {5}".format(
            "{0:3d}/{1:3d}".format(epoch_id + 1, self.epochs),
            "{0:4d}/{1:4d}".format(batch_id + 1, self.batches),
            "{0:02.0f}:{1:02.0f}:{2:02.3f}".format(elapsed_time // 3600, elapsed_time // 60 % 60, elapsed_time % 60),
            "d_loss: {0:.8f}".format(d_loss),
            "g_loss: {0:.8f}".format(g_loss),
            "counter: {0:1d}".format(counter)
        )

        # 10バッチ毎にサンプル画像を出力
        if np.mod(counter, 10) == 0:
            # サンプル画像の生成
            images, d_loss, g_loss = proxy.create_sample_imgages(self.sample_z, self.sample_images)

            def _merge(images, size):
                h, w = images.shape[1], images.shape[2]
                img = np.zeros((int(h * size[0]), int(w * size[1]), images.shape[3]))
                for idx, image in enumerate(images):
                    i = idx % size[1]
                    j = idx // size[1]
                    img[j * h:j * h + h, i * w:i * w + w, :] = image

                return img

            assert (self.sample_img_row_col[0] * self.sample_img_row_col[1]) == len(images),\
                "画像の枚数が一致しない : on_sampling_image()"

            # 画像を 0 ～ 255 に戻す
            images = (images + 1.0) * 127.5
            # サンプル画像を1枚の画像にマージ
            images = _merge(images, self.sample_img_row_col)

            # ファイルの保存先へのパス
            image_path = os.path.join(
                self.working_dir,
                'train_{0:02d}_{1:04d}_{2:06d}_d_loss{{{3:.4f}}}_g_loss{{{4:.4f}}}.png'.format(
                    epoch_id + 1,
                    batch_id + 1,
                    counter,
                    d_loss,
                    g_loss)
            )

            print image_path

            # 生成した画像をファイルに保存
            scipy.misc.imsave(image_path, images.astype(np.uint8))

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
