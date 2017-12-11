# -*- coding: utf-8 -*-
# example5.py

from dcgan import train
from dcgan.BaseObserver import BaseObserver
from dcgan.FeatureModel import FeatureModel
from keras.datasets import cifar10
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from keras import backend as K

def exe():
    """
    実践的な実装例

    Keras で Generator を定義し、
    エポック数、バッチサイズ、学習率、β、作業ディレクトを設定して、
    CIFAR 10 の犬の画像を用いて学習を行う

    Observer は学習の進捗を確認するために使用する（_CustumObserver.on_completed_batch_train()を参照）
    　・ミニバッチ処理後にlossの値をコンソールとファイルに出力
    　　（例）Epoch: [  1/ 25] [  50 /  50] time: 00:00:32.986, d_loss: 2.80770445, g_loss: 0.46044752, counter: 50
    　・作業ディレクトリにサンプル画像を出力
    　・Generator の破損を検出し、復元ポイントにロールバック
    　・学習データの一部を用いて、それらのレプリカをGeneratorに生成させる

    :return: None
    """

    # cifar10のデータのロード
    (x_train, y_train), (_, _) = cifar10.load_data()
    # 犬画像の抽出
    org_dataset = np.array([x for (x, y) in zip(x_train, y_train) if y == 5])
    # -1.0 ～ 1.0 に正規化
    dataset = _normalize_img_dataset(org_dataset)

    epochs = 100
    data_size_per_batch = 64
    working_dir = "./temp"
    sample_img_row_col = [8, 8]
    rollback_check_point_cycle = 200

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess, _LogWriter(working_dir) as writer:
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
            rollback_check_point_cycle=rollback_check_point_cycle,  # 200バッチ毎にロールバック用のチェックポイントを作成
            observer=_CustumObserver(
                generator,
                epochs,
                len(dataset) // data_size_per_batch,
                org_dataset[0:sample_img_row_col[0] * sample_img_row_col[1]],
                sample_img_row_col,
                org_dataset[-6:], # Generatorの評価に使用する画像
                working_dir,
                writer
            )
        )

def _normalize_img_dataset(dataset):
    return (dataset / 127.5) - 1

def _denormalize_img_dataset(dataset):
    return (dataset + 1) * 127.5

class _LogWriter:
    def __init__(self, parent_dir):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        self._parent_dir = parent_dir

    def __enter__(self):
        self.log_file = open(os.path.join(self._parent_dir, "log.txt"), "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_file.close()

    def write(self, s):
        print s
        self.log_file.write(s)
        self.log_file.write("\r\n")
        self.log_file.flush()

class _CustumObserver(BaseObserver):
    def __init__(
            self,
            generator,
            epochs,
            batches,
            sample_images,
            sample_img_row_col,
            evaluation_images,
            working_dir,
            writer
    ):
        self.generator = generator
        self.epochs = epochs
        self.batches = batches
        self.working_dir = working_dir
        self.writer = writer

        self.sample_z = np.random.uniform(-1, 1, size=(len(sample_images), generator.input_shape[1]))
        self.sample_images = sample_images
        self.sample_img_row_col = sample_img_row_col

        # 学習中の Generator の評価に使用するパラメータ
        self.evaluation_images = evaluation_images
        self.normalized_evaluation_images = _normalize_img_dataset(self.evaluation_images)
        self.featureModel = FeatureModel(generator, seed=123)

        # self.GeneratorSaveCycle = 100
        self.ImageSamplingCycle = 10

        self.g_loss_moving_avg = 0.0
        self.g_loss_moving_avg_counter = 0
        self.last_check_point = None

    def on_completed_batch_train(
            self,
            proxy,
            epoch_id,
            batch_id,
            counter,
            g_loss,
            d_loss,
            elapsed_time
    ):
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
        # ログファイルとコンソールに出力
        # （例）Epoch: [  1/ 25] [  50 /  50] time: 00:00:32.986, d_loss: 2.80770445, g_loss: 0.46044752, counter: 50
        self.writer.write(
            "Epoch: [{0}] [{1}] time: {2}, {3}, {4}, {5}".format(
                "{0:3d}/{1:3d}".format(epoch_id + 1, self.epochs),
                "{0:4d}/{1:4d}".format(batch_id + 1, self.batches),
                "{0:02.0f}:{1:02.0f}:{2:02.3f}".format(elapsed_time // 3600, elapsed_time // 60 % 60, elapsed_time % 60),
                "d_loss: {0:.8f}".format(d_loss),
                "g_loss: {0:.8f}".format(g_loss),
                "counter: {0:1d}".format(counter)
            )
        )

        # 定期的にサンプル画像を出力
        if np.mod(counter, self.ImageSamplingCycle) == 0:
            self._save_sample_image(proxy, epoch_id, batch_id, counter, g_loss, d_loss)

        # 定期的にGenratorを保存（評価用画像も出力する）
        # if np.mod(counter, self.GeneratorSaveCycle) == 0:
        #     self._save_generator(counter)
        #     self._save_evaluation_image(counter)

        # Genratorの破損を検知した場合ロールバックを実行
        return self._rollback(proxy, epoch_id, batch_id, counter, g_loss, d_loss)

    def _save_generator(self, counter):
        f = os.path.join(self.working_dir, "generator_{0}.h5".format(counter))
        f = os.path.abspath(f)
        self.generator.save(f)
        self.writer.write(f)

    def _save_evaluation_image(self, counter):
        # レプリカ（GAN画像）の生成
        v1 = np.array([self.featureModel.replicate(y) for y in self.normalized_evaluation_images])
        # 非正規化
        v1 = _denormalize_img_dataset(v1)

        # 実画像とそのレプリカとの相関係数の算出
        coef = np.corrcoef(self.evaluation_images.flatten(), v1.flatten())[0, 1]
        self.writer.write("相関係数 : {0}".format(coef))

        v2 = []

        for (e1, e2) in zip(self.evaluation_images, v1):
            v2.append(e1)  # 実画像
            v2.append(e2)  # レプリカ

        images = np.array(v2)
        path = os.path.join(self.working_dir, "generator_{0}_({1:.2}).png".format(counter, coef))
        rows = len(self.evaluation_images) / 2
        cols = len(images) / rows
        # 実画像（奇数列）とレプリカ（偶数列）を並べた1枚の画像を生成し、ファイルに保存
        self._fix_raw_image_and_save_image(images, [rows, cols], path)
        self.writer.write(os.path.abspath(path))

    def _save_sample_image(self, proxy, epoch_id, batch_id, counter, g_loss, d_loss):
        # サンプル画像の生成
        images, d_loss, g_loss = proxy.create_sample_imgages(self.sample_z, self.sample_images)
        # 非正規化
        images = _denormalize_img_dataset(images)
        # ファイルの保存先へのパス
        path = os.path.join(
            self.working_dir,
            'train_{0:02d}_{1:04d}_{2:06d}_d_loss{{{3:.4f}}}_g_loss{{{4:.4f}}}.png'.format(
                epoch_id + 1,
                batch_id + 1,
                counter,
                d_loss,
                g_loss)
        )
        path = os.path.abspath(path)

        # 生成した画像をファイルに保存
        self._fix_raw_image_and_save_image(images, self.sample_img_row_col, path)
        self.writer.write(path)

    def _fix_raw_image_and_save_image(self, images, size, path):
        def _merge(images, size):
            h, w = images.shape[1], images.shape[2]
            img = np.zeros((int(h * size[0]), int(w * size[1]), images.shape[3]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w, :] = image

            return img

        assert (size[0] * size[1]) == len(images), "画像の枚数が一致しない : on_sampling_image()"

        # サンプル画像を1枚の画像にマージ
        images = _merge(images, size)

        import scipy.misc

        # 生成した画像をファイルに保存
        scipy.misc.imsave(path, images.astype(np.uint8))

    def _rollback(self, proxy, epoch_id, batch_id, counter, g_loss, d_loss):
        self.g_loss_moving_avg_counter += 1
        n = min(self.g_loss_moving_avg_counter, 10)
        self.g_loss_moving_avg = ((n - 1) * self.g_loss_moving_avg + g_loss) / n

        # 3回目のエポック以降に Generator ロスの10バッチ移動平均が4.0を超えた場合にロールバックを実行
        if (epoch_id > 2) and (n == 10) and (self.g_loss_moving_avg > 4.0):
            # ロールバックを実行
            check_point = proxy.rollback()

            if check_point == None:
                self.writer.write(
                    "{0} : {1}, {2}".format(
                        "チェックポイントが存在しないことによりロールバックに失敗したため、学習を終了",
                        "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                        "counter = {0}".format(counter)
                    )
                )
                return False
            elif check_point == self.last_check_point:
                self.writer.write(
                    "{0} : {1}, {2}".format(
                        "前回と同じチェックポイントにロールバックしたため、学習を終了",
                        "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                        "counter = {0}".format(counter)
                    )
                )
                return False
            else:
                self.last_check_point = check_point
                self.writer.write(
                    "{0} : {1}, {2}".format(
                        "ロールバックを実行",
                        "(d_loss, g_loss, mean(g_loss)) = ({0}, {1}, {2})".format(d_loss, g_loss, self.g_loss_moving_avg),
                        "counter = {0}".format(counter)
                    )
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
        # Generator の保存
        self._save_generator(counter)
        self._save_evaluation_image(counter)

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
