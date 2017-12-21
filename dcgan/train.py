# -*- coding: utf-8 -*-
# train.py

from BaseObserver import BaseObserver
import time
import numpy as np
import os
import tensorflow as tf
from keras import backend as K

class _EmptyObserver(BaseObserver):
    def on_completed_batch_train(self, facade, epoch_id, batch_id, counter, g_loss, d_loss, elapsed_time):
        """
        バッチ単位の学習完了後にコールされる
        :param facade: 仲介者
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param g_loss: Generator のロス
        :param d_loss: Discriminator のロス
        :param elapsed_time: 経過時間
        :return: True : 学習を継続, False : 学習を中止
        """
        return True

    def on_completed_epoch_train(self, facade, epoch_id, batch_id, counter, elapsed_time):
        """
        バッチ単位の学習完了後にコールされる
        :param facade: 仲介者
        :param epoch_id: 現在のエポック番号
        :param batch_id: 現在のパッチ番号
        :param counter: 実行したバッチ処理の回数
        :param elapsed_time: 経過時間
        :return: True で学習を継続, False で学習を中止
        """
        return True


from BaseDataSet import BaseDataSet


class _ListDataSet(BaseDataSet):
    def __init__(self, dataset, data_size_per_batch):
        self._dataset = dataset
        self._data_size_per_batch = data_size_per_batch

    def batch(self):
        """
        バッチデータをリストで返すジェネレータ
        （呼び出し側は正規化済みのデータが帰ることを前提としている）
        :return: バッチデータ
        """
        for batch_id in range(self.batch_size()):
            first = batch_id * self.size_per_batch()
            last = first + self.size_per_batch()
            yield self._dataset[first:last]

    def batch_size(self):
        """
        1エポックのバッチ数を返す
        :return: 1エポックのバッチ数
        """
        return self.size() // self.size_per_batch()

    def size(self):
        """
        全データ数を返す
        :return: 全データ数
        """
        return len(self._dataset)

    def shape(self):
        """
        データのタプル
        :return: (全レコード数, 1データの高さ, 幅, 深さ)のタプル
        """
        return self._dataset.shape

    def size_per_batch(self):
        """
        1バッチのデータ数を返す
        :return: 1バッチのデータ数
        """
        return self._data_size_per_batch

def fit(generator,
        dataset,
        epochs=25,
        data_size_per_batch=64,
        d_learning_rate = 2.0e-4,
        d_beta1 = 0.5,
        g_learning_rate = 2.0e-4,
        g_beta1=0.5,
        rollback_check_point_cycle=500,
        working_dir = "./",
        custom_discriminator=None,
        observer=_EmptyObserver()
        ):
    if not len(dataset.shape) == 4:
        raise ValueError("学習データのshapeがシステム仕様を満たしていません : (レコード数, 幅, 高さ,深さ)")

    fit_dataset(
        generator,
        _ListDataSet(dataset, data_size_per_batch),
        epochs=epochs,
        d_learning_rate = d_learning_rate,
        d_beta1 = d_beta1,
        g_learning_rate = g_learning_rate,
        g_beta1=g_beta1,
        rollback_check_point_cycle=rollback_check_point_cycle,
        working_dir = working_dir,
        custom_discriminator=custom_discriminator,
        observer=observer
    )

def fit_dataset(generator,
        dataset,
        epochs=25,
        d_learning_rate = 2.0e-4,
        d_beta1 = 0.5,
        g_learning_rate = 2.0e-4,
        g_beta1=0.5,
        rollback_check_point_cycle=500,
        working_dir = "./",
        custom_discriminator=None,
        observer=_EmptyObserver()
        ):

    if (dataset is None) or (dataset.size() == 0):
        raise ValueError("学習データがありません")

    if not len(dataset.shape()) == 4:
        raise ValueError("学習データのshapeがシステム仕様を満たしていません : (レコード数, 幅, 高さ,深さ)")

    param = _Parameter(
        image_shape=dataset.shape()[1:],
        working_dir=working_dir,
        epochs=epochs,
        d_learning_rate=d_learning_rate,
        d_beta1=d_beta1,
        g_learning_rate=g_learning_rate,
        g_beta1=g_beta1,
        rollback_check_point_cycle=rollback_check_point_cycle
    )

    # 一時ファイルの出力用ディレクトリの作成
    delete_flg = _create_checkpoint_dir(param)

    try:
        # 学習開始後の最初のモデル保存でロールバックが発生したような場合に陥ることがある
        # その対策として、ここでモデルの保存をして、そのファイルをすぐに削除する
        f = os.path.join(param.checkpoint_dir, 'generator_0.h5')
        generator.save(f)
        os.remove(f)

        _do_fitting(
            K.get_session(),
            param,
            generator,
            dataset,
            custom_discriminator,
            observer
        )
    except:
        # 一時ファイルの出力用ディレクトリの削除
        _delete_checkpoint_dir(param, delete_flg)
        raise

    # 一時ファイルの出力用ディレクトリの削除
    _delete_checkpoint_dir(param, delete_flg)


def _create_checkpoint_dir(param):
    if not os.path.exists(param.checkpoint_dir):
        os.makedirs(param.checkpoint_dir)
        print "create checkpoint directory : {0}".format(os.path.abspath(param.checkpoint_dir))

        return True
    else:
        return False

def _delete_checkpoint_dir(param, delete_flg):
    if not delete_flg:
        return

    import shutil

    shutil.rmtree(param.checkpoint_dir)
    print "delete checkpoint directory : {0}".format(os.path.abspath(param.checkpoint_dir))

def _do_fitting(
        sess,
        param,
        generator,
        dataset,
        custom_discriminator,
        observer
):
    tf_var = _TensorflowVariable(param, generator, custom_discriminator)
    batches = dataset.batch_size()
    counter = 0
    proxy = _Proxy(sess, param, tf_var)

    try:
        tf.global_variables_initializer().run()
    except:
        tf.initialize_all_variables().run()

    start_time = time.time()

    for epoch_id in xrange(param.epochs):
        for batch_id, batch_images in enumerate(dataset.batch()):
            counter += 1
            batch_z = np.random.uniform(-1, 1, [dataset.size_per_batch(), generator.input_shape[1]]).astype(np.float32)

            # Update D network
            sess.run([tf_var.d_optim], feed_dict={tf_var.images:batch_images, tf_var.z:batch_z, tf_var.is_training:True, K.learning_phase():1})
            # Update G network
            sess.run([tf_var.g_optim], feed_dict={tf_var.z:batch_z, tf_var.is_training:True, K.learning_phase():1})
            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            sess.run([tf_var.g_optim], feed_dict={tf_var.z:batch_z, tf_var.is_training:True, K.learning_phase():1})

            # print 出力用
            errD_fake = tf_var.d_loss_fake.eval({tf_var.z:batch_z, tf_var.is_training:False, K.learning_phase():0})
            errD_real = tf_var.d_loss_real.eval({tf_var.images:batch_images, tf_var.is_training:False, K.learning_phase():0})
            errG = tf_var.g_loss.eval({tf_var.z:batch_z, tf_var.is_training:False, K.learning_phase():0})

            if not observer.on_completed_batch_train(
                proxy,
                epoch_id,
                batch_id,
                counter,
                errG,
                errD_fake + errD_real,
                time.time() - start_time
            ):
                return

            if np.mod(counter, param.rollback_check_point_cycle) == 0:
                _save(tf_var, param, sess, counter)

        if not observer.on_completed_epoch_train(
            proxy,
            epoch_id,
            batch_id,
            counter,
            time.time() - start_time
        ):
            return

_discriminator_scope_name = 'discriminator'
_is_training_name = 'is_training'
_images_name = 'real_images'
_z_name = 'z'

_SystemUniqueNames = [
    _discriminator_scope_name,
    _is_training_name,
    _images_name,
    _z_name,
]

def _save_generator(generator, param, epoch, idx, counter, observer):
    file = os.path.join(param.working_dir, 'generator_{0}.h5'.format(counter))
    generator.save(file)
    observer.on_sampling_generator(generator, epoch, idx, counter)

def _save(tf_var, param, sess, step):
    if not os.path.exists(param.checkpoint_dir):
        os.makedirs(param.checkpoint_dir)
        print "Directory created : ", param.checkpoint_dir

    path = os.path.join(param.checkpoint_dir, param.model_name)
    tf_var.saver.save(sess, path, global_step=step)

def _load(tf_var, sess, checkpoint_dir, index=None):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        if index == None:
            tf_var.saver.restore(sess, ckpt.model_checkpoint_path)
            return ckpt.model_checkpoint_path
        elif index < len(ckpt.all_model_checkpoint_paths):
            tf_var.saver.restore(sess, ckpt.all_model_checkpoint_paths[index])
            return ckpt.all_model_checkpoint_paths[index]
        else:
            return None
    else:
        return None

def _discriminator(param, inputs, is_training, reuse=False):
    if (param.image_shape[0] < 16) or (param.image_shape[1] < 16):
        raise ValueError("デフォルト Discriminator の定義仕様では学習データの幅と高さは16以上必要です : ({0}, {1})".formatparam.image_shape[0], param.image_shape[1])

    with tf.variable_scope(_discriminator_scope_name) as scope:
        if reuse:
            scope.reuse_variables()
            # print "reuse : True"

        # TODO: Investigate how to parameterise discriminator based off inputs sampling_image_size.
        # print param.image_shape
        # shpae(inputs) = (?, 64, 64, 1)

        depth = max(param.image_shape[0], param.image_shape[1])

        h0 = _lrelu(_conv2d(inputs, depth, name='d_h0_conv'))
        # shpae(h0) = (?, 32, 32, 64(= depth(= 64)))

        depth *= 2
        h1 = _lrelu(param.d_bns[0](_conv2d(h0, depth, name='d_h1_conv'), is_training))
        # shpae(h1) = (?, 16, 16, 128(= depth(= 64) * 2))

        depth *= 2
        h2 = _lrelu(param.d_bns[1](_conv2d(h1, depth, name='d_h2_conv'), is_training))
        # shpae(h2) = (?, 8, 8, 256(= depth(= 64) * 4))

        depth *= 2
        h3 = _lrelu(param.d_bns[2](_conv2d(h2, depth, name='d_h3_conv'), is_training))
        # shpae(h3) = (?, 4, 4, 512( = depth(= 64) * 8))

        w = param.image_shape[0] / 16
        h = param.image_shape[1] / 16
        h4 = _linear(tf.reshape(h3, [-1, w * h * depth]), 1, 'd_h4_lin')
        # shpae(h4) = (?, 8192(=4 * 4 * 512))

    t_vars = tf.trainable_variables()

    # for var in t_vars:
    #     if _discriminator_scope_name in var.name:
    #         print "***", var.name

    return tf.nn.sigmoid(h4), h4, [var for var in t_vars if var.name.startswith(_discriminator_scope_name)]

def _conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        # print input_.get_shape()[:]
        # # print [k_h, k_w, input_.get_shape()[-1], output_dim]

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def _conv2d_transpose(input_,
                     output_shape,
                     k_h=5,
                     k_w=5,
                     d_h=2,
                     d_w=2,
                     stddev=0.02,
                     name="conv2d_transpose",
                     with_w=False,
                     rand_seed = None):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev, seed = rand_seed))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def _lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def _linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, rand_seed=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev, seed = rand_seed))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class _batch_norm():
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,
            epsilon=self.epsilon,
            center=True,
            scale=True,
            is_training=train,
            scope=self.name)

class _Parameter():
    def __init__(self,
                 image_shape,
                 working_dir,
                 epochs,
                 d_learning_rate,
                 d_beta1,
                 g_learning_rate,
                 g_beta1,
                 rollback_check_point_cycle,
                 model_name="dcgan_generator_model"):
        self.image_shape = image_shape

        self.epochs = epochs

        self.d_learning_rate=d_learning_rate
        self.d_beta1=d_beta1
        self.g_learning_rate=g_learning_rate
        self.g_beta1=g_beta1

        self.rollback_check_point_cycle = rollback_check_point_cycle

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [_batch_norm(name='d_bn{}'.format(i, )) for i in range(4)]

        import math

        log_size = int(math.log(image_shape[0]) / math.log(2))
        self.g_bns = [_batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]
        self.working_dir = working_dir

        self.checkpoint_dir = os.path.join(working_dir, "checkpoint")
        self.model_name = model_name

class _TensorflowVariable:
    def __init__(self, param, generator, custom_discriminator):
        self.is_training = tf.placeholder(tf.bool, name=_is_training_name)
        self.images = tf.placeholder(tf.float32, [None] + list(param.image_shape), name=_images_name)
        self.z = tf.placeholder(tf.float32, [None, generator.input_shape[1]], name=_z_name)

        self.G = generator(self.z)

        if custom_discriminator == None:
            D, D_logits, d_vars = _discriminator(param, self.images, self.is_training)
            D_, D_logits_, d_vars = _discriminator(param, self.G, self.is_training, reuse=True)
        else:
            D, D_logits, d_vars = custom_discriminator(self.images, self.is_training)
            D_, D_logits_, d_vars = custom_discriminator(self.G, self.is_training, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_vars = d_vars
        self.g_vars = generator.trainable_weights

        # print [var.name for var in self.d_vars]
        # for name in [var.name for var in self.g_vars]:
        #     print name

        self.d_optim = tf.train.AdamOptimizer(param.d_learning_rate, beta1=param.d_beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(param.g_learning_rate, beta1=param.g_beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver(max_to_keep=2)

class _Proxy:
    def __init__(self, sess, param, tf_var):
        self.sess = sess
        self.tf_var = tf_var
        self.param = param

    def create_sample_imgages(self, sample_z, sample_images):
        return self.sess.run(
            [self.tf_var.G, self.tf_var.d_loss, self.tf_var.g_loss],
            feed_dict={
                self.tf_var.z:sample_z,
                self.tf_var.images:sample_images,
                self.tf_var.is_training:False,
                K.learning_phase():0
            }
        )

    def rollback(self):
        # 2つバックアップを取っているので、古い方に戻す(インデックスで指定 : 0)
        # 戻り値はロールバックに使用したファイル名（ファイルが無い場合は None）
        return _load(self.tf_var, self.sess, self.param.checkpoint_dir, 0)

