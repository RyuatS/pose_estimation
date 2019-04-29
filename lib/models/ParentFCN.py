import tensorflow as tf
import math
from collections import OrderedDict
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim

class ParentFCN:
    def __init__(self, is_use_bn=False):
        self.weights = {}
        self.biases = {}
        # batch normを使うか。　使う場合、True
        self.is_use_bn = is_use_bn

        if is_use_bn:
            self.bn_gamma = {}
            self.bn_beta = {}





    DEFAULT_CONV_STRIDES = [1, 1, 1, 1]
    DEFAULT_POOL_STRIDES = [1, 2, 2, 1]
    DEFAULT_ACTIVATION = 'relu'

    def conv2d(self, input,
               name,
               filter_shape,
               strides=None,
               padding='SAME',
               mean=0.0,
               sigma=1.0,
               activation='relu',
               dilations=[1, 1, 1, 1],
               is_training=False,
               reuse=False,
               keep_prob=1.):
        """
        perform convolution2d to input.

        Args:
            filter_shape: [kernel_height, kernel_width, input_channel, output_channle]
            strides: [batch, in_height, in_width, in_channle]
            padding: 'SAME' or 'VALID'
            mean:
            sigma:
            activation:
            dilations:
            is_training: whether training or not.
            reuse: whether batch normalization values use or not. if now you are training model, reuse should be False.


        Return:
            convolution2d output
        """

        if strides is None:
            strides = DEFAULT_CONV_STRIDES
        assert len(strides) == 4

        if dilations is None:
            dilations = [1, 1, 1, 1]


        with tf.variable_scope(name):
            self.weights[name] = self.get_weight(filter_shape, name, mean=mean, sigma=sigma)
            conv = tf.nn.conv2d(input, self.weights[name], strides, padding=padding, name=name)
            if self.is_use_bn:
                conv = tf.layers.batch_normalization(conv, training=is_training, reuse=reuse)
            self.biases[name] = self.get_bias([filter_shape[3]], name, mean=mean, sigma=sigma)
            conv = tf.nn.bias_add(conv, self.biases[name])
            if activation=='relu':
                out = tf.nn.relu(conv)
            elif activation=='leaky':
                out = tf.nn.leaky_relu(conv)
            elif activation=='sigmoid':
                out = tf.nn.sigmoid(conv)
            elif activation=='none':
                out = tf.nn.sigmoid(conv)
            else:
                raise ValueError('conv-activation {} is not defined'.format(activation))
            out = tf.nn.dropout(out, keep_prob)

        return out


    def deconv2d(self, input,
                 name,
                 filter_shape,
                 output_shape,
                 strides=None,
                 padding='SAME',
                 mean=0.0,
                 sigma=1.0,
                 activation='relu',
                 is_training=False,
                 reuse=False):
        """
        Args:
            input: input data [batch, height, width, in_channel].
            name: name_scope
            filter_shape: [kernel_height, kernel_width, out_channel, in_channel]. filter's in_channle must match the input's one.
            output_shape: [batch, height, width, in_channel].
            strides: [batch in_height, in_widht, in_channel]
            padding: 'SAME' or 'VALID'

        Return:
            deconvolution2d output
        """

        if strides is None:
            strides = DEFAULT_CONV_STRIDES
        assert len(strides) == 4


        with tf.variable_scope(name):
            input_shape = tf.shape(input)
            if output_shape[0] is None:
                output_shape[0] = input_shape[0]
            if output_shape[1] is None:
                output_shape[1] = tf.multiply(strides[1], input_shape[1])
            if output_shape[2] is None:
                output_shape[2] = tf.multiply(strides[2], input_shape[2])

            self.weights[name] = self.get_weight(filter_shape, name, mean=mean, sigma=sigma)
            deconv = tf.nn.conv2d_transpose(input, self.weights[name], output_shape=output_shape, strides=strides, padding=padding, name=name)
            if self.is_use_bn:
                deconv = tf.layers.batch_normalization(deconv, training=is_training, reuse=reuse)
            self.biases[name] = self.get_bias([filter_shape[2]], name, mean=mean, sigma=sigma)
            deconv = tf.nn.bias_add(deconv, self.biases[name])
            if activation == 'relu':
                out = tf.nn.relu(deconv)
            elif activation =='leaky':
                out = tf.nn.leaky_relu(deconv)

        return out


    def Dense(self,
              input,
              name,
              units,
              mean=0.0,
              sigma=1.0,
              activation='relu',
              is_training=False,
              reuse=False,
              keep_prob=1.):

        with tf.variable_scope(name):
            input_shape = input.get_shape().as_list()[1]
            self.weights[name] = self.get_weight([input_shape, units], name, mean=mean, sigma=sigma*0.01)
            self.biases[name] = self.get_bias([units], name, mean=mean, sigma=sigma*0.01)
            out = tf.matmul(input, self.weights[name]) + self.biases[name]
            if self.is_use_bn:
                out = tf.layers.batch_normalization(out, training=is_training, reuse=reuse)
            if activation == 'relu':
                out = tf.nn.relu(out)
            elif activation == 'sigmoid':
                out = tf.nn.sigmoid(out)
            elif activation == 'leaky':
                out = tf.nn.leaky_relu(out)
            elif activation == None:
                pass
            else:
                raise ValueError('{}, activation {} is not defined'.format(name, activation))
            out = tf.nn.dropout(out, keep_prob)
        return out



    def max_pool(self, input, name, strides=None):
        """
        perform max pooling to input.

        Args:
            input: input data.
            name: name scope
            strides: [batch, in_height, in_width, in_channel]

        Return:
            max_pool output
        """
        if strides is None:
            strides = self.DEFAULT_POOL_STRIDES
        assert len(strides) == 4

        with tf.variable_scope(name):
            out = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=strides, padding='SAME', name=name)

        return out


    def avg_pool(self, input, name, strides=None, ksize=[1, 2, 2, 1]):
        """
        perform average pooling to input.

        Args:
            input: input data
            name: name scope
            strides: [batch, in_height, in_width, in_channel]

        Return:
            average pooling output
        """
        if strides is None:
            strides = self.DEFAULT_POOL_STRIDES
        assert len(strides) == 4
        assert len(ksize) == 4

        return tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding='VALID', name=name)


    def get_weight(self, filter_shape, name, mean=0.0, sigma=0.01):
        """
        create variable weights.

        Args:
            filter_shape: [kernel_height, kernel_width, input_channel, output_channel]
            name: variable name.
            mean: gaussian distribution mean.
            sigma: gaussian distribution standard deviation.

        Return:
            variable weights.
        """
        weight = tf.truncated_normal(filter_shape, mean=mean, stddev=sigma, dtype=tf.float32)
        weight = tf.Variable(weight, name='w_' + name)

        return weight


    def get_bias(self, shape, name, mean=0.0, sigma=0.01):
        """
        create variable bias.

        Args:
            shape: 1-dimention
            name: variable name
            maen: gaussian distribution mean.
            sigma: gaussian distribution standard deviation.

        Return:
            variable biases.
        """
        bias = tf.truncated_normal(shape, mean=mean, stddev=sigma, dtype=tf.float32)
        bias = tf.Variable(bias, name='b_' + name)

        return bias


    def perform_conv(input, key, is_training):
        """
        perform convolution to input.

        1) get convolution parameter(dilation, activation , keep_prob, strides, filter_shape)
        2) call conv2d function with parameter.

        Args:
            input: input tensor. [batch, height, width, channel]
            key: self.model_structure's key.
        Returns:
            output performed convolution.
        """

        filter_shape = self.model_structure[key]['filter_shape']
        strides = self.model_structure[key]['strides']
        if 'dilations' in self.model_structure[key].keys():
            dilation = self.model_structure[key]['dilations']
        else:
            dilation = None

        if 'activation' in self.model_structure[key].keys():
            activation = self.model_structure[key]['activation']
        else:
            activation = DEFAULT_ACTIVATION

        if 'dropout' in self.model_structure[key].keys():
            keep_prob = self.model_structure[key]['dropout']
        else:
            keep_prob = 1.0

        # ノードの数を使って重みを初期値を計算する (Heの初期値)
        num_node = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]
        sigma = 1 / math.sqrt(num_node)

        # convolution
        output = self.conv2d(input, name=key,
                             filter_shape=filter_shape,
                             strides=strides,
                             sigma=sigma,
                             activation=activation,
                             dilations=dilations,
                             is_training=is_training,
                             keep_prob=keep_prob)
        return output


    def perform_deconv(input, key, is_training):
        """
        perform deconvolution to input.

        1) get deconvolution parameter(dilation, activation , keep_prob, strides, filter_shape)
        2) call deconv2d function with parameter.

        Args:
            input: input tensor. [batch, height, width, channel]
            key: self.model_structure's key.
            is_training: whether now training or not.
        Returns:
            output performed convolution.
        """

        # get deconvolution parameter.
        filter_shape = self.model_structure[key]['filter_shape']
        strides = self.model_structure[key]['strides']
        output_shape = self.model_structure[key]['output_shape']
        if 'activation' in self.model_structure[key].keys():
            activation = self.model_structure[key]['activation']
        else:
            activation=DEFAULT_ACTIVATION
        # if output_shape[0] is None:
        #     output_shape[0] = batch_size
        # if output_shape[1] is None:
        #     output_shape[1] = strides[1] * pre_layer.get_shape()[1]
        # if output_shape[2] is None:
        #     output_shape[2] = strides[2] * pre_layer.get_shape()[2]


        # ノードの数を使って重みの初期値を計算する (Heの初期値)
        num_node = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]
        sigma = 1 / math.sqrt(num_node)
        # Deconvolution
        output = self.deconv2d(input, name=key,
                               filter_shape=filter_shape,
                               output_shape=output_shape,
                               strides=strides,
                               sigma=sigma,
                               activation='leaky',
                               is_training=is_training)

        return output


    def build(self, input_img, name, is_training, reuse=False, visualize=False, batch_norm_reuse=False):
        """
        Args:
            input_img: rgb image batch.
            is_training: トレーニング中かどうか
            batch_norm_reuse: 他の計算グラフに展開中のbatch_normalizationの値を使うか.
                              これは、トレーニング中のグラフでのbatch_normalizationの値が使われていて、
                              それとは他に同じ値を使いたい時に用いる。

        Returns:
            predict: 最後の層の予測
        """
        # batch_size = tf.shape(input_img)[0]


        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            self.layers = OrderedDict()

            # if len(input_img.get_shape().as_list()) > 2:
            #     mean, var = tf.nn.moments(input_img, axes=[1, 2])
            #     self.input_img = (input_img - mean) / tf.sqrt(var)
            # else:
            self.input_img = input_img
            # 一つ前の層のデータ
            self.layers['input'] = self.input_img
            pre_layer = self.input_img
            num_node = 4*4*512

            # self.model_structureで層を判断し、処理を行う
            for key in self.model_structure.keys():
                # model_structureのキー名で処理を決める。
                # 'conv'   => perform 'convolutional'
                # 'deconv' => perform 'deconvolutional'
                # 'Dense'  => perform 'fully connected'
                # 'pool'   => perform 'pooling'
                # 'gray'   => perform 'gray'
                # 'GAP'    => perform 'global average pooling'
                # 'Flatten'=> perform 'flatten'
                # 'reshape'=> perform 'reshape'
                # else => raise Error

                # print('pre_layer.shape: {}, key: {}'.format(pre_layer.get_shape(), key))
                if key.startswith('conv'):
                    self.layers[key] = self.perform_conv(pre_layer, key, is_training)

                elif key.startswith('deconv'):
                    self.layers[key] = self.perform_deconv(pre_layer, key, is_training)

                elif key.startswith('Dense'):
                    units = self.model_structure[key]['units']
                    if 'activation' in self.model_structure[key].keys():
                        activation = self.model_structure[key]['activation']
                    else:
                        activation = 'relu'
                    if 'dropout' in self.model_structure[key].keys():
                        keep_prob = self.model_structure[key]['dropout']
                    else:
                        keep_prob = 1.0
                    self.layers[key] = self.Dense(pre_layer,
                                                  key, units,
                                                  activation=activation,
                                                  keep_prob=keep_prob,
                                                  is_training=is_training)

                elif key.startswith('pool'):
                    strides = self.model_structure[key]['strides']
                    self.layers[key] = self.avg_pool(pre_layer, name=key, strides=strides)

                elif key.startswith('gray'):
                    if pre_layer.get_shape()[3] == 6:
                        image1, image2 = tf.split(pre_layer, [3, 3], axis=3)
                        gray1 = tf.image.rgb_to_grayscale(image1)
                        gray2 = tf.image.rgb_to_grayscale(image2)
                        self.layers[key] = tf.concat([gray1, gray2], axis=3)
                    else:
                        self.layers[key] = tf.image.rgb_to_grayscale(pre_layer)

                elif key.startswith('GAP'):
                    b, h, w, c = pre_layer.get_shape().as_list()
                    ksize = [1, h, w, 1]
                    gap = self.avg_pool(pre_layer, key, ksize=ksize)
                    self.layers[key] = tf.reshape(gap, [-1, c])

                elif key.startswith('Flatten'):
                    self.layers[key] = tf.layers.Flatten()(pre_layer)

                elif key.startswith('reshape'):
                    shape = self.model_structure[key]['shape']
                    self.layers[key] = tf.reshape(pre_layer, [-1] + shape)

                elif key.startswith('backbone'):
                    net_name = self.model_structure[key]['net']

                    with tf.name_scope(key):
                        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                            _, end_points = resnet_v1.resnet_v1_101(pre_layer, 1000, is_training=is_training)
                        conv_out = end_points[name + "/resnet_v1_101/block3"]
                    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=key)
                    saver1 = tf.train.Saver(var_list)

                    pre_layer[key] = conv_out


                else:
                    raise ValueError("model structure's key is not correct")

                # skipレイヤーがあれば、そのレイヤーからの値をもってきて、合体させる
                if 'concat_from' in self.model_structure[key].keys():
                    # どこのレイヤーから持ってくるか
                    layer_name = self.model_structure[key]['concat_from']
                    layer_values = self.layers[layer_name]
                    self.layers[key] = tf.concat([self.layers[key], layer_values], axis=3)

                # residualレイヤーがあれば、そのレイヤーからの値を持ってきて、足し合わせる
                if 'residual' in self.model_structure[key].keys():
                    residual_layer = self.model_structure[key]['residual']
                    residual_layer_values = self.layers[residual_layer]
                    self.layers[key] = tf.add(self.layers[key], residual_layer_values)

                pre_layer = self.layers[key]


            for key in self.weights.keys():
                tf.summary.histogram('histogram-' + key, self.weights[key])


        if visualize:
            print('-' * 40)
            print(' {} '.format(self.model_type))
            print('-' * 40)
            for key in self.layers.keys():
                print('{}\t{}'.format(key, self.layers[key].get_shape()))
            print('-' * 40)

        return pre_layer


    def loss(self, predict, target, use_l2_loss=True, decay_rate=0.005):

        self.softmax = tf.nn.softmax(predict, axis=2)
        self.cross_entropy = -tf.reduce_mean(tf.multiply(tf.log(self.softmax), target))

        loss = self.cross_entropy
        if use_l2_loss:
            sum_L2_loss = 0
            for key in self.weights.keys():
                L2 = tf.nn.l2_loss(self.weights[key], name=key+'l2_loss')
                sum_L2_loss += L2

            loss += decay_rate * sum_L2_loss

        return self.cross_entropy, loss


    def train(self, input, target, is_training, deconv_learning_rate=0.001, conv_learning_rate=0.1, decay_rate=0.005, visualize=False):
        """
        学習させるためのオペレーションを返す。

        Args:
            input: 入力画像. [batch_size, height, width, channel]
            target: 教師ラベル. [batch_size, height, width, num_classes]
            deconv_learning_rate: 逆畳み込み層のためのラーニングレート
            conv_learning_rate: 畳み込み層のためのラーニングレート
            decay_rate: 荷重減衰
            visualize: モデルアーキテクチャを表示するか

        Return:
            predict: 予測した結果
            cross_entropy: ピクセルごとのクロスエントロピー誤差の平均
            train_op: 学習するためのオペレーション
            grads_and_vars: 変数の名前とそれに対応する勾配
            extra_update_ops:　batch_normalizationを使う場合にbeta,gammaを更新するためのもの
        """

        predict = self.build(input, visualize=visualize, is_training=is_training)
        with tf.name_scope('var'):
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name, var)
        with tf.name_scope('loss'):
            cross_entropy, loss = self.loss(predict, target, decay_rate=decay_rate)
        with tf.name_scope('train'):

            deconv_optimizer = tf.train.GradientDescentOptimizer(learning_rate=deconv_learning_rate)
            conv_optimizer = tf.train.GradientDescentOptimizer(learning_rate=conv_learning_rate)

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            conv_list = [var for var in variables if var.name.startswith('conv')]
            deconv_list = [var for var in variables if var.name.startswith('deconv')]

            # take the each weight Gradient
            conv_grads_and_vars = conv_optimizer.compute_gradients(loss, conv_list)
            deconv_grads_and_vars = deconv_optimizer.compute_gradients(loss, deconv_list)

            for grad, var in conv_grads_and_vars:
                tf.summary.histogram(var.name + '-grad', grad)
            for grad, var in deconv_grads_and_vars:
                tf.summary.histogram(var.name + '-grad', grad)

            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            #
            # for grad, var in grads_and_vars:
            #     tf.summary.histogram(var.name + '-grad', grad)

            if self.is_use_bn:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    conv_train_op = conv_optimizer.apply_gradients(conv_grads_and_vars, name='apply_gradients_to_conv')
                    deconv_train_op = deconv_optimizer.apply_gradients(deconv_grads_and_vars, name='apply_gradients_to_deconv')
                    train_op = tf.group(conv_train_op, deconv_train_op)
                    grads_and_vars = tf.group(conv_grads_and_vars, deconv_grads_and_vars)
                return predict, cross_entropy, train_op, grads_and_vars, extra_update_ops
            else:
                conv_train_op = conv_optimizer.apply_gradients(conv_grads_and_vars, name='apply_gradients_to_conv')
                deconv_train_op = deconv_optimizer.apply_gradients(deconv_grads_and_vars, name='apply_gradients_to_deconv')
                train_op = tf.group(conv_train_op, deconv_train_op)
                grads_and_vars = tf.group(conv_grads_and_vars, deconv_grads_and_vars)
                return predict, cross_entropy, train_op, grads_and_vars, tf.constant(0)


    def predict(self, input, is_training, visualize=False, batch_norm_reuse=False):
        predicts = self.build(input,
                              is_training,
                              visualize=visualize,
                              )
        return predicts


    def evaluate(self, input, targets, num_classes):

        predicts = self.build(input, visualize=True, is_training=tf.constant(False))
        predicts_softmax = tf.nn.softmax(predicts, axis=2)
        cross_entropy = -tf.multiply(tf.log(predicts_softmax), targets)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=2)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)


        labels = tf.argmax(predicts, axis=3)
        targets = tf.argmax(targets, axis=3)
        # ピクセルごとに正しいかを比較 (正しければ 1)
        pixel_wise_equal = tf.equal(labels, targets)

        # クラスごとの正解数
        each_class_IoU = []
        for i in range(num_classes):
            # true positiveを計算
            # predictに対してクラスiのピクセルを取り出す。
            class_pixel_in_labels = tf.equal(labels, i)
            # targetに対してクラスiのピクセルを取り出す。
            class_pixel_in_targets = tf.equal(targets, i)
            # 上記二つに対してAND演算
            casted_labels = tf.cast(class_pixel_in_labels, tf.int32)
            casted_targets = tf.cast(class_pixel_in_targets, tf.int32)
            true_positive = tf.reduce_sum(tf.multiply(casted_labels, casted_targets))

            # true negativeを計算
            # predictに対してクラスiじゃないピクセルを取り出す。
            not_class_pixel_in_labels = tf.logical_not(class_pixel_in_labels)
            # targetに対してクラスiじゃないピクセルを取り出す。
            not_class_pixel_in_targets = tf.logical_not(class_pixel_in_targets)
            # この2つのAND演算によって、true negative(モデルが正しくないと判断したピクセルと実際に正しくないピクセルの一致の場所を計算)
            casted_not_labels = tf.cast(not_class_pixel_in_labels, tf.int32)
            casted_not_targets = tf.cast(not_class_pixel_in_targets, tf.int32)
            true_negative = tf.reduce_sum(tf.multiply(casted_not_labels, casted_not_targets))

            # false negativeを計算(クラスのピクセルが実在するのに、実在しないと判断)
            # predictに対してクラスiじゃないピクセルを取り出す。 => not_class_pixel_in_labels
            # targetsに対してクラスiのピクセル => class_pixel_in_targets
            # not_class_pixel_in_labelsとclass_pixel_in_targetsのAND演算
            false_negative = tf.reduce_sum(tf.multiply(casted_targets, casted_not_labels))

            # false_positive
            false_positive = tf.reduce_sum(tf.multiply(casted_not_targets, casted_labels))

            # IoU値を計算
            IoU = true_positive / (true_positive + true_negative + false_negative)
            each_class_IoU.append(IoU)




        # 正しいピクセルの数の合計
        num_correct_pixel = tf.reduce_sum(tf.cast(pixel_wise_equal, tf.int32), axis=0)
        # 正答率
        pixel_wise_accuracy = tf.reduce_mean(tf.cast(pixel_wise_equal, tf.float32))


        return predicts, cross_entropy, num_correct_pixel, pixel_wise_accuracy, each_class_IoU