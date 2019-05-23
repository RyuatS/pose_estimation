# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-05-15T03:00:51.267Z
# Description:
#
# ===============================================

# lib
import tensorflow as tf
import math
from collections import OrderedDict
from tensorflow.contrib.slim.nets import resnet_v1, vgg
from tensorflow.contrib import slim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'models', 'research', 'slim'))

# user packages
from core.config import BACKBONE_NAME_LIST, R_MEAN, G_MEAN, B_MEAN

class ParentFCN:
    def __init__(self, is_use_bn=False):
        self.weights = {}
        self.biases = {}
        self.num_params_each_layer = {}
        # batch normを使うか。　使う場合、True
        self.is_use_bn = is_use_bn

        if is_use_bn:
            self.bn_gamma = {}
            self.bn_beta = {}

        self.layer_funcs = {
            'conv'     : self.perform_conv,
            'deconv'   : self.perform_deconv,
            'backbone' : self.perform_backbone,
            'pool'     : self.perform_pool,
            'gray'     : self.convert_gray,
            'Flatten'  : self.flatten,
            'reshape'  : self.reshape,
            'concat'   : self.concat,
            'residual' : self.residual
        }





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
            elif activation=='no':
                out = conv
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
            elif activation == 'no':
                out = deconv

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
            elif activation == 'no':
                pass
            else:
                raise ValueError('{}, activation {} is not defined'.format(name, activation))
            out = tf.nn.dropout(out, keep_prob)
        return out


    def max_pool(self, input, name, strides=None, ksize=[1, 2, 2, 1]):
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
            out = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME', name=name)

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


    def get_activation(self, key):
        """
        get activation function name.

        Args:
            key: self.model_structure's key.
        Return:
            activation function name.
        """
        if 'activation' in self.model_structure[key]['activation']:
            activation = self.model_structure[key]['activation']
        else:
            activation = self.DEFAULT_ACTIVATION

        return activation


    def perform_conv(self, input, key, param_dict, is_training):
        """
        perform convolution to input.

        1) get convolution parameter(dilation, activation , keep_prob, strides, filter_shape)
        2) call conv2d function with parameter.

        Args:
            input       : input tensor. [batch, height, width, channel]
            key         : key.
            param_dict  : parameter_dictionary.
            is_training : whether train or not.
        Returns:
            output performed convolution.
        """

        filter_shape = param_dict['filter_shape']
        strides = param_dict['strides']
        if 'dilations' in param_dict.keys():
            dilations = param_dict['dilations']
        else:
            dilations = None

        if 'activation' in param_dict.keys():
            activation = param_dict['activation']
        else:
            activation = ParentFCN.DEFAULT_ACTIVATION

        if 'dropout' in param_dict.keys():
            keep_prob = param_dict['dropout']
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

        # calculate number of parameters
        num_params = 0
        num_params = (self.weights[key].get_shape().as_list()[0] *
                     self.weights[key].get_shape().as_list()[1] *
                     self.weights[key].get_shape().as_list()[2] *
                     self.weights[key].get_shape().as_list()[3])
        num_params += self.biases[key].get_shape().as_list()[0]
        self.num_params_each_layer[key] = num_params

        return output


    def perform_deconv(self, input, key, param_dict, is_training):
        """
        perform deconvolution to input.

        1) get deconvolution parameter(dilation, activation , keep_prob, strides, filter_shape)
        2) call deconv2d function with parameter.

        Args:
            input       : input tensor. [batch, height, width, channel]
            param_dict  : parameter_dictionary.
            is_training : whether now training or not.
        Returns:
            output performed deconvolution.
        """

        # get deconvolution parameter.
        filter_shape = param_dict['filter_shape']
        strides = param_dict['strides']
        output_shape = param_dict['output_shape']
        if 'activation' in param_dict.keys():
            activation = param_dict['activation']
        else:
            activation = ParentFCN.DEFAULT_ACTIVATION
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

        # calculate number of parameters
        num_params = 0
        num_params = (self.weights[key].get_shape().as_list()[0] *
                     self.weights[key].get_shape().as_list()[1] *
                     self.weights[key].get_shape().as_list()[2] *
                     self.weights[key].get_shape().as_list()[3])
        num_params += self.biases[key].get_shape().as_list()[0]
        self.num_params_each_layer[key] = num_params

        return output


    def perform_backbone(self, input, key, param_dict, is_training):
        """
        build backbone and input data to backbone net.
        This function does following steps.
        1) take layer information (backbone name and output stride) from self.model_structure[key].
        2) build backbone net.

        Args:
            input       : pre_layer's data.
            param_dict  : parameter_dictionary. for taking layer information.
            is_training : whether train or not you build this model.
        Reuturn:
            backbone's output.
        """

        # 1. take layer information
        net_name = param_dict['net']
        if 'output_stride' in param_dict.keys():
            output_stride = param_dict['output_stride']
        else:
            output_stride = 16

        # preprocess to image for backbone.
        mean_filter = tf.constant([R_MEAN, G_MEAN, B_MEAN], dtype=tf.float32)
        mean_filter = tf.reshape(mean_filter, [1, 1, 1, 3])
        input = input - mean_filter

        # 2. build backbone net.
        if net_name in BACKBONE_NAME_LIST:
            if net_name == 'resnet_v1_101':
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    net, end_points = resnet_v1.resnet_v1_101(input,
                                                     num_classes=None,
                                                     is_training=is_training,
                                                     global_pool=False,
                                                     output_stride=output_stride)
            elif net_name == 'resnet_v1_50':
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    net, end_points = resnet_v1.resnet_v1_50(input,
                                                    num_classes=None,
                                                    is_training=is_training,
                                                    global_pool=False,
                                                    output_stride=output_stride)
        else:
            raise ValueError("backbone '{}' is not implemented".format(net_name))

        self.backbone_end_points = end_points
        ##### write backbone layer's shape
        # with open('%s_layers.txt' % net_name, 'w') as f:
        #     for k in end_points:
        #         write_content = '{}: {}\n'.format(k, end_points[k].get_shape())
        #         f.write(write_content)

        # calculate number of parameters.
        var_list = tf.contrib.framework.get_variables_to_restore(include=[net_name])
        num_param = 0
        for var in var_list:
            shape = var.get_shape().as_list()
            param = 1
            for dim in shape:
                param *= dim
            num_param += param

        self.num_params_each_layer[key] = num_param

        return net


    def perform_dense(self, input, key, param_dict, is_training):
        """
        perform fully-connected layer to input.

        1) get dence parameter(number of units(nodes), keep_prob and activation)
        2) call dense function with parameter.
        3) calculate the number of parameters.

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter_dictionary.
            is_training : whether now training or not.
        Returns:
            output performed dense.
        """
        units = param_dict['units']

        activation = self.get_activation(key)
        if 'dropout' in param_dict.keys():
            keep_prob = param_dict['dropout']
        else:
            keep_prob = 1.0

        self.num_params_each_layer[key] = units * input.get_shape().as_list()[1]

        output = self.Dense(input, key, units,
                            activation=activation,
                            keep_prob=keep_prob,
                            is_training=is_training)

        return output


    def perform_pool(self, input, key, param_dict, is_training):
        """
        perform pooling to input.

        1) get pool information (max pool or avg pool, strides).
        2) call pooling function with stride.

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter_dictionary.
            is_training : whether now training or not.
        Returns:
            output performed pooling.
        """
        strides   = param_dict['strides']
        pool_type = param_dict['pool_type']

        if 'kernel' in param_dict.keys():
            ksize = param_dict['kernel']
        else:
            ksize = self.DEFAULT_POOL_STRIDES

        if pool_type == 'average':
            output = self.avg_pool(input, name=key, strides=strides, ksize=ksize)
        elif pool_type == 'max':
            output = self.max_pool(input, name=key, strides=strides, ksize=ksize)
        else:
            raise ValueError("pool type '{}' is not defined".format(pool_type))
        self.num_params_each_layer[key] = 0

        return output


    def convert_gray(self, input, key, param_dict, is_training):
        """
        convert image to grayscale.

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter_dictionary.
            is_training : Dummy variable. This is not used in this function.
        Returns:
            output converted to grayscale.
        """
        # グレイスケールに変換するときの画像がステレオ場合は、それぞれを変換する
        if input.get_shape()[3] == 6:
            image1, image2 = tf.split(input, [3, 3], axis=3)
            gray1 = tf.image.rgb_to_grayscale(image1)
            gray2 = tf.image.rgb_to_grayscale(image2)
            output = tf.concat([gray1, gray2], axis=3)
        else:
            output = tf.image.rgb_to_grayscale(input)

        self.num_params_each_layer[key] = 0

        return output


    def flatten(self, input, key, param_dict, is_training):
        """
        do flatten to input.

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter dictionary.
            is_training : Dummy variable. This is not used in this function.
        Returns:
            output done flatten.
        """

        output = tf.layers.Flatten()(input)

        return output


    def reshape(self, input, key, param_dict, is_training):
        """
        reshape to input.

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter_dictionary.
            is_training : Dummy variable. This is not used in this function.
        Returns:
            reshaped output.
        """
        shape = param_dict['shape']
        output = tf.reshape(input, [-1] + shape)

        return output


    def concat(self, input, key, param_dict, is_training):
        """
        concatenate input and param_dict['concat_from']

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter dictionary.
            is_training : Dummy variable. This is not used in this function.
        Returns:
            concatenated output .
        """
        layer_name = param_dict['concat_from']

        if layer_name.startswith('backbone'):
            # バックボーンの途中から取り出したい時の処理
            # example) layer_name = 'backbone/resnet_v1_50/block2'
            # 'backbone'のあとのレイヤーネームをとりだす。 example) 'backbone/resnet_v1_50/block2' => 'resnet_v1_50/block2'
            layer_name = layer_name.split('/')[1:]
            layer_name = '/'.join(layer_name)
            layer_values = self.backbone_end_points[layer_name]
        else:
            layer_values = self.layers[layer_name]

        output = tf.concat([input, layer_values], axis=3)
        return output


    def residual(self, input, key, param_dict, is_training):
        """
        add input and param_dict['residual']

        Args:
            input       : input tensor. [batch, nodes]
            param_dict  : parameter dict.
            is_training : Dummy variable. This is not used in this function.
        Returns:
            added output.
        """
        residual_layer_name = param_dict['residual']
        if residual_layer_name.startswith('backbone'):
            # example) residual_layer_name = 'backbone/resnet_v1_50/block2'
            residual_layer_name = residual_layer_name.split('/')[1:]
            residual_layer_name = '/'.join(residual_layer_name)
            residual_layer_values = self.backbone_end_points[residual_layer_name]
        else:
            residual_layer_values = self.layers[residual_layer_name]
        output = tf.add(input, residual_layer_values)
        return output


    def build(self, input_img, name, is_training, reuse=False, visualize=False, batch_norm_reuse=False):
        """
        Args:
            input_img        : rgb image batch.
            is_training      : トレーニング中かどうか
            reuse            : 同じ重みを使うか
            visualize        : 層を可視化するか
            batch_norm_reuse : 他の計算グラフの展開中のbatch_normalizationの値を使うか。
                               これは、トレーニング中のグラフのbatch_normalizationの値が使われていて、
                               それとは他に同じ値を使いたい時に用いる。
        """

        with tf.name_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # 層を通したあとのテンソルを保持する
            self.layers = OrderedDict()

            self.layers['input'] = input_img
            self.num_params_each_layer['input'] = 0
            pre_layer = input_img

            for key in self.model_structure.keys():

                # model_structureのキー名で処理を決める
                # layer_funcsにそれぞれの層のキー名と関数を登録する。
                for func_name in self.layer_funcs.keys():
                    if key.startswith(func_name):
                        layer_func = self.layer_funcs[func_name]
                        break

                if 'input' in self.model_structure[key].keys():
                    input_layer_name = self.model_structure[key]['input']
                    pre_layer = self.layers[input_layer_name]
                    
                self.layers[key] = layer_func(pre_layer, key, self.model_structure[key], is_training=is_training)

                pre_layer = self.layers[key]

                if key not in self.num_params_each_layer.keys():
                    self.num_params_each_layer[key] = 0



            if visualize:
                self.visualize_layers()

            return pre_layer


    def visualize_layers(self):
        """
        visualize layers information.
        layer name, layer output shape and number of parameters.

        """
        # それぞれの文字列の長さを最大値を求めるのは、表示するときに最大値を基準に表示するからである。
        #　最大値に合わせないと、ズレが起きる.

        # self.layersのkeysの文字列としての長さで一番長い値を保持する.
        key_max_length = 0
        # self.layersのそれぞれのshapeのリストの文字列で表した時の長さを表す。
        # 例えば, shape = '[ None, 1024, 1024, 1024]' なら、25である。
        # また、今回は、shapeのそれぞれの値は、4桁と仮定しているため、カンマが3つ、カッコが2つ,それぞれのshapeの長さが5つで shapeは4次元.
        # よって、 3 + 2 + (5 * 4)で文字列として25を保持する。
        shape_max_str_length = 25
        # パラメータの桁数で最も長いものを保持する。
        param_max_str_length = 0

        # それぞれの文字列フィールドの最大値を計算する。
        # 実際には、layer名の長さの最大値とパラメータ数の桁数の最大値を計算する。
        for key in self.layers.keys():
            if key_max_length < len(key):
                key_max_length = len(key)
            if param_max_str_length < len(str(self.num_params_each_layer[key])):
                param_max_str_length = len(str(self.num_params_each_layer[key]))

        # 文字列を表示する最大値を求める。
        # 実際には、以下のようになっている(_はスペースを表す)
        # _<layer_name>_:_<output_shape>____<num_params>_
        # これの文字列としての長さを表す。
        disp_str_len = (key_max_length +
                        shape_max_str_length +
                        param_max_str_length + 9)
        print('-' * disp_str_len)
        print('{name:^{len}}'.format(len=disp_str_len,
                                     name=self._model_type))
        print('-' * disp_str_len)
        print(' {key_name:^{key_len}} : {shape:^{shape_len}}    {param:^{param_len}} '.format(
            key_name='layer',
            key_len=key_max_length,
            shape='output shape',
            shape_len=shape_max_str_length,
            param='params',
            param_len=param_max_str_length
        ))
        print('-' * disp_str_len)
        for key in self.layers.keys():
            print(' {key_name:^{key_len}} : '.format(key_name=key, key_len=key_max_length), end='')
            shape_list = ['None' if shape is None else str(shape) for shape in self.layers[key].get_shape().as_list()]
            print('[{:>5},{:>5},{:>5},{:>5}]'.format(*shape_list), end='')
            print('    {param:{param_len}} '.format(param=self.num_params_each_layer[key], param_len=param_max_str_length))

        print('-' * disp_str_len)


    def loss(self, predict, target, use_weight_decay=True, decay_rate=0.005):
        """
        calculate loss against predict and target.

        Args:
            predict: predict heatmap Tensor. format=> [batch, height, width, keypoints_num]
            target: target heatmap Tensor. format is same as predict.
            use_weight_decay: boolean. whether you use weight decay.
            decay_rate: weights decay rate.
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=predict)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        loss = tf.reduce_mean(loss)


        # self.softmax = tf.nn.softmax(predict, axis=2)
        # self.cross_entropy = -tf.reduce_mean(tf.multiply(tf.log(self.softmax), target))
        loss_with_weight_decay = loss
        if use_weight_decay:
            weight_decay_sum = 0
            for key in self.weights.keys():
                weight_decay = tf.nn.l2_loss(self.weights[key], name=key+'l2_loss')
                weight_decay_sum += weight_decay

            loss_with_weight_decay += decay_rate * weight_decay_sum

        return loss, loss_with_weight_decay


    def get_train_op(self, predict, target, scope, learning_rate=0.001, use_weight_decay=True, decay_rate=0.005,  is_training_backbone=False):
        """
        return train operater.
        This function do following steps.
        1) Calculate loss. used argument => predict, target, use_weight_decay nad decay_rate.
        2) Create optimizer and train operater. used argument => learning_rate, is_training_backbone.
        3) Return train operater.

        Args:
            predict: 4-dimention tensor. Model's predict.
            target: 4-dimention tensor. Target.
            learning_rate: float. Learning rate for optimizer.
            use_weight_decay: boolean. Whether you use weight decay when training.
            decay_rate: float. Decay rate. If you use weight decay, this is used.
            is_training_backbone: boolean. Whether you train backbone net.

        Returns:
            loss without weight decay and train operater.
        """

        # 1) calculate loss
        loss, loss_with_weight_decay = self.loss(predict, target, use_weight_decay, decay_rate)
        loss_with_weight_decay /= target.get_shape().as_list()[3]
        loss /= target.get_shape().as_list()[3]

        # create trainable variable list
        trainable = tf.trainable_variables()
        # train_var_list = [var for var in trainable if 'resnet_v1_101' not in var.name]
        # for var in train_var_list:
        #     print(var)
        # train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        for var in trainable:
            tf.summary.histogram(var.name, var)
        # 2) create optimizer and train op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_with_weight_decay, var_list=trainable)

        # 3) return
        return loss, train_op
