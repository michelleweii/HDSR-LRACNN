# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed,Conv2DTranspose
from keras.layers.convolutional import UpSampling2D, Conv2D
import numpy as np
from keras import backend as K
import tensorflow as tf

from keras_frcnn.roi_pooling_conv import RoiPoolingConv
from keras_frcnn.fixed_batch_normalization import FixedBatchNormalization


def get_weight_path():
    if K.image_dim_ordering() == 'th':
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        # input_length += 6
        # # apply 4 strided convolutions
        # filter_sizes = [7, 3, 1, 1]
        # stride = 2
        # for filter_size in filter_sizes:
        #     input_length = (input_length - filter_size + stride) // stride
        # return input_length
        return input_length // 2 # add by me
    print("feature_map_width:{}, height:{}".format(get_output_length(width), get_output_length(height)))
    return get_output_length(width), get_output_length(height)


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                      padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(
        input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(
        Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                                      kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c',
                        trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(
        Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # 自定义输入，为了显示
    # img_numpy = np.full((1,38,166,3),2)
    # x = tf.convert_to_tensor(img_numpy,dtype=tf.float32)
    x = ZeroPadding2D((3, 3))(img_input)

    # stage1
    # conv1 7*7,64,stride 2
    conv1 = Convolution2D(64, (6, 6), strides=(2, 2), name='conv1', trainable=trainable)(x)
    bn_conv1 = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(conv1)
    conv1_relu = Activation('relu')(bn_conv1)
    conv1_pool = MaxPooling2D((3, 3), strides=(2, 2))(conv1_relu)
    print("stage1:{}".format(conv1_pool.get_shape())) # (1, 8, 40, 64)
    # print("stage1:{}".format(x.shape))
    # stage2
    # conv2_x [ [1*1,64], [3*3,64], [1*1,256] ]*3
    conv2_a = conv_block(conv1_pool, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    conv2_b = identity_block(conv2_a, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    conv2_c = identity_block(conv2_b, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)
    # 最后相加的
    x2 = UpSampling2D((2, 2))(conv2_c)
    # conv2_output (1, 16, 80, 1024)
    x2_output = Conv2D(filters=1024, kernel_size=3, strides=(1,1), padding='same')(x2)
    # x2_output = conv_block(x2, 3, [64, 64, 1024], stage=2, block='diy', strides=(1, 1), trainable=trainable)

    # tmp_conv2_1024 = identity_block(conv2_b, 3, [128, 128, 1024], stage=2, block='c', trainable=trainable)
    # ValueError: Operands could not be broadcast together with shapes (8, 40, 1024) (8, 40, 256) 是因为shortcut要同一尺度相加
    # x2 = UpSampling2D((2, 2))(conv2_c)
    print("stage2:{}".format(x2.get_shape()))
    print("x2_output:{}".format(x2_output.shape))
    # stage3
    # conv3_x [ [1*1,128], [3*3,128], [1*1,512] ]*4
    conv3_a = conv_block(x2, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    conv3_b = identity_block(conv3_a, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    conv3_c = identity_block(conv3_b, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    conv3_d = identity_block(conv3_c, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)
    x3 = UpSampling2D((2, 2))(conv3_d)
    print("stage3:{}".format(x3.get_shape()))
    # stage4
    # conv4_x [ [1*1,256], [3*3,256], [1*1,1024] ]*6
    conv4_a = conv_block(x3, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    conv4_b = identity_block(conv4_a, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    conv4_c = identity_block(conv4_b, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    conv4_d = identity_block(conv4_c, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    conv4_e = identity_block(conv4_d, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    conv4_f = identity_block(conv4_e, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)
    x4 = UpSampling2D((2, 2))(conv4_f)
    x = x4 + x2_output

    print("stage4:{}".format(x4.get_shape()))
    print("return x_shape:{}".format(x.shape))
    return x




def classifier_layers(x, input_shape, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    print("a")
    if K.backend() == 'tensorflow':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                          trainable=trainable)
    elif K.backend() == 'theano':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(1, 1),
                          trainable=trainable)

    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    print("classifier_layers:{}".format(x.shape))
    return x


def rpn(base_layers, num_anchors):
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)
    print("rpn_class:{}".format(x_class.shape))
    print("rpn_regr:{}".format(x_regr.shape))
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=11, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 1024, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    print("out_roi_pool{}".format(out_roi_pool))
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
