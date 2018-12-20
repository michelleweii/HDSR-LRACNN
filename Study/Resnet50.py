from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

# 然后添加预训练的权重，keras已经训练好的模型
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# blick 1
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last': # 代表图像通道维度的位置
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'


    x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

# 有一次padding

    x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2c')(x)

    shortcut = Conv2D(filters3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,name=bn_name_base+'1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)

    return x



# block 2
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor  #输入变量#
        kernel_size: defualt 3, the kernel size of middle conv layer at main path #卷积核的大小#
        filters: list of integers, the filterss of 3 conv layer at main path  #卷积核的数目#
        stage: integer, current stage label, used for generating layer names #当前阶段的标签#
        block: 'a','b'..., current block label, used for generating layer names #当前块的标签#
    # Returns
        Output tensor for the block.  #返回块的输出变量#
    """
    filters1, filters2, filters3 = filters # 滤波器的名称
    if K.image_data_format() == 'channels_last': # 代表图像通道维度的位置
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'

    # 卷积层
    x = Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
    # BN层
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2a')(x)
    # 激活函数层
    x = Activation('relu')(x)

# 有一次pad
    x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=bn_name_base,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2c')(x)

    x = layers.add(x,input_tensor)
    x = Activation('relu')(x)

    return x

# 框架
def ResNet50(include_top=True, weights='imagenet', input_tensor=None,
             input_shape=None, pooling=None, classes=1000):  # 这里采用的权重是imagenet，可以更改，种类为1000
    # 这个include_top是什么？
    # 参数include_top表示是否包含模型底部的全连接层，
    # 如果包含，则可以将图像分为ImageNet中的1000类，如果不包含，则可以利用这些参数来做一些定制的事情。

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

# stage1
    x = ZeroPadding2D((3, 3))(img_input)  # 对图片界面填充0，保证特征图的大小
    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x) #定义卷积层
    x = BatchNormalization(axis=bn_axis,name='bn_conv1')(x) # 批标准化
    x = Activation('relu')(x) # 激活函数
    x = MaxPooling2D((3,3),strides=(2,2))(x) # 最大池化层
# stage2
    x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))   # 虚线
    x = identity_block(x,3,[64,64,256],stage=2,block='b') # 实线
    x = identity_block(x,3,[64,64,256],stage=2,block='c') # 实线
# stage3
    x = conv_block(x,3,[128,128,512],stage=3,block='a') # stride(2,2)
    x = identity_block(x,3,[128,128,512],stage=3,block='b')
    x = identity_block(x,3,[128,128,512],stage=3,block='c')
    x = identity_block(x,3,[128,128,512],stage=3,block='d')
# stage4
    x = conv_block(x,3,[256,256,1024],stage=4,block='a') # stride(2,2)
    x = identity_block(x,3,[256,256,1024],stage=4,block='b')
    x = identity_block(x,3,[256,256,1024],stage=4,block='c')
    x = identity_block(x,3,[256,256,1024],stage=4,block='d')
    x = identity_block(x,3,[256,256,1024],stage=4,block='e')
    x = identity_block(x,3,[256,256,1024],stage=4,block='f')
# stage5
    x = conv_block(x,3,[512,512,2048],stage=5,block='a') # stride(2,2)
    x = identity_block(x,3,[512,512,2048],stage=5,block='b')
    x = identity_block(x,3,[512,512,2048],stage=5,block='c')

    x = AveragePooling2D((7,7),name='avg_pool')(x)


    if include_top:
        x = Flatten()(x)
        x = Dense(classes,activation='softmax',name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs,x,name='resnet50')


    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

#
if __name__ == '__main__':
    model = ResNet50(include_top=True,weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    print('Input image shape:',x.shape)

    preds = model.predict(x)
    print('Predicted:',decode_predictions(preds))










