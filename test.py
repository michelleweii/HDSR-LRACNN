from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import argparse
import os
import keras_frcnn.resnet as nn
# vgg
# import keras_frcnn.vgg as nn
from keras_frcnn.visualize import draw_boxes_and_label_on_image_cv2

# 参数设置与网络构建
# RPN网络预测与边框识别
# 分类网络边框分类与回归
def format_img_size(img, cfg):
    # 对图片每一个通道的像素值做规整
    """ formats the image size based on config """
    # 首先从配置文件夹中得到最小边的大小
    img_min_side = float(cfg.im_size)
    # 得到图片的高度和宽度
    (height, width, _) = img.shape

    # 根据高度和宽度谁大谁小，确定规整后图片的大小。
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    # 将图片放缩到指定的大小，用的是线性插值。
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # 返回缩放后的图片img和相应的缩放比例。
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    # 将图片的rgb变为bgr，因为网上训练好的vgg图片都是以此训练的。
    img = img[:, :, (2, 1, 0)]
    # 将图片数据类型转换为np.float32
    img = img.astype(np.float32)
    # 并减去每个通道的均值，理由同上
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    # 图片的像素值除以一个缩放因子，此处为1.
    img /= cfg.img_scaling_factor
    # 将图片的深度变到第一个位置。
    img = np.transpose(img, (2, 0, 1))
    # 给图片增加一个维度
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    # 将图片规定到指定的大小
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping, Ap):
    st = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)

    X, ratio = format_img(img, cfg)
    # 如果用的是tensorflow内核，需要将图片的深度变换到最后一位。
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # 进行区域预测
    # get the feature maps and output from the RPN
    # Y1: anchor包含物体的概率
    # Y2：每一个anchor对应的回归梯度
    #  F：卷积后的特征图
    [Y1, Y2, F] = model_rpn.predict(X)

    # this is result contains all boxes, which is [x1, y1, x2, y2]
    # result是一个又一个框
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    bbox_threshold = 0.50

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    # 分批训练，每一次遍历num_rois个预选框，总共要（result.shape[0] // cfg.num_rois + 1）次
    # 遍历所有的预选框，需要注意的是每一次预选框的个数为num_rois
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        # 取出num_rois个预选框，并增加一个维度（注：当不满一个num_rois，自动只取到最后一个）
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        # 没框
        if rois.shape[1] == 0:
            break
        # 当最后一次取不足num_rois个预选框时，补第一个框使其达到num_rois个
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            # 得到当前rois的shape
            curr_shape = rois.shape
            # 得到目标rois的shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            # 创建一个元素都为0的目标rois
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            # 将目标rois前面用现在的rois填充
            rois_padded[:, :curr_shape[1], :] = rois
            # 剩下的用第一个框填充
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        # 进行类别预测和边框回归
        # p_cls：该边框属于某一类别的概率
        # p_regr：每一个类别对应的边框回归梯度
        # F：rpn网络得到的卷积后的特征图
        # rois：处理得到的区域预选框
        [p_cls, p_regr] = model_classifier_only.predict([F, rois])
        # 遍历每一个预选框（p_cls.shape[1]预选框的个数）
        for ii in range(p_cls.shape[1]):
            # 如果该预选框的最大概率小于设定的阈值（即预测的肯定程度大于一定的值，我们才认为这次的类别的概率预测是有效的）
            # 或者最大的概率出现在背景上，则认为这个预选框是无效的，进行下一次预测。
            # p_cls[0, ii, :]类
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue
            # 不属于上面的两种情况，取最大的概率点处为此边框的类别得到其名称。
            cls_num = np.argmax(p_cls[0, ii, :])
            # 创建两个list，用于存放不同类别对应的边框和概率
            if cls_num not in boxes.keys():
                # cls_num类别对应的编号
                boxes[cls_num] = []
            # 得到该预选框的信息
            (x, y, w, h) = rois[0, ii, :] # ii是框
            try:
                # 根据类别对应的编号得到该类的边框回归梯度
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                # 对回归梯度进行规整化
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                # 对预测的边框进行修正
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            # 向相应的类里添加信息。
            # cfg.rpn_stride，边框的预测都是在特征图上进行的，要将其映射到规整后的原图上。是16吧？
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])

    print(boxes) # add by me
    result_txt_filename = "./predict_labels/" + os.path.basename(img_path).split('.')[0] + ".txt"

    with open(result_txt_filename, 'w') as f:
        for cls_num, box in boxes.items():
            # add some nms to reduce many boxes
            # 进行NMS
            boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
            boxes[cls_num] = boxes_nms
            print(class_mapping[cls_num] + ":")
            accall = 0

            for b in boxes_nms:
                b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
                print('{} prob: {}'.format(b[0: 4], b[-1]))
                accall += b[-1]
                #print("accall:".format(accall))
                f.write('{} {} {} {} {} {}\n'.format(class_mapping[cls_num], b[-1], b[0], b[1], b[2], b[3]))
            print("accall:{}".format(accall))
            avg = accall/len(boxes_nms)
            print("{} acc:{}".format(class_mapping[cls_num],avg))
            Ap[cls_num].append(avg)

    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    print('Elapsed time = {}'.format(time.time() - st))
    # cv2.imshow('image', img) # 显示图片 # 注释掉

    result_path = './predict_images/{}.png'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img) # 显示图片的操作
    # cv2.waitKey(0) # 注释掉，集群不能用


def predict(args_):
    path = args_.path
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    # tensorflow的输入方式是none,none,3
    input_shape_img = (None, None, 3)
    # 这里如果是resnet，num_features = 1024
    # 如果是vgg，num_features = 512
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    # 构建rpn输出
    # anchor的个数
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    # 构建classifier的输出，参数分别是：特征层输出，预选框，探测框的输入，多少个类，是否可训练。
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)

    # 构建网络
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    # 加载参数
    print('Loading weights from {}'.format(cfg.model_path))
    # self.model_path = 'model_trained/model_frcnn.vgg.hdf5'
    # 这里是resnet，所以不太明白这里的model_path？？？
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    # 这里是测试阶段。不用训练，但是这种加载模型，之后要求编译是keras要求的，随便找个mse即可
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    Ap = [[] for i in range(len(class_mapping)) ]
    print("Appp", Ap)
    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier_only, cfg, class_mapping, Ap)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping, Ap)
    print("Ap:")
    print(Ap)
    Acc = []
    for i in range(len(class_mapping)):
        sum = 0
        avg = 0
        for j in Ap[i]:
            sum += j
        if len(Ap[i]) != 0:
            avg = sum/(len(Ap[i]))
            Acc.append(avg)
        else:
            Acc.append(0)
    for i in range(len(class_mapping)):
        print(class_mapping[i]+"acc:{}".format(Acc[i]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='./test_sample', help='image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
