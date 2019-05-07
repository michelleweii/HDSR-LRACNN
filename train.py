from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses_fn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
import os
import os.path
from keras_frcnn import resnet as nn
from keras_frcnn.simple_parser import get_data
# 如果用vgg
# from keras_frcnn import vgg as nn

def train_net():
    # config for data argument
    cfg = config.Config()

    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False
    cfg.num_rois = 32  # config中设置的是4
    cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())

    # TODO: the only file should to be change for other data to train
    cfg.model_path = 'samples.hdf5'

    cfg.simple_label_file = 'annotations_train.txt' # 训练集产生的标签

    all_images, classes_count, class_mapping = get_data(cfg.simple_label_file)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_imgs = [s for s in all_images if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_images if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    # there图片
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
                                                   K.image_dim_ordering(), mode='train')

    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,
                                                 K.image_dim_ordering(), mode='val')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)
    # classifier是什么？
    # classes_count {} 每一个类的数量：{'cow': 4, 'dog': 10, ...}
    # C.num_rois每次取的感兴趣区域，默认为32
    # roi_input = Input(shape=(None, 4)) 框框
    # classifier是faster rcnn的两个损失函数[out_class, out_reg]
    # shared_layers是vgg的输出feature map
    classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)
    # 定义model_rpn
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.model_path, by_name=True)
        model_classifier.load_weights(cfg.model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 10
    num_epochs = int(cfg.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:
                # 用来监督每一次epoch的平均正回归框的个数
                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        # 每次都框不到正样本，说明rpn有问题
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')

                # 迭代器，取数据
                # 训练rpn网络，X是图片，Y是对应类别和回归梯度（不是所有的点都参加训练，符合条件才参加训练）
                # next(data_gen_train)是一个迭代器。
                # 返回的是 np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)],
                # img_data_aug（我们这里假设数据没有进行水平翻转等操作。那么，x_img = img_data_aug）,
                # y_rpn_cls和y_rpn_regr是RPN的两个损失函数。
                X, Y, img_data = next(data_gen_train)


                # classifer和rpn网络交叉训练
                loss_rpn = model_rpn.train_on_batch(X, Y)
                P_rpn = model_rpn.predict_on_batch(X)

                # result是得到的预选框
                # 得到了region proposals，接下来另一个重要的思想就是ROI pooling，
                # 可将不同shape的特征图转化为固定shape，送到全连接层进行最终的预测。
                # rpn_to_roi接收的是每张图片的预测输出，返回的R = [boxes, probs]
                # ---------------------
                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=300)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # Y1根据预选框，得到这个预选框属于哪一类，
                # Y2这个类相应的回归梯度
                # X2是返回这个框
                """
                # 通过calc_iou()找出剩下的不多的region对应ground truth里重合度最高的bbox，从而获得model_classifier的数据和标签。
                # X2保留所有的背景和match bbox的框； Y1 是类别one-hot转码； Y2是对应类别的标签及回归要学习的坐标位置; IouS是debug用的。
                """
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)

                if X2 is None:
                    # 如果没有有效的预选框则结束本次循环
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                # 因为是one—hot，最后一位是1，则代表是背景
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0] # 将其变为1维的数组
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if cfg.num_rois > 1:
                    # 选择num_rois个数的框，送入classifier网络进行训练。 分类网络一次要训练多少个框
                    # 思路：当num_rois大于1的时候正负样本尽量取到一半，小于1的时候正负样本随机取一个。
                    if len(pos_samples) < cfg.num_rois // 2:
                        # 挑选正样本
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
                    try:
                        # 挑选负样本
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # 训练classifier网络
                # 是从位置中挑选，
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                #
                losses[iter_num, 0] = loss_rpn[1] # rpn_cls平均值
                losses[iter_num, 1] = loss_rpn[2] # rpn_regr平均值

                losses[iter_num, 2] = loss_class[1] # detector_cls平均值
                losses[iter_num, 3] = loss_class[2] # detector_regr平均值
                losses[iter_num, 4] = loss_class[3] # 4是准确率

                iter_num += 1

                # 进度条更新
                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])  # loss中存放了每一次训练出的losses
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        # 打印出前n次loss的平均值
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        # 当结束一轮的epoch时，只有当这轮epoch的loss小于最优的时候才会存储这轮的训练数据，
                        # 并结束这轮epoch进入下一轮epoch。
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
    print('Training complete, exiting.')


if __name__ == '__main__':
    train_net()
