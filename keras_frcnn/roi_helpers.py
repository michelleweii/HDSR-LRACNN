import numpy as np
import pdb
import math
from . import data_generators
import copy


# 1、遍历预选框得到其最大的交并比
# 2、根据交并比得到类别和回归梯度
# rpn_to_roi输出作为calc_iou的输入
# # 通过calc_iou()找出剩下的不多的region对应ground truth里重合度最高的bbox，
# # 从而获得model_classifier的目标和标签。
def calc_iou(R, img_data, C, class_mapping):
    """
    （如何得到用于训练RPN的gt的）
    该函数的作用是生成classifier网络训练的数据,需要注意的是它对提供的预选框还会做一次选择,就是将容易判断的背景删除
    判断框是哪一类
    :param R: 预选框
    :param img_data: img_data包含一张图片的路径,bbox坐标和对应的分类（可能一张图片有多组，即表示图片里包含多个对象）
    :param C: 训练信息
    :param class_mapping:类别与映射数字之间的关系
    :return:
    np.expand_dims(X, axis=0)：筛选后的预选框
    np.expand_dims(Y1, axis=0)：对应的类别
    np.expand_dims(Y2, axis=0)：相应的回归梯度
    IoUs：交并比
    """
    # 得到图片的基本信息，并将图片的最短边规整到相应的长度。
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])

    # get image dimensions for resizing
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((len(bboxes), 4))

    # 将每一个预选框与所有的bboxes求交并比，记录最大交并比。用来确定该预选框的类别。
    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # /C.rpn_stride是因为在特征图上进行的，需要将原图映射到特征图
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []  # for debugging only

    # 遍历所有的预选框R，它并不需要做规整。由于RPN网络预测的框就是基于最短框被规整后的
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                           [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        """
        对最佳的交并比作不同的判断
        
        1、当最佳交并比小于最小的阈值时，放弃此框。因为，交并比太低就说明是很好判断的背景没必要训练。
        当大于最小阈值时，则保留相关的边框信息
        2、当在最小和最大之间，就认为是背景。有必要进行训练。
        3、大于最大阈值时认为是物体，计算其边框回归梯度
        """
        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # 得到该类别对应的数字
        class_num = class_mapping[cls_name]
        # 将该数字对应的地方置为1【one-hot】
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        # 将该类别加入到y_class_num
        y_class_num.append(copy.deepcopy(class_label))
        # coords是用来存储边框回归梯度，labels来决定是否要加入计算loss中
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            # 如果不是背景的话，计算相应的回归梯度
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            #  # coords: 坐标调整：相当于coords是回归要学习的内容
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi) # 框
    Y1 = np.array(y_class_num) # 类别
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1) # 坐标+回归梯度
    # 返回数据
    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    """
    从所给定的所有框中选择指定个数最合理的边框
    :param boxes: 框
    :param overlap_thresh:
    :param max_boxes:
    :return: 框（x1,y1,x2,y2）的形式
    """
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob] format!!!! with prob built in
    if len(boxes) == 0:
        # 没有框
        return []
    # normalize to np.array
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 对输入数据进行确认
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)
    # 转换数据类型
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexed
    pick = []
    # calculate the areas
    area = (x2 - x1) * (y2 - y1)
    # sorted by boxes last element which is prob
    # indexes是数值所在的位置。
    # list = [1,3,2], after argsort
    # idxs = [0,2,3]
    indexes = np.argsort([i[-1] for i in boxes])

    # 按照概率从大到小取出框，且框的重合度不可以高于阈值：
    # 思路：
    # 1、每一次取概率最大的框（即indexes最后一个）
    # 2、删除掉剩下的框中重合度高于阈值的框
    # 3、直到取满max_boxes为止
    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        # 取出idexes队列中最大概率框的序号，将其添加到pick中
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        # 交并比
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        # 删除重叠率较高的位置
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    # 返回相应的框，pick中存的是位置
    return boxes

# 遍历每一个预选框
# 根据rpn网络的结果进行预选框修正
# 删除不合理的预选框，并选出指定个数的预选框
# 此函数的主要作用是：把由RPN输出的所有可能的框过滤掉重合度高的框，降低计算复杂度。
def rpn_to_roi(rpn_layer, regr_layer, cfg, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """
    将rpn网络的预测结果转化到一个个预选框
    :param rpn_layer: 框对应的概率（是否存在物体）
    :param regr_layer: 每个框对应的回归梯度
    :param cfg: C信息对象
    :param dim_ordering: 维度组织形式
    :param use_regr: 是否进行边框回归
    :param max_boxes: 要取出多少个框
    :param overlap_thresh: 重叠度的阈值
    :return: 返回指定个数的预选框，形式是(x1,y1,x2,y2)
    """
    regr_layer = regr_layer / cfg.std_scaling

    anchor_sizes = cfg.anchor_box_scales
    anchor_ratios = cfg.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'th':
        (rows, cols) = rpn_layer.shape[2:]

    elif dim_ordering == 'tf':
        (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    if dim_ordering == 'tf':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    elif dim_ordering == 'th':
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # 得到框的长宽在原图上的映射
            anchor_x = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride # 除以特征图的缩放因子
            anchor_y = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride
            if dim_ordering == 'th':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                # 得到相应尺寸的框对应的回归梯度，将深度都放到第一个维度
                # curr_layer代表的是特定长度和比例的框所代表的编号
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))
            # 得到每个点所对应的anchor坐标
            # cols宽度，rows高度
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x / 2 # x
            A[1, :, :, curr_layer] = Y - anchor_y / 2 # y
            A[2, :, :, curr_layer] = anchor_x # w
            A[3, :, :, curr_layer] = anchor_y # h

            if use_regr:
                # 回归梯度
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # 对修正后的边框存在的一些不合理的地方进行校正
            # e.g. 边框回归后的左上角和右下角的点不能超过图片外，框的宽高不可以小于0
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            curr_layer += 1

    # 得到的all_boxes形状是(n,4),和每一个框对应的概率all_probs形状是(n,)
    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # np.where返回位置信息，删除不符合要求点的一种方法
    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    # 删除一些不合理的点，即右下角的点值要小于左上角的点值
    all_boxes = np.delete(all_boxes, ids, 0)
    # np.delete()最后一个参数实在哪一个维度删除
    all_probs = np.delete(all_probs, ids, 0)

    # I guess boxes and prob are all 2d array, I will concat them
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs])))
    result = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes)
    # omit the last column which is prob
    result = result[:, 0: -1]
    return result
