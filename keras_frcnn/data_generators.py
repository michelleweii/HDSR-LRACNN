from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=28):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		# 连接字的数据集都使用这个
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		# self.classes：去除类别数为0的类，比如背景。
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		# itertools.cycle：将一个list做成一个迭代器对象
		# next：取出这个值
		# 组合以实现 无限地反复地从数组中取值
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	# 当输入一张图片时，决定是否要跳过该图片，该图片中包含需要的类返回False,否则返回True
	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
	"""
	得到每一个anchor的性质
	1、遍历anchor计算iou
	2、根据iou得到anchor的属性
	3、选择指定个数的anchors返回最终结果
	:param C:训练信息类
	:param img_data:图片信息包含一张图片的路径，bbox坐标和对应的分类（可能一张图片有多组，即表示图片里包含多个对象）
	:param width:图片宽度
	:param height:图片高度（重新计算bboxes要用）
	:param resized_width:规整化后的图片宽度
	:param resized_height:规整化后的图片高度
	:param img_length_calc_function:计算特征图大小的函数
	:return: np.copy(y_rpn_cls)返回锚点是否包含类，np.copy(y_rpn_regr)相应的回归梯度
	注：只会返回num_regions（这里设置为256）个有效的正负样本
	"""

	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# calculate the output map size based on the network architecture
	# 得到特征图的宽度和高度
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	num_bboxes = len(img_data['bboxes'])

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	# 把bbox中的x1,x2,y1,y2分别通过缩放匹配到resize以后的图像。这里记做gta，尺寸为(num_of_bbox,4)。
	gta = np.zeros((num_bboxes, 4))
	# 将最短边规整到指定长度600后，相应的边框长度也需要发生变化。
	# gta的存储形式是(x1,x2,y1,y2)
	for bbox_num, bbox in enumerate(img_data['bboxes']):

		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth
	# 遍历所有的anchor，一个点产生9个anchor
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			# 长
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			# 宽
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			# 遍历每个点，把中心点映射回原图上
			for ix in range(output_width):					
				# x-coordinates of the current anchor box
				# downscale映射回原图，ix坐标
				# x1_anc，x2_anc是左上，右下中心点的坐标
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries
				# 要求不超出这张图，删掉这个框
				if x1_anc < 0 or x2_anc > resized_width:
					continue

				#
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					# 通过以上步骤，确定了一个预选框组合，又确定了中心点，即唯一确定一个框了，
					# 接下来确定这个框的性质：是否包含物体、包含物体其回归梯度是多少
					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
						# 计算预选框和bbox的交并比
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						# 满足条件，计算回归梯度
						# 回归梯度：对预选框进行修正，tx是修正后的坐标
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))


						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							# 每一个类记录一个和它最好的框，
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							# 如果现在的交并比大于阈值
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								# 不仅大于阈值，还大于最好的
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
					# 当结束对所有的bbox的遍历时，来确定该预选框的性质
					# y_is_box_valid：该预选框是否可用（nertual就是不可用）
					# y_rpn_overlap 该预选框是否包含物体
					#
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						# 计算回归梯度
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

	# 以上，先遍历anchor_size，再遍历ratio，遍历每一个点，得到每一个点的框，判断框的性质，将相应的地方置位0，1，
	# 经过一次循环，所有点的性质都得到了


	# 查漏补缺
	# 有物体的地方，确保相关的框有东西将它包住。如果有一个bbox没有pos的预选框和其对应，这时找一个与它交并比最高的anchor，设置为pos
	# we ensure that every bbox has at least one positive RPN region
	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			# 将相交最多的框设置为1
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	# 将深度变到第一位，给向量增加一个维度
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
	# 正框总是少数
	# 正框所在的位置
	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	# 负框所在的位置
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.

	# 从可用的预选框中挑选256个
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		# 如果pos的个数大于一半，则将多下来的地方设置为不可用。如果小于不作处理
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		# 将pos和neg总数超过256个的neg预选框设为不可用
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
	# axis=0是batch_size
	# rpn model的output和label形状不匹配？
	# 因为region proposal过程针对每一个锚点的每一个anchor都是有输出的，其实有很多anchor是不可用的，
	# 在y_is_box_valid那个array里面有记录。那么我们在计算loss时，也是不计算这些anchor的。
	# 因此我们在输出时，将与输出等形状的y_is_box_valid array拼接起来，计算loss时做一个对应元素的乘法，
	# 就可以舍去这些anchor产生的loss了。所以regr那里要将y_is_box_valid repeat 4倍，再与输出concatenate起来。
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
	# y_rpn_cls, y_rpn_regr。分别用于确定anchor是否包含物体，和回归梯度。
	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)
	"""
	为了得到rpn网络的训练数据
	1、读取并增强图片（旋转、镜像等）
	2、得到rpn网络训练数据
	3、对图片和数据做最后处理并返回
	:param all_img_data:图片信息
	:param class_count:类别统计信息
	:param C:训练信息类
	:param img_length_calc_function:计算输出特征图大小的函数
	:param backend:keras用什么内核
	:param mode:是否为训练
	:return:1.图片 2.数据对象（是否包含对象+回归梯度） 3.增强后的图片信息
	"""
	# 图片选择，以达到类平衡，是否跳过此图片
	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			random.shuffle(all_img_data)

		for img_data in all_img_data:
			# 从这个地方可以看到最终提取RPN训练集数据, 是一张图片一张图片的去提取的.
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation
				# augment用来增强图片
				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False) # 图像增强关闭
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				# 返回图片经过图像增强后的长and宽
				(width, height) = (img_data_aug['width'], img_data_aug['height'])

				# 实际图片的长and宽
				(rows, cols, _) = x_img.shape

				assert cols == width
				assert rows == height

				# get image dimensions for resizing
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

				# resize the image so that smalles side is length = 600px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

				try:
					# 得到每一张图片的每一个点的两个特性以供rpn网络训练
					# y_rpn_cls：是否包含物体；
					# y_rpn_regr：回归梯度是多少
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				# 减去均值，vgg是按照bgr训练的
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor
				# 将深度变为第一个维度
				x_img = np.transpose(x_img, (2, 0, 1))
				# 给图片增加一个维度，batch_size
				x_img = np.expand_dims(x_img, axis=0)
				# 给回归梯度除以一个因子
				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1))
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue



