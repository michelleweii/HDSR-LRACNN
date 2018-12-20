import cv2
import numpy as np
import copy

# 1、检查图片信息和读取图片
# 2、随机翻转和旋转图片
# 3、返回图片信息和图片

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	# 当进行图片增强的时候，不会改变原有的信息
	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		# shape：先行后列
		rows, cols = img.shape[:2]
		# 是否要水平翻转and随机数
		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			# cv2.flip(img, 1)：将图片延y轴翻转（左右对调）
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		# 是否要垂直翻转and随机数
		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			# 上下对调
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		# 旋转角度
		if config.rot_90:
			# np.random.choice([0,90,180,270],1)[0]：从给定的list选择1代表一个数【如果没有后面[0]，那么它返回的还是一个list】
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				# img = np.transpose(img, (1,0,2))：高宽互换
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			# 调整坐标
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
