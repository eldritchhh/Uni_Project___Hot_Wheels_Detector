from __future__ import division
import cv2
import numpy as np
import os
import math
from scipy import ndimage

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

###########################################################################
	
def square_crop(img, real_x1, real_y1, real_x2, real_y2):
	offset = int(round(((real_x2 - real_x1) - (real_y2 - real_y1))/2))
	l = img.shape[0]

	if (offset > 0):
		offset = abs(offset)
		new_y1 = real_y1 - offset if real_y1 - offset > 0 else 0
		new_y2 = real_y2 + offset if real_y2 + offset < l else l-1
		real_x1 = 0 if real_x1 < 0 else real_x1
		real_x2 = l-1 if real_x2 > l else real_x2

		return img[ new_y1 : new_y2,
					real_x1 : real_x2]
	else:
		offset = abs(offset)
		new_x1 = real_x1 - offset if real_x1 - offset > 0 else 0
		new_x2 = real_x2 + offset if real_x2 + offset < l else l-1
		real_y1 = 0 if real_y1 < 0 else real_y1
		real_y2 = l-1 if real_y2 > l else real_y2

		return img[ real_y1 : real_y2,
					new_x1 : new_x2]

def save_cars(img, filename, car, prob, mode):
	if(mode == 'res'):
		dir = 'res'
	else: 
		if not os.path.exists('res/car_' + car):
			os.makedirs('res/car_' + car)
		dir = 'res/car_' + car

	img.save(dir + '/' + filename[-11:][0:7] + '_car_' + car + '_prob_' + str(prob) + '.png')

def insert_img(img, marker, x1, y1, x2, y2):

	l = img.shape[0]
	x1 = 0 if x1 < 0 else x1
	y1 = 0 if y1 < 0 else y1
	x2 = l-1 if x2 > l else x2
	y2 = l-1 if y2 > l else y2

	offset = int(round(((x2 - x1) - (y2 - y1))/2))
	
	if (offset > 0):
		offset = abs(offset)
		new_x1 = x1 + offset
		new_x2 = x2 - offset
		marker = cv2.resize(marker, (new_x2 - new_x1, y2 - y1))
		alpha_marker = marker[:, :, 3] / 255.0
		alpha_img = 1.0 - alpha_marker

		for c in range(0, 3):
			img[y1:y2, new_x1:new_x2, c] = (alpha_marker * marker[:, :, c] + 
				alpha_img * img[y1:y2, new_x1:new_x2, c])
	else:
		offset = abs(offset)
		new_y1 = y1 + offset
		new_y2 = y2 - offset
		marker = cv2.resize(marker, (x2 - x1, new_y2 - new_y1))
		alpha_marker = marker[:, :, 3] / 255.0
		alpha_img = 1.0 - alpha_marker

		for c in range(0, 3):
			img[new_y1:new_y2, x1:x2, c] = (alpha_marker * marker[:, :, c] + 
				alpha_img * img[new_y1:new_y2, x1:x2, c])

	return img

def gamma(img):

	ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	y = ycbcr[:,:,0] / 255.0
	mask = ndimage.gaussian_filter(1 - y, sigma=5, mode='nearest')
	mask = 2 ** ((0.5 - mask) / 0.5)
	y = y ** mask
	ycbcr[:,:,0] = y * 255.0

	return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)