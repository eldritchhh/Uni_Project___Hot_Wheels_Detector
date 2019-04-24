from __future__ import division
import cv2
import numpy as np
import os

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


	
def square_crop(img, real_x1, real_y1, real_x2, real_y2):
	#
	#  NB. AL MOMENTO NON CONTROLLA CHE I VALORI ESCANO FUORI DALL'IMMAGINE
	#
	offset = int(round(((real_x2 - real_x1) - (real_y2 - real_y1))/2))

	if (offset > 0):
		offset = abs(offset)
		return img[ real_y1 - offset : real_y2 + offset,
					real_x1 : real_x2]
	else:
		offset = abs(offset)
		return img[ real_y1 : real_y2,
					real_x1 - offset : real_x2 + offset]

def save_cars(img, filename, car, prob, mode):
	if(mode == 'res'):
		dir = 'res'
	else: 
		if not os.path.exists('res/car_' + car):
			os.makedirs('res/car_' + car)
		dir = 'res/car_' + car

	img.save(dir + '/' + filename[-11:][0:7] + '_car_' + car + '_prob_' + prob + '.png')