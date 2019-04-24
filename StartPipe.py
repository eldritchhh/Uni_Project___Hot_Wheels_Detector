from __future__ import division

from Updater import Updater
from keras.models import load_model, Model
from keras import backend as K
from keras.layers import Input
from keras.preprocessing import image

import cv2
import numpy as np
import os, sys, platform, subprocess, time, glob, pickle

from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
import keras_frcnn.utils as utils

def im_processing():
	return

def detection(img, local_filename):
	X, ratio = utils.format_img(img, C)
	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			pass

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < 0.8 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = utils.get_real_coordinates(ratio, x1, y1, x2, y2)
			
			cropped = utils.square_crop(img, real_x1, real_y1, real_x2, real_y2)
			#print("Saving cropped images")
			cv2.imwrite('res/cropped' + str(jk) + '.png', cropped)
		
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = utils.get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), 
						int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
	
	cv2.imwrite('res/' + local_filename[-11:][0:7] + '.png', img)
	
def classification(bot, chat_id, local_filename):
	for file in glob.glob('res/cropped*.png'):
		print(file)
		try:
			temp = image.load_img(file, target_size = (classif_input,classif_input))
			temp = image.img_to_array(temp)
			temp = np.expand_dims(temp, axis = 0)
			temp = temp / 255.0
			
			prob = str(round(np.max(classifier.predict(temp)[0]),2))
			car = str(alphabeth[np.argmax(classifier.predict(temp)[0])])

			temp = image.array_to_img(temp[0])
			utils.save_cars(temp, local_filename, car, prob, '4classification')
			
			bot.sendMessage(chat_id, 'Car: ' + car + ' - Prob: ' + prob)
			print('Car: ' + car + ' - Prob: ' + prob)

		except:
			pass

def imageHandler(bot, message, chat_id, local_filename):
	print('Image Name: ', local_filename)
	bot.sendMessage(chat_id, "Hi, please wait until the output is ready")

	for file in glob.glob('res/cropped*.png'):
		os.remove(file)
	
	img = cv2.imread(local_filename)
	
	# RUN PROCESSING
	###############################################################
	print('---------- PROCESSING')
	start = time.time()

	im_processing()

	end = time.time()
	print("Total time for processing:", end - start)
	
	# RUN DETECTION
	###############################################################
	print('---------- DETECTION')
	start = time.time()
    
	detection(img, local_filename)
	
	end = time.time()
	print("Total time to detect image:", end - start)

	# RUN CLASSIFIER
	###############################################################
	print('---------- CLASSIFICATION')
	start = time.time()
	
	classification(bot, chat_id, local_filename)
	
	end = time.time()
	print("Total time to classify images:", end - start)
	
	print("---------- IMAGE COMPLETED ----------")

##########################   MAIN   ###############################

# SETUP DETECTION MODEL

# READ PICKLE
config_output_filename = 'models/config.pickle'
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# TURN OFF DATA AUGM
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# SETUP CLASS MAPPING
class_mapping = C.class_mapping
if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = 32
num_features = 1024

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

# LOAD MODEL
detector_model_path = 'models/model_frcnn.hdf5'
model_rpn.load_weights(detector_model_path, by_name=True)
model_classifier.load_weights(detector_model_path, by_name=True)

# COMPILE MODEL
model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# LOAD PREDICTION MODEL
classifier_model_path = 'models/NoCarAdded_MobileNetV2.h5'
classifier = load_model(classifier_model_path)

# SETUP BOT AND FOLDERS
if not os.path.exists("res"):
    os.makedirs("res")

alphabeth = ['NOCAR', 'A','B','C','D','E','F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','Z']
bot_id = '875125863:AAHfSvyBHwGOiCZ31_etH_pGFyD_8hQOHrc'
classif_input = 192

updater = Updater(bot_id)
updater.setPhotoHandler(imageHandler)
print('---------- BOT IS READY')
updater.start()