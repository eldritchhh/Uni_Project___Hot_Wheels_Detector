from __future__ import division

###############   PROJECT IMPORT   ###############

import cv2
import numpy as np
import os, sys, platform, subprocess, time, glob, pickle
import matplotlib.image as mpimg

from Updater import Updater
from keras.models import load_model, Model
from keras import backend as K
from keras.layers import Input
from keras.preprocessing import image

from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
import keras_frcnn.utils as utils

###############   TELEGRAM BOT IMPORT   ############### 

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, Filters

###############   STUFF SETUP   ############### 

if not os.path.exists("Game Results"):
    os.makedirs("Game Results")

keyboard = [[InlineKeyboardButton(text='VERO', callback_data='V'),
             InlineKeyboardButton(text='FALSO', callback_data='F')]]

###############   GLOBAL VARIABLES   ############### 

alphabeth = ['A','B','C','D','E','F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','Z']
descriptions = ["E' la Ferrari completamente nera con il posteriore e l'anteriore tondeggiante? (Car A)",
"E' la Hot Wheels viola e nera senza tettuccio? (Car B)",
"E' la Ferrari completamente nera con il posteriore e l'anteriore squadrato? (Car C)",
"E' la Hot Wheels bianca con le strisce verdi e arancioni sul cofano e il tetto? (Car D)",
"E' la Ferrari completamente bianca? (Car E)",
"E' la Toyota da rally bianca e rossa? (Car F)",
"E' la Hot Wheels grigia con il numero 4 in rosso sul cofano? (Car G)",
"E' la Hot Wheels completamente grigia con gli interni rossi? (Car H)",
"E' la Hot Wheels grigia con le strisce blu e rosse sul cofano senza tettuccio? (Car I)",
"E' la Hot Wheels gialla con le strisce nere e rosse? (Car L)",
"E' la Hot Wheels gialla con la bandiera italiana sul cofano e il tettuccio? (Car M)",
"E' la Hot Wheels completamente gialla e tondeggiante? (Car N)",
"E' la Hot Wheels verde chiaro con le fiamme rosse sul cofano e il 6 sul posteriore? (Car O)",
"E' la Hot Wheels verde scuro con le aerografie rosse sul cofano e sul tetto? (Car P)",
"E' la Hot Wheels verde scuro da corsa? (Car Q)",
"E' la Hot Wheels blu e bianca con il motore a vista? (Car R)",
"E' la Hot Wheels blu con la scritta sul lato e tondeggiante? (Car S)",
"E' la Hot Wheels blu con la scritta sul lato e squadrata? (Car T)",
"E' la Hot Wheels completamente rossa con il cofano lungo? (Car U)",
"E' la Hot Wheels rossa con il cofano e il tettuccio nero? (Car V)",
"E' la Hot Wheels rossa con le aerografie in nero e il 15 sul cofano? (Car Z)"]

game_img_idx = 0
temp_crop_idx = 0
detect_res = []
class_res = []
img = []
##########   SETUP MODELS   ##########

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
model_classifier_only._make_predict_function()

model_classifier = Model([feature_map_input, roi_input], classifier)

# LOAD MODEL
detector_model_path = 'models/model_frcnn.hdf5'
model_rpn.load_weights(detector_model_path, by_name=True)
model_classifier.load_weights(detector_model_path, by_name=True)

# COMPILE MODEL
model_rpn.compile(optimizer='sgd', loss='mse')
model_rpn._make_predict_function()
model_classifier.compile(optimizer='sgd', loss='mse')
model_classifier._make_predict_function()


# LOAD PREDICTION MODEL
classifier_model_path = 'models/MobileNetV2_192_NoCar.h5'
classifier = load_model(classifier_model_path)
classifier._make_predict_function()





########################################   FUNCTIONS   ######################################## 

def detection(img):
 
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
			if(new_probs[jk] > 0.95):
				(x1, y1, x2, y2) = new_boxes[jk,:]
				(real_x1, real_y1, real_x2, real_y2) = utils.get_real_coordinates(ratio, x1, y1, x2, y2)
				
				cropped = utils.square_crop(img, real_x1, real_y1, real_x2, real_y2)
			
				cv2.imwrite('Game Results/crop_' + str(jk) + '.png', cropped)

				detect_res.append([real_x1, real_y1, real_x2, real_y2])
		#'''
		for jk in range(new_boxes.shape[0]):
			if(new_probs[jk] > 0.95):
			
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
		#'''

	cv2.imwrite('Game Results/' + str(game_img_idx) + '_Detector_Results.png', img)
 
	return detect_res


def classification():

	for idx, file in enumerate(glob.glob('Game Results/crop_*.png')):

		temp = image.load_img(file, target_size = (192,192))
        
		temp = image.img_to_array(temp)
		temp = np.expand_dims(temp, axis = 0)
		temp = temp / 255.0
		prob = str(round(np.max(classifier.predict(temp)[0]),2))
		temp_car = np.argmax(classifier.predict(temp)[0])
		car = str(alphabeth[temp_car])

		temp = image.array_to_img(temp[0])
		temp.save('Game Results/'+ str(game_img_idx) + '_Class_Res_Img_' + str(idx) + '_car_' + car + '_prob_' + prob + '.png')
		
		class_res.append(temp_car)
  
	return class_res


def image_handler(update, context):
    
    global game_img_idx
    global temp_crop_idx
    global class_res
    global detect_res
    global img
    
    print('---------- IMAGE SAVED')
    file = context.bot.getFile(update.message.photo[-1].file_id)
    file.download('Game Results/' + str(game_img_idx) + '_Original_Image.png')

    
    img = cv2.imread('Game Results/' + str(game_img_idx) + '_Original_Image.png')
    
    print('---------- DETECTION')
    detect_res = detection(img)

    print('---------- CLASSIFICATION')
    class_res = classification()
    
    img = cv2.imread('Game Results/' + str(game_img_idx) + '_Original_Image.png')

    if(len(detect_res) == len(class_res)):
        if(len(class_res) != 0):
            context.bot.sendPhoto(chat_id=update.message.chat_id, photo=open('Game Results/crop_' + str(temp_crop_idx) + '.png', "rb"),
                                caption=descriptions[class_res[temp_crop_idx]],  
                                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))
        else:
            context.bot.sendMessage(chat_id=update.message.chat_id,
									text = 'No Hot Wheels detected')
            
            game_img_idx = game_img_idx + 1
            
            for file in glob.glob('Game Results/crop_*.png'):
                os.remove(file)
                
            class_res = []
            detect_res = []
            print('------------------------------ END')
    
    else:
        print('ERRORE')
    

def callback_method(update, context):
    
    global game_img_idx
    global temp_crop_idx
    global class_res
    global detect_res
    global img
    
    data = update.callback_query.data
    
    if(temp_crop_idx == len(class_res) -1):
        
        if(data == 'V'):
            img = utils.insert_img(img, cv2.imread("images/V_tick.png", -1),
									detect_res[temp_crop_idx][0], 
									detect_res[temp_crop_idx][1],
									detect_res[temp_crop_idx][2],
									detect_res[temp_crop_idx][3])
        if(data == 'F'):
            img = utils.insert_img(img, cv2.imread("images/X_tick.png", -1),							
									detect_res[temp_crop_idx][0], 
									detect_res[temp_crop_idx][1],
									detect_res[temp_crop_idx][2],
									detect_res[temp_crop_idx][3])
        
        cv2.imwrite('Game Results/'+ str(game_img_idx) + '_Game_Result_Img.png', img)
        context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open('Game Results/'+ str(game_img_idx) + '_Game_Result_Img.png', "rb"))
        
        temp_crop_idx = 0
        game_img_idx = game_img_idx + 1
        
        for file in glob.glob('Game Results/crop_*.png'):
        	os.remove(file)
         
        class_res = []
        detect_res = []
        print('------------------------------ END')

    else:
        print('---------- WRITING')
        if(data == 'V'):
            img = utils.insert_img(img, cv2.imread("images/V_tick.png", -1),
									detect_res[temp_crop_idx][0], 
									detect_res[temp_crop_idx][1],
									detect_res[temp_crop_idx][2],
									detect_res[temp_crop_idx][3])
        if(data == 'F'):
            img = utils.insert_img(img, cv2.imread("images/X_tick.png", -1), 								
									detect_res[temp_crop_idx][0], 
									detect_res[temp_crop_idx][1],
									detect_res[temp_crop_idx][2],
									detect_res[temp_crop_idx][3])
        
        temp_crop_idx = temp_crop_idx + 1
        
        context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open('Game Results/crop_' + str(temp_crop_idx) + '.png', "rb"),
                                caption=descriptions[class_res[temp_crop_idx]], 
                                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))
        






########################################   BOT   ########################################

updater = Updater(token='*********', use_context=True)

dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
dispatcher.add_handler(CallbackQueryHandler(callback=callback_method))
print('---------- BOT IS READY')
updater.start_polling()
updater.idle()