#!/usr/bin/env python3
# -*- coding: utf-8 -*-odel()
"""
Created on Sat Oct 13 08:46:31 2018

@author: Admin
"""

import os
import glob
import numpy as np
import cv2
from imutils import face_utils
from imutils import *
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import argparse
from keras import backend as K
import dlib
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-p","--shape-predictor",required=True,help="path to face detector")
args = vars(ap.parse_args())

K.set_image_data_format('channels_first')

#FRmodel = faceRecoModel(input_shape=(3,96,96))
FRmodel = tf.keras.models.load_model('FRmodel')
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = face_utils.FaceAligner(predictor,desiredFaceWidth=256)

def triplet_loss(y_true,y_pred,alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
   
    return loss

#FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# add images to database

def add_images():
    database = {}
    i=0
    for file in glob.glob("images/*"):
        i+=1
        print("Adding face %d ..." % (i))
        name = os.path.splitext(os.path.basename(file))[0]
        file = cv2.imread(file,1)
        print(file)
        database[name]=img_to_encoding(file, FRmodel)
        print(database)
    return database

def recognize(image,database,model):
    encoding = img_to_encoding(image,model)
    
    min_dist = 100
    _id = None
    
    for(name,enc) in database.items():
        
        dist = np.linalg.norm(enc - encoding)
        
        if dist < min_dist:
            min_dist = dist
            _id = name
        
        if min_dist > 1.00:
            return None
        else:
            return _id
            