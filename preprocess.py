#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 05:17:17 2018

@author: Saicharan
"""


import os
import dlib
import glob
import imutils
import numpy as np
import cv2
from imutils import face_utils
from imutils import *
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import argparse
import pickle

DATADIR="images"
folders = os.listdir('images') #0-5716 ish
categories= [folder for folder in folders]

ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True,help="path to face detector")
args = vars(ap.parse_args())

extra_file_index = categories.index('.DS_Store')
print(type(extra_file_index))
del(categories[extra_file_index])

#print(categories)

training_data = []

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = face_utils.FaceAligner(predictor,desiredFaceWidth=256)

def create_training_data():
    for category in categories:
        path = os.path.join(DATADIR, category)
        class_num = categories.index(category)       
        for person_image in os.listdir(path):
            image = cv2.imread(person_image)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            rect = detector(gray, 2)
            aligned_img_array = fa.align(image,gray,rect) 
            training_data.append([aligned_img_array,class_num])

create_training_data()

X=[]
y=[]

for feature,label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X)

filename = 'X.pickle'
file = open('X.pickle','wb')
pickle.dump(X, file)

filename = 'y.pickle'
file = open('y.pickle','wb')
pickle.dump(y, file)

file = open('X.pickle','rb')
data = pickle.load(file)     
print(data)        

file = open('y.pickle','rb')
data = pickle.load(file)     
print(data)           
               
            
