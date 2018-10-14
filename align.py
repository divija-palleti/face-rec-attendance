#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:03:02 2018

@author: Saicharan
"""

from imutils import face_utils
import imutils
import numpy as np
import argparse
import dlib
import cv2
from model import *

ap = argparse.ArgumentParser()

ap.add_argument("-p","--shape-predictor",required=True,help="path to face detector")
args = vars(ap.parse_args())

#facial detector and landmark detector (HOG based)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = face_utils.FaceAligner(predictor,desiredFaceWidth=256)
add_images()# prepare database

cap = cv2.VideoCapture(0)
color = (0,255,0)
stroke = 5
font = cv2.FONT_HERSHEY_SIMPLEX
stroke = 2 

while(True):
# Grayscaling
        ret, image = cap.read() 
        image = imutils.resize(image,width=500)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
        # Detect facess
        cv2.imshow("Input", image)
        rects = detector(gray, 3)
            
                
        for rect in rects:           
            
            (x,y,w,h) = face_utils.rect_to_bb(rect)
            face_orig = imutils.resize(image[x:x+w,y:y+h],width=256)
            face_Aligned = fa.align(image,gray,rect)
            name = recognize(face_Aligned,database,FRModel)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(image, (x,y), (end_cord_x,end_cord_y), color, stroke)
            cv2.putText(image,name,(x,y),font,1,color,stroke,cv2.LINE_AA)            