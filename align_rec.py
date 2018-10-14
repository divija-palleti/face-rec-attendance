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


#facial detector and landmark detector (HOG based)

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"]) 
fa = face_utils.FaceAligner(predictor,desiredFaceWidth=256)
database = add_images()# prepare database

cap = cv2.VideoCapture(0)
color = (0,255,0)
stroke = 5
font = cv2.FONT_HERSHEY_SIMPLEX
stroke = 2 

while(True):
# Grayscaling, aligning and euclidean comparison
        ret, frame = cap.read() 
        image = imutils.resize(frame,width=256)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                
        rects = detector(gray,2)
        for rect in rects:
            (x,y,w,h) = face_utils.rect_to_bb(rect)
            aligned_img = fa.align(image,gray,rect)
            name = recognize(aligned_img, database, FRmodel)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(image, (x,y), (end_cord_x,end_cord_y), color, stroke)
            print(name)
            cv2.putText(image,name,(x,y),font,1,color,stroke,cv2.LINE_AA)   
            cv2.imshow('Classroom',image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cap.destroyAllWindows()