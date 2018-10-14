from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import dlib
import imutils
from imutils import face_utils


ready_to_detect_identity = True

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
FRmodel.save('FRmodel')
load_weights_from_FaceNet(FRmodel)

DATADIR="images"
folders = os.listdir('images') #0-5716 ish
categories= [folder for folder in folders]

def prepare_database():
    database = {}
    i=0
    extra_file_index = categories.index('.DS_Store')
    print(type(extra_file_index))
    del(categories[extra_file_index])

    for category in categories:
        path = os.path.join(DATADIR, category)

        for image in os.listdir(path):


            print("Adding face %d ..." % (i))
            name = categories[i]
            i+=1
            database[name]=img_path_to_encoding(image, FRmodel)
            print(database)
    return database

def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.
    """
    cv2.namedWindow("Class")
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        img = frame

        if ready_to_detect_identity:
            img = process_frame(img, frame)

        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,2)


    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []

    for rect in rects:

        (x,y,w,h) = face_utils.rect_to_bb(rect)

        img = cv2.rectangle(frame,(x, y),(x+w, y+h),(255,0,0),2)

        identity = find_identity(frame, x, y, x+w, y+h)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        cv2.imwrite('example.png',img)

        ready_to_detect_identity = False

    return img

def find_identity(frame, x1, y1, x2, y2):
    height, width, channels = frame.shape

    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    return recognize(part_image, database, FRmodel)

def recognize(image, database, model):

   encoding = img_to_encoding(image, model)

   min_dist = 5000
   identity = None

   for (name, db_enc) in database.items():
       dist = np.linalg.norm(db_enc - encoding)
       print('distance for %s is %s' %(name, dist))
       if dist < min_dist:
           min_dist = dist
           identity = name

   if min_dist > 0.52:
       return None
   else:
        print(str(identity))
        return str(identity)

def welcome_users(identities):
	print(identities)

    # Allow the program to start detecting identities again
ready_to_detect_identity = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)
