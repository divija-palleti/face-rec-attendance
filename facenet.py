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
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
from mtcnn.mtcnn import MTCNN
import dlib
from imutils import face_utils
import imutils
import pickle

ready_to_detect_identity = True

FRmodel = load_model('face-rec_Google.h5')
# detector = dlib.get_frontal_face_detector()
detector = MTCNN()
# FRmodel = faceRecoModel(input_shape=(3, 96, 96))
#
# # detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# def triplet_loss(y_true, y_pred, alpha = 0.3):
#     """
#     Implementation of the triplet loss as defined by formula (3)
#
#     Arguments:
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor images, of shape (None, 128)
#             positive -- the encodings for the positive images, of shape (None, 128)
#             negative -- the encodings for the negative images, of shape (None, 128)
#
#     Returns:
#     loss -- real number, value of the loss
#     """
#
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#
#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
#     basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#     loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
#
#     return loss
#
# FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
# load_weights_from_FaceNet(FRmodel)
def ret_model():
    return FRmodel

def prepare_database():
    pickle_in = open("data.pickle","rb")
    database =  pickle.load(pickle_in)
    return database

def webcam_face_recognizer(database):
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    while vc.isOpened():
        ret, frame = vc.read()
        img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = frame
        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img_rgb, frame)

        cv2.imshow("Preview", img)
        cv2.waitKey(1)

    vc.release()

def process_frame(img, frame):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    # rects = detector(img)
    rects = detector.detect_faces(img)
    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (i,rect) in enumerate(rects):
        (x,y,w,h) = rect['box'][0],rect['box'][1],rect['box'][2],rect['box'][3]
        img = cv2.rectangle(frame,(x, y),(x+w, y+h),(255,0,0),2)

        identity = find_identity(frame, x-50, y-50, x+w+50, y+h+50)
        cv2.putText(img, identity,(10,500), cv2.FONT_HERSHEY_SIMPLEX , 4,(255,255,255),2,cv2.LINE_AA)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        cv2.imwrite('example.png',img)

    return img

def find_identity(frame, x,y,w,h):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[y:y+h, x:x+w]

    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):

    encoding = img_to_encoding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

        if min_dist >0.1:
            print('Unknown person')
        else:
            print(identity)
    return identity

    # Allow the program to start detecting identities again
    ready_to_detect_identity = True

def recognizer(img1,img2):
    pass
if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)
