import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
import dlib
import imutils
from imutils import face_utils
from inception_blocks_v2 import *
from keras import backend as K
K.set_image_data_format('channels_first')
from imutils.face_utils import FaceAligner

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


cap = cv2.VideoCapture(0)

def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 1.00:
        return None
    else:
        return identity
def recognize():
    while(True):
        ret, frame = cap.read()
        image = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray,1)

        for (i,rect) in enumerate(rects):
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            for (x,y) in shape:
                cv2.circle(image,(x,y),1,(0,0,255),-1)

            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(image, gray, rect)

            identity = who_is_it(faceAligned, database, FRmodel)
            print(identity)
        cv2.imshow("Output", image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cap.destroyAllWindows()


def prepare_database():
    database = {}
    for file in glob.glob("Images/*"):
        for image in glob.glob(os.path.join(file,"*")):
            identity = os.path.dirname(image)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rect = detector(gray,1)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(image, gray, rect)
            database[identity] = img_path_to_encoding(faceAligned, FRmodel)
    return database

if __name__=='__main__':
    database = prepare_database()
    recognize()
