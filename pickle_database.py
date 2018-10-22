import glob
import os
from fr_utils import img_path_to_encoding
from facenet import ret_model
import pickle

def prepare_database():
    database = {}
    for file in glob.glob("Images/*"):
        for image in glob.glob(os.path.join(file,"*")):
            identity = os.path.dirname(image)
            database[identity] = img_path_to_encoding(image, FRmodel)
    return database

def prepare_pickle():
    database = prepare_database()
    pickle_out = open("data.pickle","wb")
    pickle.dump(database,pickle_out)
    pickle_out.close()

if __name__=='__main__':
    FRmodel = ret_model()
    prepare_pickle()
