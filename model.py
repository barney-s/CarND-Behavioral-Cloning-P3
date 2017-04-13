import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense


data_path="/Users/bseethar/Desktop/sim_data/"

def load_driving_log(logfile=data_path+"driving_log.csv"):
    ## measurements are:
    ## center_image, left_image, right_image, steering angle, ...
    entries = [entry for entry in csv.reader(open(logfile))]
    X, y = [], []
    for entry in entries:
        for path in entry[0:3]:
            img = data_path + "IMG/" + path.split('/')[-1]
            X.append(cv2.imread(img))
            y.append(float(entry[3]))
    return np.array(X), np.array(y)

def trivial_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_and_save(name, model, X,y):
    import pdb; pdb.set_trace()
    model.fit(X, y, validation_split=0.2, shuffle=True)
    model.save(name+"_model.h5")


train_and_save("trivial", trivial_model(), *load_driving_log())
