import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D 
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path="/Users/bseethar/Desktop/sim_data/"

def load_training_data(logfile=data_path+"driving_log.csv"):
    # measurements are:
    # center_image, left_image, right_image, steering angle, ...
    # Tuning: steering_correction
    steering_correction = 0.2
    batch_size = 32
    test_size = 0.2

    train, valid = train_test_split(
        [entry for entry in csv.reader(open(logfile))],
        test_size=test_size)

    def _gen(entries):
        while 1: # Loop forever so the generator never terminates
            shuffle(entries)
            for offset in range(0, len(entries), batch_size):
                batch = entries[offset:offset+batch_size]
                features, values = [], []
                for entry in batch:
                    correction = 0.0
                    for path in entry[0:3]:
                        img = cv2.imread(data_path + "IMG/" + path.split('/')[-1])
                        steering = float(entry[3])
                        features.append(img)
                        values.append(steering + correction)
                        img = np.fliplr(img)
                        features.append(img)
                        values.append(-steering + correction)
                        # cycle through correction
                        if correction == 0.0:
                            correction = steering_correction
                        elif correction > 0.0:
                            correction = -correction
                yield shuffle(np.array(features), np.array(values))

    return _gen(train), len(6*train), _gen(valid), len(6*valid)

def trivial_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_and_save(name, model, epochs, train_gen, train_len, valid_gen, valid_len):
    name = name + "_model"
    model.save(name+".h5")
    history = model.fit_generator(train_gen,
                                  samples_per_epoch=train_len,
                                  validation_data=valid_gen,
                                  nb_val_samples=valid_len,
                                  nb_epoch=epochs,
                                  verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} MSE loss'.format(name))
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(name+".png")
    #plt.show()


train_and_save("trivial", trivial_model(), 3, *load_training_data())

