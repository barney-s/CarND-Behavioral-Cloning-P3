"""
Code to define the CNN model and train it on the images collected from
Training Runs on the simulator. Started with trivial model and expanded to
well known CNN models.

Barni S
"""

import cv2
import glob
import errno
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from image import preprocess
from image import CROPPED_SHAPE
from keras.utils.visualize_util import plot as model_plot


# Using TensorFlow app flags to simplify command line arguments
TF_FLAGS = tf.app.flags
FLAGS = TF_FLAGS.FLAGS
TF_FLAGS.DEFINE_integer('epochs', 20, "Training epochs.")
TF_FLAGS.DEFINE_boolean('drive', False, "Drive after training.")
TF_FLAGS.DEFINE_boolean('balanced', False,
                        "When using already balanced data set this")
TF_FLAGS.DEFINE_integer('batch_size', 32, "Training batch size.")
TF_FLAGS.DEFINE_string('model', 'comma.ai', "Model to be trained")
TF_FLAGS.DEFINE_string('datapath', '', "Data directory path")


def generate_histogram(df, model, name, col="steering"):
    fig, ax = plt.subplots()
    df.hist(column=col, bins=100, ax=ax)
    fig.savefig('{}/{}_{}_hist.png'.format(model, model, name))


def balance_steering_data(df):
    """
     from ....|.....  more data between -0.25 to 0.25, ignoring extreme points
     to   .|.|||.|.|  data spread on all bins
     Balance the data by sampling a fraction of the data equal to the bin
     start value.  That way the skewed bell shaped histogram become evenly
     spread out.
    """
    y = -0.5
    yd = 1/100
    ndf = None
    for x in range(1, 100):
        cdf = df[df["steering"] >= y]
        cdf = cdf[cdf["steering"] < y+yd]
        if len(cdf):
            cdf = cdf.sample(frac=abs(y))
        y += yd
        if ndf is None:
            ndf = cdf
        else:
            ndf = ndf.append(cdf)
    ndf = ndf.append(df[df["steering"] < -0.5])
    ndf = ndf.append(df[df["steering"] > 0.5])
    return ndf


def save_balanced_data(df, model):
    """
    Save the selected image files as well as the csv
    """
    imgdir = model + "/data/balanced/IMG/"
    mkdir_p(imgdir)
    for entry in df.values.tolist():
        for idx in range(3):
            _file = imgdir + "/" + entry[idx].split("/")[-1]
            shutil.copy(entry[idx], _file)
            # print("copy: {} -> {}".format(entry[idx], _file))
    df['center'] = df['center'].map(lambda x: imgdir+x.split("/")[-1])
    df['left'] = df['left'].map(lambda x: imgdir+x.split("/")[-1])
    df['right'] = df['right'].map(lambda x: imgdir+x.split("/")[-1])
    df.to_csv("{}/data/balanced/{}_input.csv".format(model, model),
              index=False, header=False,
              columns=['center', 'left', 'right', 'steering',
                       'throttle', 'brake', 'speed'])


def load_training_data(model):
    """
    procedure to load training data
    """
    steering_correction = 0.15
    test_size = 0.01

    df = None
    # Supports reading multiple runs saved as separate directories.
    # Assumes called from one level up
    for logfile in glob.glob(FLAGS.datapath + "*/*.csv"):
        print("using csv: {}".format(logfile))
        imgpath = FLAGS.datapath + logfile.split("/")[-2] + "/IMG/"
        _df = pd.read_csv(logfile,
                          names=['center', 'left', 'right', 'steering',
                                 'throttle', 'brake', 'speed'])
        _df['center'] = _df['center'].map(lambda x: imgpath+x.split("/")[-1])
        _df['left'] = _df['left'].map(lambda x: imgpath+x.split("/")[-1])
        _df['right'] = _df['right'].map(lambda x: imgpath+x.split("/")[-1])
        if df is None:
            df = _df
        else:
            df = df.append(_df)

    # if we are dealing with already balanced data, we can skip this step.
    # useful when retraining the model on a previously balanced model.
    if not FLAGS.balanced:
        print("balancing data")
        # plot the histogram of original data
        generate_histogram(df, model, "orig")
        ndf = balance_steering_data(df)
        save_balanced_data(ndf, model)
        # plot the histogram of balanced data
        generate_histogram(ndf, model, "balanced")
    else:
        print("not balancing data")
        ndf = df
    entries = ndf.values.tolist()
    train, valid = train_test_split(entries, test_size=test_size)
    print("Train: {}, Validate: {}".format(len(train), len(valid)))

    # utility to correct steering angle based on camera used
    def _cam_correction(cam):
        return {
             'c': 0,
             'l': steering_correction,
             'r': -steering_correction
          }[cam]

    # procedure that acts as a generator
    def _gen(entries, augment=True):
        if augment:
            yield 6*len(entries)
        else:
            yield len(entries)
        while 1:  # forever
            shuffle(entries)
            for offset in range(0, len(entries), FLAGS.batch_size):
                batch = entries[offset:offset+FLAGS.batch_size]
                features, values = [], []
                for entry in batch:
                    for cam, imgfile in zip(['c', 'l', 'r'], entry[0:3]):
                        img = cv2.imread(imgfile)
                        # be resilient to errors that may occur when curating
                        # the input data
                        if img is None:
                            print("Img missing: {}".format(imgfile))
                            break
                        img = preprocess(img)
                        steering = float(entry[3]) + _cam_correction(cam)
                        features.append(img)
                        values.append(steering)
                        # if augment is not set, even left, right cam
                        # images are ignored
                        if not augment:
                            break
                        # augment by flipping the image
                        img = np.fliplr(img)
                        features.append(img)
                        values.append(-steering)
                yield shuffle(np.array(features), np.array(values))

    # Please note: i have turned off augmenting the input images
    return _gen(train, augment=False), _gen(valid, augment=False)


def trivial_model(name):
    """
    trivial model that consists of just one Dense layer
    """
    shape = CROPPED_SHAPE
    model = Sequential(name=name)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                     input_shape=shape, name=name))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def nvidia_model(name):
    """
    modified NVIDIA model
    """
    c2d2_p = {"border_mode": "valid",
              "activation": "tanh",
              "subsample": (2, 2)}
    c2d1_p = {"border_mode": "valid",
              "activation": "tanh",
              "subsample": (1, 1)}
    d_p = {"activation": "tanh", "W_regularizer": l2(0.001)}
    model = Sequential(name=name)
    model.add(Conv2D(24, 5, 5, input_shape=CROPPED_SHAPE, **c2d2_p))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, 5, 5, **c2d2_p))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, 5, 5, **c2d2_p))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, 3, **c2d1_p))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, 3, **c2d1_p))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50, **d_p))
    model.add(Dropout(0.5))
    model.add(Dense(20, **d_p))
    model.add(Dropout(0.3))
    model.add(Dense(10, **d_p))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


def comma_ai_model(name):
    """
    Modified comma.ai model based on comma github
    https://github.com/commaai/research/blob/3429b061cf2a15dc37661552775aa983206f7561/train_steering_model.py#L24
    """
    shape = CROPPED_SHAPE
    model = Sequential(name=name)
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=shape,
                            border_mode="same", activation="tanh"))
    # ELU activation performed very poorly in this case !!
    # model.add(ELU())
    model.add(Dropout(.5))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2),
                            border_mode="same", activation="tanh"))
    # model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2),
                            border_mode="same", activation="tanh"))
    model.add(Dropout(.2))
    model.add(Flatten())
    # model.add(ELU())
    model.add(Dense(512, activation="tanh", W_regularizer=l2(0.01)))
    model.add(Dropout(.5))
    model.add(Dense(128, activation="tanh", W_regularizer=l2(0.01)))
    model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Dense(1, activation="tanh"))
    return model


def get_model(name):
    """
    get model based on name
    """
    if name == "trivial":
        model = trivial_model
    elif name == "comma.ai":
        model = comma_ai_model
    elif name == "nvidia":
        model = nvidia_model
    return model(name)


def mkdir_p(path):
    """
    mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train_and_save(epochs, name, train_gen, valid_gen):
    """
    train model given by 'name' for 'epochs'
    """
    print("Training using {}".format(name))
    # the generators return len first
    model = get_model(name)
    train_len = next(train_gen)
    valid_len = next(valid_gen)
    # print the model summary
    model.summary()
    # Using adam optimizer
    model.compile(optimizer='adam', loss="mse")
    history = model.fit_generator(train_gen,
                                  samples_per_epoch=train_len,
                                  validation_data=valid_gen,
                                  nb_val_samples=valid_len,
                                  nb_epoch=epochs,
                                  verbose=1)
    file_prefix = "{}/{}".format(name, name)
    # #save as arch + weights
    with open(file_prefix + '.json', 'w') as jsonf:
        jsonf.write(model.to_json())
    model.save_weights(file_prefix + '.h5')
    # #save as model with weights
    # model.save(file_prefix + ".h5")

    # plot the mse over training epochs
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} MSE loss'.format(name + " model"))
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(file_prefix + "_training.png")
    # plt.show()

    # save visual model plot
    model_plot(model, to_file=file_prefix + '_model.png', show_shapes=True)


def main(_):
    """
    main entry point
    """
    mkdir_p(FLAGS.model)
    train_and_save(FLAGS.epochs, FLAGS.model, *load_training_data(FLAGS.model))
    if FLAGS.drive:
        os.system("python drive.py {}".format(FLAGS.model))


# Main entry point
if __name__ == '__main__':
    tf.app.run()
