"""

https://arxiv.org/abs/1511.07289
https://arxiv.org/pdf/1511.07289.pdf
https://www.reddit.com/r/MachineLearning/comments/2x0bq8/some_questions_regarding_batch_normalization/?su=ynbwk&st=iprg6e3w&sh=88bcbe40
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

"""

import csv
import cv2
import glob
import os
import numpy as np
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
from image import model_image_params

TF_FLAGS = tf.app.flags
FLAGS = TF_FLAGS.FLAGS
TF_FLAGS.DEFINE_integer('epochs', 3, "Training epochs.")
TF_FLAGS.DEFINE_boolean('drive', False, "Drive after training.")
TF_FLAGS.DEFINE_integer('batch_size', 32, "Training batch size.")
TF_FLAGS.DEFINE_string('model', 'trivial', "Model to be trained")
TF_FLAGS.DEFINE_string('datapath', '', "Data directory path")


def load_training_data(model):
    """
    procedure to load training data
    """
    steering_correction = 0.15
    test_size = 0.2

    entries = []
    for logfile in glob.glob(FLAGS.datapath + "*/*.csv"):
        imgpath = FLAGS.datapath + logfile.split("/")[-2] + "/IMG/"
        for line in csv.reader(open(logfile)):
            line[0] = imgpath + line[0].split("/")[-1]
            line[1] = imgpath + line[1].split("/")[-1]
            line[2] = imgpath + line[2].split("/")[-1]
            entries.append(line)
    train, valid = train_test_split([entry for entry in entries],
                                    test_size=test_size)
    print("Train: {}, Validate: {}".format(len(train), len(valid)))

    def _cam_correction(cam):
        return {
             'c': 0,
             'l': steering_correction,
             'r': -steering_correction
          }[cam]

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
                    for cam, path in zip(['c', 'l', 'r'], entry[0:3]):
                        imgfile = FLAGS.datapath + "/".join(path.split("/")[-3:])
                        img = preprocess(cv2.imread(imgfile), model)
                        steering = float(entry[3]) + _cam_correction(cam)
                        features.append(img)
                        values.append(steering)
                        if not augment:
                            break
                        img = np.fliplr(img)
                        features.append(img)
                        values.append(-steering)
                yield shuffle(np.array(features), np.array(values))
    return _gen(train), _gen(valid, augment=False)


def trivial_model(name):
    """
    trivial model that consists of just one Dense layer
    """
    shape = model_image_params[name]["shape"]
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
    c2d2_p = {"border_mode": "valid", "activation": "elu", "subsample": (2, 2)}
    c2d1_p = {"border_mode": "valid", "activation": "elu", "subsample": (1, 1)}
    d_p = {"activation": "elu", "W_regularizer": l2(0.001)}
    shape = model_image_params[name]["shape"]
    model = Sequential(name=name)
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=shape, name=name))
    model.add(Conv2D(24, 5, 5, **c2d2_p))
    model.add(Conv2D(36, 5, 5, **c2d2_p))
    model.add(Conv2D(48, 5, 5, **c2d2_p))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3, **c2d1_p))
    model.add(Conv2D(64, 3, 3, **c2d1_p))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, **d_p))
    model.add(Dropout(0.5))
    model.add(Dense(50, **d_p))
    model.add(Dense(10, **d_p))
    model.add(Dense(1))
    return model


def comma_ai_model(name):
    """
    comma.ai model based on comma github
    https://github.com/commaai/research/blob/3429b061cf2a15dc37661552775aa983206f7561/train_steering_model.py#L24
    """
    shape = model_image_params[name]["shape"]
    model = Sequential(name=name)
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=shape, output_shape=shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
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


def train_and_save(epochs, name, train_gen, valid_gen):
    """
    train model given by 'name' for 'epochs'
    """
    print("Training using {}".format(name))
    # the generators return len first
    model = get_model(name)
    train_len = next(train_gen)
    valid_len = next(valid_gen)
    model.summary()
    # adam = Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=Adam(), loss='mae')
    history = model.fit_generator(train_gen,
                                  samples_per_epoch=train_len,
                                  validation_data=valid_gen,
                                  nb_val_samples=valid_len,
                                  nb_epoch=epochs,
                                  verbose=1)
    model.save(name+"_model.h5")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} MSE loss'.format(name + " model"))
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(name+".png")
    # plt.show()


def main(_):
    """
    main entry point
    """
    train_and_save(FLAGS.epochs, FLAGS.model, *load_training_data(FLAGS.model))
    if FLAGS.drive:
        os.system("python drive.py {}_model.h5".format(FLAGS.model))


if __name__ == '__main__':
    tf.app.run()
