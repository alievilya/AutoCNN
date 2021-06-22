import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes Tensorflow debuggin ouputs

import tensorflow as tf

tf.get_logger().setLevel('INFO')  # Removes Tensorflow debugging ouputs

from auto_cnn.gan import AutoCNN
from sklearn.metrics import roc_auc_score as roc_auc
import statistics

import random
import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
import pandas as pd

# Sets the random seeds to make testing more consisent
random.seed(12)
tf.random.set_seed(12)


class RocCallback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train, 4)), str(round(roc_val, 4))),
              end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def auc_f(truth, prediction):
    roc_auc_values = []
    for predict, true in zip(prediction, truth):
        y_true = [0 for _ in range(3)]
        y_true[true[0]] = 1
        roc_auc_score = roc_auc(y_true=y_true,
                                y_score=predict)
        roc_auc_values.append(roc_auc_score)
    roc_auc_value = statistics.mean(roc_auc_values)
    return roc_auc_value


def from_json(file_path):
    df_train = pd.read_json(file_path)
    Xtrain = get_scaled_imgs(df_train)
    Ytrain = np.array(df_train['is_iceberg'])
    df_train.inc_angle = df_train.inc_angle.replace('na', 0)
    idx_tr = np.where(df_train.inc_angle > 0)
    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0], ...]
    Ytrain_new = []
    for y in Ytrain:
        new_Y = []
        new_Y.append(y)
        Ytrain_new.append(y)
    Ytrain = np.array(Ytrain_new)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.05)
    Xtr_more = get_more_images(Xtrain)
    Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))

    return (Xtr_more, Ytr_more), (Xtest, Ytest)
    # return Xtrain, Ytrain, Xtest, Ytest


def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        im = np.dstack((a, b, c))
        im = cv2.resize(im, (72, 72), interpolation=cv2.INTER_AREA)
        imgs.append(im)

    return np.array(imgs)


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def run_autocnn():
    # Loads the data as test and train
    file_path = 'D:/PythonProjects/itmo/AutoCNN/dataset_files/train.json'
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = from_json(file_path=file_path)

    # Puts the data in a dictionary for the algorithm to use
    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    roc = RocCallback(training_data=(x_train, y_train),
                      validation_data=(x_test, y_test))
    # Sets the wanted parameters
    a = AutoCNN(population_size=3, maximal_generation_number=5, dataset=data,
                epoch_number=5, loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=('accuracy', tf.keras.metrics.AUC(from_logits=True)))

    # Runs the algorithm until the maximal_generation_number has been reached
    best_cnn = a.run()
    print(best_cnn)


if __name__ == '__main__':
    run_autocnn()
