import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Removes Tensorflow debuggin ouputs

import tensorflow as tf

tf.get_logger().setLevel('INFO') # Removes Tensorflow debugging ouputs

from auto_cnn.gan import AutoCNN

import random
import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
# Sets the random seeds to make testing more consisent
random.seed(12)
tf.random.set_seed(12)

class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
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
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return




def load_images(size=120, is_train=True):

    # file_path='C:/Users/aliev/Documents/GitHub/nas-fedot/10cls_Generated_dataset'
    # with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/labels_10.json', 'r') as fp:
    #     labels_dict = json.load(fp)
    # with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/encoded_labels_10.json', 'r') as fp:
    #     encoded_labels = json.load(fp)

    file_path='C:/Users/aliev/Documents/GitHub/nas-fedot/Generated_dataset'
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/encoded_labels.json', 'r') as fp:
        encoded_labels = json.load(fp)
    Xarr = []
    Yarr = []
    number_of_classes = 3
    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (size, size))
        Xarr.append(image)
        label_names = labels_dict[filename[:-4]]
        each_file_labels = [0 for _ in range(number_of_classes)]
        for name in label_names:
            num_label = encoded_labels[name]
            # each_file_labels.append(num_label)
            each_file_labels[num_label] = 1
        Yarr.append(each_file_labels)
    Xarr = np.array(Xarr)
    Yarr = np.array(Yarr)
    # Xarr = Xarr.reshape(-1, size, size, 1)

    return Xarr, Yarr


def load_patches():
    Xtrain, Ytrain = load_images(size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        ys = np.argmax(y)
        new_Ytrain.append(ys)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return (Xtrain, Ytrain), (Xval, Yval)


def run_autocnn():
    # Loads the data as test and train
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_patches()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    # Puts the data in a dictionary for the algorithm to use
    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    roc = RocCallback(training_data=(x_train, y_train),
                      validation_data=(x_test, y_test))
    # Sets the wanted parameters
    a = AutoCNN(population_size=3, maximal_generation_number=5, dataset=data,
                epoch_number=5, loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics = ('accuracy', tf.keras.metrics.AUC(from_logits=True)))

    # Runs the algorithm until the maximal_generation_number has been reached
    best_cnn = a.run()
    print(best_cnn)

if __name__ == '__main__':
    run_autocnn()