## The following code was largely taken from https://github.com/carlini/nn_breaking_detection

import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append('../')
sys.path.append(os.path.dirname(__file__))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import load_model

from model import RobustModel
from dataset import MNIST

class CNN(RobustModel):

    def __init__(self,
                 dataset,
                 restore=None,
                 bayesian=False):
        global Dropout

        # I/O Dimensions
        self.image_size = dataset.image_size
        self.num_channels = dataset.num_channels
        self.num_labels = dataset.num_labels

        if bayesian:
          Dropout = RobustModel.getTestDropout()

        model = Sequential()

        nb_filters = 64
        layers = [Conv2D(nb_filters, (5, 5), strides=(2, 2), padding="same",
                         input_shape=(self.image_size, self.image_size, self.num_channels)),
                  Activation('relu'),
                  Conv2D(nb_filters, (3, 3), strides=(2, 2), padding="valid"),
                  Activation('relu'),
                  Conv2D(nb_filters, (3, 3), strides=(1, 1), padding="valid"),
                  Activation('relu'),
                  Flatten(),
                  Dense(32),
                  Activation('relu'),
                  Dropout(.5),
                  Dense(self.num_labels)]

        for layer in layers:
            model.add(layer)

        if restore != None:
            model.load_weights(restore)
        
        self.model = model


if __name__=='__main__':
  model = CNN(MNIST())