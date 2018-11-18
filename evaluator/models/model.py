## The following code was largely taken from https://github.com/carlini/nn_breaking_detection

import numpy as np
import keras
import tensorflow as tf

class RobustModel(object):
    # has a self.model
    # To have a test-dropout model, use RobustModel.get_test_dropout() as Dropout parameter

    def __init__(self):
        pass

    def predict(self, data):
        return self.model(data)

    @staticmethod
    def getTestDropout(dropout=True, fixed=False):
        def Dropout(p):
            if not dropout: 
                p = 0
            def my_dropout(x):
                if fixed:
                    shape = x.get_shape().as_list()[1:]
                    keep = np.random.random(shape)>p
                    return x*keep
                else:
                    return tf.nn.dropout(x, 1-p)
            return keras.layers.core.Lambda(my_dropout)

        return Dropout


class CarliniCifarWrapper:
    # this is a wrapper for models that are passed to the CW attacks
    def __init__(self, model):
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = 10
        self.model = model

    def predict(self, xs):
        return self.model(xs)