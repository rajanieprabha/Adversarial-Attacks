## The following code was largely taken from

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request
from keras.preprocessing.image import ImageDataGenerator

class Dataset(object):

    def __init__(self):
        pass


class MNIST(Dataset):
    """The MNIST Dataset.
    """
    
    def __init__(self):

        # I/O dimensions
        self.image_size=28
        self.num_channels = 1
        self.num_labels = 10

        self.name = 'MNIST'

        # Loading data
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = self._extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = self._extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = self._extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = self._extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


    def getImageDataGenerator(self):
        return ImageDataGenerator(
                    rotation_range=0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    horizontal_flip=False)
            
    def _extract_data(self, filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(num_images*28*28)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = (data / 255) - 0.5
            data = data.reshape(num_images, 28, 28, 1)
            return data
        
    def _extract_labels(self, filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        return (np.arange(10) == labels[:, None]).astype(np.float32)


class CIFAR(Dataset):
    """The CIFAR Dataset.
    """
    
    def __init__(self):

        # I/O dimensions
        self.image_size= 32
        self.num_channels = 3
        self.num_labels = 10

        self.name = 'CIFAR'

        # Loading data
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = self._load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = self._load_batch("cifar-10-batches-bin/test_batch.bin")
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]
    
    def getImageDataGenerator(self):
        return ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)

    def _load_batch(self, fpath, label_key='labels'):
        f = open(fpath, 'rb')
        d = pickle.load(f, encoding="bytes")
        for k, v in d.items():
            del(d[k])
            d[k.decode("utf8")] = v
        f.close()
        data = d["data"]
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
        final[:,:,:,0] = data[:,0,:,:]
        final[:,:,:,1] = data[:,1,:,:]
        final[:,:,:,2] = data[:,2,:,:]

        final /= 255
        final -= .5
        labels2 = np.zeros((len(labels), 10))
        labels2[np.arange(len(labels2)), labels] = 1

    def _load_batch(self, fpath):
        f = open(fpath,"rb").read()
        size = 32*32*3+1
        labels = []
        images = []
        for i in range(10000):
            arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
            lab = np.identity(10)[arr[0]]
            img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

            labels.append(lab)
            images.append((img/255)-.5)
        return np.array(images),np.array(labels)