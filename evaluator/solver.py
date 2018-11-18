## The following code was largely taken from https://github.com/carlini/nn_breaking_detection

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

class Solver(object):

    def __init__(self, robustModel, dataset):

        self.robustModel = robustModel
        self.dataset = dataset

    def train(self,
              file_name,
              num_epochs=50,
              batch_size=128,
              init=None):

        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted)

        print(self.robustModel.model.summary())

        def get_lr(epoch):
            return base_lr*(.5**(epoch/num_epochs*10))
        sgd = SGD(lr=0.00, momentum=0.9, nesterov=False)
        schedule= LearningRateScheduler(get_lr)

        self.robustModel.model.compile(loss=fn,
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        datagen = self.dataset.getImageDataGenerator()
        base_lr = 0.1

        datagen.fit(self.dataset.train_data)

        self.robustModel.model.fit_generator(
                            datagen.flow(
                                self.dataset.train_data,
                                self.dataset.train_labels,
                                batch_size=batch_size),
                            steps_per_epoch=self.dataset.train_data.shape[0] // batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            validation_data=(self.dataset.validation_data, self.dataset.validation_labels),
                            callbacks=[schedule])

        print('Test accuracy:',
            np.mean(np.argmax(
                    self.robustModel.model.predict(self.dataset.test_data),axis=1
                )==np.argmax(
                    self.dataset.test_labels,axis=1
                )
            )
        )

        if file_name != None:
            self.robustModel.model.save_weights(file_name)

        return

    def attack(self, attack, nExamples=100):
        """Adv. examples stored in "data/adversarial/{att}_{ds}.npy"
        """

        adversarialExamples = attack.attack(
            ds.test_data[:nExamples],
            get_labs(ds.test_data[:nExamples])
        )
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists("data/adversarial"):
            os.mkdir("data/adversarial")

        np.save(
            'data/adversarial/{}_{}.npy'.format(
                attack.name,
                self.dataset.name
            ),
            adversarialExamples
        )

    def evaluate(self, defense, attack):
        """Evaluate defense against attack
        """

        advPath = 'data/adversarial/{}_{}.npy'.format(
                attack.name,
                self.dataset.name
        )

        if not os.path.exists(advPath):
            self.attack(attack, nExamples=100)

        adversarialExamples = np.load(advPath)


    