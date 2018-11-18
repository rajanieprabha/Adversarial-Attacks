import numpy as np
import tensorflow  as tf
from scipy.spatial.distance import cdist
import warnings
import keras.backend as K
import tqdm as tqdm

# BEGIN --- inspired by: https://github.com/lsgos/uncertainty-adversarial-paper/blob/master/src/utilities.py#L302-L332
def entropy(x):
    #
    return np.sum(- x * np.log(np.clip(x, 1e-30, 1)), axis=-1)


def expected_entropy(x):
    """
    Take a tensor of MC predictions [#images x #MC x #classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """

    return np.mean(entropy(x), axis=-1)


def predictive_entropy(x):
    """
    Take a tensor of MC predictions [#images x #MC x #classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    return entropy(np.mean(x, axis=1))


# END --- inspired by: https://github.com/lsgos/uncertainty-adversarial-paper/blob/master/src/utilities.py#L302-L332

class DropoutUncertainty:
    def __init__(self, model, sess):
        """
        :param sess: TF session
        :param model: expect MC dropout ready model without softmax
        """
        self.model = model
        self.sess = sess

    def __call__(self, inputs, mc_passes):
        # When the instance is called, MI should be computed
        self.outputs = np.array(
            list(zip(*[self.sess.run(tf.nn.softmax(self.model.predict(inputs))) for _ in range(mc_passes)])))
        print("Predicted outputs with shape {} and {} MC passes".format(self.outputs.shape, mc_passes))
        return self.outputs


class MutualInformation(DropoutUncertainty):
    def __init__(self, model, sess):
        """
        :param sess: TF session
        :param model: expect MC dropout ready model without softmax
        """
        super().__init__(model, sess)

    def __call__(self, inputs, mc_passes):
        # When the instance is called, MI should be computed
        outputs = super().__call__(inputs, mc_passes)
        mi = self.mutual_information(outputs)

        return mi

    def mutual_information(self, x):
        """
        Take a tensor of MC predictions [#images x #MC x #classes] and return the
        mutual information for each image
        """
        mi = predictive_entropy(x) - expected_entropy(x)
        negative_mi = mi[mi < 0]
        if negative_mi != []: warnings.warn(
            "negative values in mutual information, shape: {}".format(negative_mi.shape), Warning)
        mi = np.clip(mi, 0, None)
        return mi


class SoftmaxVariance(DropoutUncertainty):
    def __init__(self, model, sess):
        """
        :param sess: TF session
        :param model: expect MC dropout ready model without softmax
        """
        super().__init__(model, sess)

    def __call__(self, inputs, mc_passes):
        # When the instance is called, SV should be computed
        outputs = super().__call__(inputs, mc_passes)
        sv = self.softmax_variance(outputs)

        return sv

    def softmax_variance(self, x):
        # inspired by: https://github.com/carlini/nn_breaking_detection/blob/master/dropout_detect.py#L292-L303
        term1 = np.mean(np.sum(x ** 2, axis=2), axis=1)

        term2 = np.sum(np.mean(x, axis=1) ** 2, axis=1)

        print('absolute mean uncertenty', np.mean(term1 - term2))

        return term1 - term2


class LID():
    def __init__(self, model,sess,  X_test, X_adv, k, batch_size):
        """
        :param sess: TF session
        :param model: expect MC dropout ready model without softmax
        : k : k-nearest neighbors
        : batch_size
        """
        self.model = model
        self.k =k
        self.X_test = X_test
        self.X_adv = X_adv
        self.batch_size = batch_size
        self.nb_layers = len(self.model.layers)

    def lid_scores(self,layers=None):
        """
        :param layers: list of layers for which the LID scores should be computed
        :return: lid scores for X_test and X_adv
        """
        # inspired by: https://github.com/xingjunm/lid_adversarial_subspace_detection/extract_characteristics.py
        lids_normal, lids_adv = self.get_lids_random_batch(layers)

        return lids_normal,lids_adv

    def get_layer_wise_activations(self,layers):
        # enable negative indexing
        layers = [self.nb_layers + l if l < 0 else l for l in layers] if layers else list(range(self.nb_layers))
        outputs = [layer.output for i, layer in enumerate(self.model.layers) if i in layers]
        return K.function([self.model.layers[0].input, K.learning_phase()], outputs)

    # lid of a batch of query points X
    def mle_batch(self, data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)

        k = min(k, len(data) - 1)
        f = lambda v: - k / np.sum(np.log(v / v[-1]))
        #  Compute euclidean distance between each pair of the two collections of inputs.
        a = cdist(batch, data)                                          # shape: (batch_size, batch_size)
        # Get k nearest neighbors
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]     # shape: (batch_size, k)
        # Calculate LID score
        lid = np.apply_along_axis(f, axis=1, arr=a)                     # shape: (batch_size,)
        return lid


    def get_lids_random_batch(self,layers):
        """
        Get the local intrinsic dimensionality of each Xi in X_adv
        estimated by k close neighbours in the random batch it lies in.
        :param layers: list of layer indices
        :param model:
        :param X: normal images
        :param X_adv: advserial images
        :param k: the number of nearest neighbours for LID estimation
        :param batch_size: default 100
        :return: lids: LID of normal images of shape (num_examples, lid_dim)
                lids_adv: LID of advs images of shape (num_examples, lid_dim)
        """
        lid_dim = len(layers) if layers else self.nb_layers
        print("Number of layers to estimate: ", lid_dim)

        get_activations = self.get_layer_wise_activations(layers)

        def estimate(i_batch):
            # setup indices and initialize arrays
            start = i_batch * self.batch_size
            end = np.minimum(len(self.X_test), (i_batch + 1) * self.batch_size)
            n_feed = end - start
            # for the last batch, start to add samples from the beginning of the dataset to fill it up
            idx = np.arange(start, end)

            if n_feed != self.batch_size:
                from_beginning = np.arange(self.batch_size-n_feed)
                idx = np.append(idx,from_beginning)

            lid_batch = np.zeros(shape=(n_feed, lid_dim))
            lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))

            # get all activations
            acts = get_activations([self.X_test[idx], 0])
            adv_acts = get_activations([self.X_adv[idx], 0])

            # iterate over all layers
            for i, (act,adv_act) in enumerate(zip(acts,adv_acts)):
                X_act = np.asarray(act, dtype=np.float32).reshape((self.batch_size, -1))
                X_adv_act = np.asarray(adv_act, dtype=np.float32).reshape((self.batch_size, -1))

                # Maximum likelihood estimation of local intrinsic dimensionality (LID)
                lid_batch[:, i] = self.mle_batch(X_act, X_act, k=self.k)[:n_feed]
                # print("lid_batch: ", lid_batch.shape)
                lid_batch_adv[:, i] = self.mle_batch(X_act, X_adv_act, k=self.k)[:n_feed]
                # print("lid_batch_adv: ", lid_batch_adv.shape)
            return lid_batch, lid_batch_adv

        lids = []
        lids_adv = []
        n_batches = int(np.ceil(self.X_test.shape[0] / float(self.batch_size)))
        for i_batch in range(n_batches):
            # print("Batch ",i_batch+1)
            lid_batch, lid_batch_adv = estimate(i_batch)
            lids.extend(lid_batch)
            lids_adv.extend(lid_batch_adv)
        #print("lids_normal: ", lids.size)
        #print("lids_adv: ", lids_adv.size)
        #print(lids)

        lids = np.asarray(lids, dtype=np.float32)
        lids_adv = np.asarray(lids_adv, dtype=np.float32)

        return lids, lids_adv




