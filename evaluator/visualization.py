import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.metrics import roc_curve
import pandas as pd
from ggplot import *

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA

def histogram(data, names, stacked=False, normed=False):
    assert len(data) == len(names)

    nb_hists = len(data)
    data = np.array(data)

    # under the assumption that negative values don't make much sense for a measure, we clip them ...
    # TODO: add assertion for negative values?
    data = [np.clip(x, 1e-30,None) for x in data]
    log_min = np.min([np.log10(np.min(x)) - 1 for x in data])
    log_max = np.max([np.log10(np.max(x)) + 1 for x in data])
    bins = np.logspace(log_min, log_max, 50)
    data = [x for x in data]

    plt.figure(figsize=(8, 3))
    plt.xscale('log')
    if not stacked:
        for i in range(nb_hists):
            crr_data = data[i]
            name = names[i]
            plt.hist(crr_data,
                     bins=bins,
                     label='{}: ${}$ samples'.format(name,crr_data.shape),
                     alpha=0.7)
    else:
        plt.hist(data,
                 bins=bins,
                 label=names,
                 stacked=stacked)
    plt.xlabel('value of detection measure')
    plt.ylabel('# occurrences')
    plt.legend()

    return plt.gcf()


def roc(data,names):
    assert len(data) == len(names)

    plt.figure(figsize=(7, 7))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    for i, (adv, real) in enumerate(data):
        adv_labels = [1] * len(adv)
        real_labels = [0] * len(real)
        labels = np.array(adv_labels + real_labels)
        samples = np.concatenate([adv, real])

        lw = 2
        fpr, tpr, _ = roc_curve(labels, samples, pos_label=1)
        plt.plot(fpr, tpr, lw=lw, label=names[i])
        plt.plot([0, 1], [0, 1], color='lightgray', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Detection Measure')
    plt.legend(loc="lower right")

    return plt.gcf()


def individual(data,names,N=None):
    assert len(data) < 4
    assert all(len(values) == len(data[0]) for values in data)

    if not N: N = len(data[0])
    # plt.title("Entropy: {} attack with eps {}".format(attack_spec['method'], attack_spec['eps']))
    plt.figure(figsize=(10,10))
    plt.xlabel('index')
    plt.ylabel('measure value')

    symbols = ['bo','r*','g.']
    for i, values in enumerate(data):
        plt.plot(values[:N], symbols[i], label=names[i])
    plt.vlines(range(N),ymin=data[0][:N],ymax=data[1][:N])
    plt.legend()

    return plt.gcf()


def images(adv_ims,real_ims=None,titles=None):
    fig = plt.figure(figsize=(5,20))

    if titles is None:
        titles = ['Images A','Images B']

    adv = fig.add_subplot(1,2,1)
    plt.imshow(np.concatenate(adv_ims,axis=0))
    plt.axis('off')
    adv.set_title(titles[0])

    if real_ims:
        real = fig.add_subplot(1, 2, 2)
        plt.imshow(np.concatenate(real_ims, axis=0))
        plt.axis('off')
        real.set_title(titles[1])

    return plt.gcf()

def PCAnalysis(x,labels,samples):
    feat_cols = [ 'pixel'+str(i) for i in range(x.shape[1]) ]

    df = pd.DataFrame(x,columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))

    X, y = None, None

    rndperm = np.random.permutation(df.shape[0]) # randomization
    pca = PCA(n_components = 3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    chart = ggplot( df.loc[rndperm[:samples],:], aes(x='pca-one', y='pca-two', color='label') ) \
            + geom_point(size=20,alpha=0.8) \
            + ggtitle("First and Second Principal Components colored by digit")
    return chart


