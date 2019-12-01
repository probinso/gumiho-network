from collections import namedtuple
from pathlib import Path
from itertools import count
import string

import torch
from matplotlib import pyplot as plt
import numpy as np

from ae import data_initializer
from gumiho_descriminator import GumihoDiscriminatorNetwork

Phi = namedtuple('Phi', ['z', 'rho'])
# Rates = namedtuple('Rates', ['TruePositive', 'FalsePositive', 'Loss'])
AScore = namedtuple('AScore', ['encoding', 'anomaly_score'])


if __name__ == '__main__':
    GumihoDiscriminatorNetwork

    mixtures = 10
    bottle_size = 8

    typical_stream, various_stream = data_initializer(various=1)

    for _ in count():
        modelfile = Path(f'{_}.model')
        if modelfile.exists():
            model = torch.load(modelfile)
            # model.__class__ = eval(model.__class__.__name__)
            print(f'{_} loaded')
        else:
            break

    with torch.no_grad():
        for key, X in zip(string.ascii_letters, various_stream):
            img = np.zeros((56, 28))
            i = 0
            img[i*28:(i+1)*28, :28] = X.view((28, 28)).numpy()
            i += 1
            *_, Y = model(X)
            R = model.reconstruction_error(X, Y)
            img[i*28:(i+1)*28, :28] = Y.view((28, 28)).numpy()
            plt.imsave(f'./{R}-{key}-ghdn.png', img)
