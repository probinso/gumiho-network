# from collections import namedtuple

from functools import partial, reduce
from operator import mul
from itertools import islice, cycle, chain
from pathlib import Path
from random import randint
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from tqdm import tqdm

prod = partial(reduce, mul)


class BottleNetwork(nn.Module):
    def __init__(self, *, bottle_size, data_shape):
        super().__init__()
        self.bottle_size = bottle_size
        self.data_shape = data_shape

    def forward(self, *args, **kwargs):
        raise NotImplementedError('not implemented')


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, *self.shape)


class LinearEncoder(BottleNetwork):
    def __init__(self, *, bottle_size, data_shape):
        super().__init__(bottle_size=bottle_size, data_shape=data_shape)

        self.network = nn.Sequential(
            Flatten(),
            nn.Linear(prod(self.data_shape), 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, self.bottle_size))

    def forward(self, X):
        return self.network(X)


class Transpose(nn.Module):
    def __init__(self, idx=1, jdx=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = idx
        self.jdx = jdx

    def forward(self, input):
        return input.transpose(self.idx, self.jdx)


class Print(nn.Module):
    def forward(self, input):
        print(input.size())
        return input


class Identity(nn.Module):
    def forward(self, input):
        return input


class LinearDecoder(BottleNetwork):
    def __init__(self, *, bottle_size, data_shape):
        super().__init__(bottle_size=bottle_size, data_shape=data_shape)
        self.network = nn.Sequential(
            nn.Linear(self.bottle_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, prod(self.data_shape)),
            nn.Sigmoid(),
            UnFlatten(self.data_shape)
        )

    def forward(self, X):
        return self.network(X)


class ConvEncoder(BottleNetwork):
    def __init__(self, *, bottle_size, data_shape):
        super().__init__(bottle_size=bottle_size, data_shape=data_shape)

        self.network = nn.Sequential(
            Transpose(),
            nn.Conv2d(self.data_shape[-1], 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(128, self.bottle_size),
        )

    def forward(self, X):
        return self.network(X)


class ConvDecoder(BottleNetwork):
    def __init__(self, *, bottle_size, data_shape):
        super().__init__(bottle_size=bottle_size, data_shape=data_shape)

        self.network = nn.Sequential(
            nn.Linear(self.bottle_size, 128),
            UnFlatten((128, 1, 1)),
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, data_shape[-1], kernel_size=2, stride=1),
            nn.Sigmoid(),
            Transpose()
        )

    def forward(self, X):
        return self.network(X)


class AutoEncoder(BottleNetwork):
    _reconstruction_error = partial(F.mse_loss, size_average=False)
    # _reconstruction_error = partial(
    #     F.binary_cross_entropy, size_average=False
    # )

    def __init__(
        self, *, bottle_size, data_shape, EncoderType, DecoderType,
        encoder=None
    ):
        super().__init__(bottle_size=bottle_size, data_shape=data_shape)
        self._encoder = encoder if encoder \
            else EncoderType(bottle_size=bottle_size, data_shape=data_shape)

        self._decoder = DecoderType(
            bottle_size=bottle_size, data_shape=data_shape
        )

    def parameters(self):
        return chain(self._encoder.parameters(), self._decoder.parameters())

    def encode(self, X):
        return self._encoder(X)

    def decode(self, z):
        return self._decoder(z)

    def forward(self, X):
        h = self.encode(X)
        Y = self.decode(h)
        return Y

    @classmethod
    def reconstruction_error(cls, Y, X):
        return cls._reconstruction_error(Y, X)

    @classmethod
    def loss(cls, X, *args):
        Y, *_ = args
        return cls.reconstruction_error(Y, X)


def data_initializer(*,
    censored=[2, 3, 4, 5],
    atypical=[1, 7],
    various=0,
    data_shape
):
    HERE = Path(".")
    _ = torch.utils.data.DataLoader(
        datasets.MNIST(
            HERE,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,))
            ])),
        shuffle=True
    )

    typical_stream = cycle(
        x.squeeze()
        for x, y in _
        if not (y in censored or y in atypical)
    )

    def to_batch(iterable):
        while True:
            n = (yield)
            batch = torch.stack(
                [x.view(data_shape) for x in islice(iterable, n)]
            )
            yield batch

    if not various:
        return to_batch(typical_stream)
    else:
        various_stream = cycle(
            x.squeeze()
            for x, y in _
            if (y in censored)
            or (y in atypical and not randint(0, various))
            or (y not in censored and y not in atypical)
        )
        return to_batch(typical_stream), to_batch(various_stream)


if __name__ == '__main__':
    bottle_size = 12
    data_shape = [28, 28, 1]

    EG = AutoEncoder(
        bottle_size=bottle_size, data_shape=data_shape,
        EncoderType=ConvEncoder, DecoderType=ConvDecoder)

    parameters = EG.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    typical_stream = data_initializer(data_shape=data_shape)

    epocs = 300
    batch_size = 400
    for _ in tqdm(range(epocs)):
        optimizer.zero_grad()
        next(typical_stream)
        batch = typical_stream.send(batch_size)
        Y = EG(batch)
        J = EG.loss(batch, Y)
        J.backward()
        optimizer.step()

    it = iter(string.ascii_uppercase)
    with torch.no_grad():
        for i in islice(it, 5):
            next(typical_stream)
            batch = typical_stream.send(batch_size)

            img = EG(batch).squeeze().numpy()
            plt.imsave(f'./{i}-ae.png', img)
