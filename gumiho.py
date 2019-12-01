from itertools import chain, islice
import string

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from vae import AutoEncoder, VariationalAutoEncoder, \
    ConvEncoder, ConvDecoder, \
    data_initializer


class GumihoNetwork(VariationalAutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tails = {}
        self.losses = {}
        self.add_tail(None, self._decoder, VariationalAutoEncoder.loss)

    def parameters(self):
        tails_params = (
            self.tails[key].parameters()
            for key in self.tails
            if key is not None  # because it is alread added
        )
        return chain(super().parameters(), *tails_params)

    def forward(self, X, *, tail):
        h = self.encode(X)
        z, μ, logσ = self.bottle(h)
        Y = self.decode(z, tail=tail)
        return Y, μ, logσ

    def decode(self, z, *, tail=None):
        return self.tails[tail](z)

    def generate(self, n, *, tail=None):
        z = self._sample(n)
        return self.decode(z, tail=tail)

    def add_tail(self, key, network, loss):
        self.tails[key] = nn.Sequential(self.η, network)
        self.losses[key] = loss

    def loss(self, X, *params, tail=None):
        return self.losses[tail](X, *params)


if __name__ == '__main__':
    bottle_size = 12
    data_shape = [28, 28, 1]
    h_size = 64

    EG = GumihoNetwork(
        h_size=h_size, z_size=bottle_size, data_shape=data_shape,
        EncoderType=ConvEncoder, DecoderType=ConvDecoder
    )
    Mirror = ConvDecoder(bottle_size=h_size, data_shape=data_shape)

    # def MirrorLoss(X, Y, *_):
    #     return EG.reconstruction_error(Y, X, size_average=False)

    EG.add_tail('mirror', Mirror, AutoEncoder.loss)

    parameters = EG.parameters()
    optimizer = torch.optim.Adam(parameters)

    typical_stream = data_initializer(data_shape=data_shape)

    Test = False
    epocs = 800 if not Test else 1
    batch_size = 400
    for _ in tqdm(range(epocs)):

        next(typical_stream)
        batch = typical_stream.send(batch_size)

        optimizer.zero_grad()
        Out = EG(batch, tail=None)
        J = EG.loss(batch, *Out, tail=None)
        J.backward()

        optimizer.step()

    for _ in tqdm(range(epocs)):

        next(typical_stream)
        batch = typical_stream.send(batch_size)

        optimizer.zero_grad()
        Out = EG(batch, tail='mirror')
        J = EG.loss(batch, *Out, tail='mirror')
        J.backward()

        optimizer.step()

    it = iter(string.ascii_uppercase)
    with torch.no_grad():
        for i in islice(it, 5):
            next(typical_stream)
            batch = typical_stream.send(1)
            img = EG(batch, tail=None)[0].squeeze().numpy()
            plt.imsave(f'./{i}-gumiho.png', img)

            next(typical_stream)
            batch = typical_stream.send(1)
            img = EG(batch, tail='mirror')[0].squeeze().numpy()
            plt.imsave(f'./{i}-mirror.png', img)

        for i in islice(it, 5):
            _ = EG.generate(1)
            plt.imsave(f'./{i}-generated.png', _[0].view((28, 28)).numpy())

            img = EG.forward(_, tail="mirror")[0]
            plt.imsave(f'./{i}-generated-mirror.png', img.view((28, 28)).numpy())
            img = EG.forward(_, tail=None)[0]
            plt.imsave(f'./{i}-generated-gumiho.png', img.view((28, 28)).numpy())
