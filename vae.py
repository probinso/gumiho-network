# from collections import namedtuple

import string
from itertools import islice, chain

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from tqdm import tqdm

from ae import ConvEncoder, ConvDecoder, AutoEncoder, data_initializer


def GaussianSample(bottle_size):
    return torch.distributions.multivariate_normal.\
        MultivariateNormal(
            torch.zeros(bottle_size),
            torch.eye(bottle_size)
        ).sample()


class VariationalAutoEncoder(AutoEncoder):
    def __init__(
        self, *, h_size, z_size, data_shape,
        EncoderType, DecoderType, encoder=None
    ):
        super().__init__(
            bottle_size=h_size, data_shape=data_shape,
            EncoderType=EncoderType, DecoderType=DecoderType, encoder=encoder)
        self.z_size = z_size
        self.h_size = h_size

        self.μ = nn.Linear(h_size, z_size)
        self.logσ = nn.Linear(h_size, z_size)
        self.η = nn.Linear(z_size, h_size)

    def parametrs(self):
        return chain(
            super().parameters(),
            self.logσ.parameters(),
            self.η.parameters(),
            self.μ.paramerters()
        )

    @classmethod
    def _reparameterize(cls, μ, logσ):
        std = logσ.mul(0.5).exp_()
        # return torch.normal(mu, std)
        ε = torch.randn_like(μ)
        z = μ + std * ε
        return z

    def bottle(self, h):
        μ = self.μ(h)
        logσ = self.logσ(h)
        z = self._reparameterize(μ, logσ)
        return z, μ, logσ

    def decode(self, z):
        h = self.η(z)
        return self._decoder(h)

    def _sample(self, n):
        z = torch.stack([GaussianSample(self.z_size) for _ in range(n)])
        return z

    def generate(self, n=1):
        z = self._sample(n)
        h = self.η(z)
        return self._decoder(h)

    def forward(self, X):
        h = self.encode(X)
        z, μ, logσ = self.bottle(h)
        Y = self.decode(z)
        return Y, μ, logσ

    @classmethod
    def _KL_loss(cls, μ, logσ):
        return -0.5 * torch.mean(1 + logσ - μ.pow(2) - logσ.exp())

    @classmethod
    def loss(cls, X, *args):
        Y, μ, logσ, *_ = args
        return cls._KL_loss(μ, logσ) + cls.reconstruction_error(Y, X)


class DisentangledVariationalAutoencoder(VariationalAutoEncoder):
    pass


if __name__ == '__main__':
    z_size = 12
    h_size = 64
    data_shape = [28, 28, 1]

    EG = VariationalAutoEncoder(
        h_size=h_size, z_size=z_size, data_shape=data_shape,
        EncoderType=ConvEncoder, DecoderType=ConvDecoder
    )

    parameters = EG.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    typical_stream = data_initializer(data_shape=data_shape)

    Test = True
    epocs = 300 if not Test else 1
    batch_size = 400
    for _ in tqdm(range(epocs)):
        optimizer.zero_grad()
        next(typical_stream)
        batch = typical_stream.send(batch_size)
        Out = EG(batch)
        J = EG.loss(batch, *Out)
        J.backward()
        optimizer.step()

    it = iter(string.ascii_uppercase)
    with torch.no_grad():
        for i in islice(it, 5):
            next(typical_stream)
            batch = typical_stream.send(batch_size)

            img = EG(batch)[0].squeeze().numpy()
            # plt.imsave(f'./{i}-vae.png', img)

        for i in islice(it, 5):
            img = EG.generate(1).squeeze().numpy()
            plt.imsave(f'./{i}-vae.png', img)
