from functools import reduce, partial
import math

# import numpy as np

import torch
import torch.nn as nn
# Inspired by https://github.com/RomainSabathe/dagmm/blob/master/gmm.py

from tqdm import tqdm

from ae import Print, Transpose


torch.set_default_dtype(torch.double)
EPSILON = 1e-15
p = Print()
t = Transpose(0, 1)

scale = 5000

count, dims = 3, 3
std = [0.02, 1, 0.1]
means = [(-.7, .9, .2), (0.0, 0.0, -.2), (.5, .6, .9)]


class GMM(nn.Module):
    def __init__(self, count, dims):
        super().__init__()
        self.count = count
        self.mixtures = nn.ModuleList(
            [Mixture(dims) for _ in range(count)]
        )

    @property
    def epsilon(self):
        return (torch.rand(1) - 0.5) * EPSILON

    # def mixed_nll(self, X):
    #     # _ = self._probs(X)
    #     # _ = torch.log(_)
    #     # return -1 * _.sum(axis=0)
    #     _ = self._prob(X)
    #     return -torch.log(_).sum(0)

    def _probs(self, X, *, norm=False):
        num = t(torch.stack(
            [model(X) for model in self.mixtures]
        )).squeeze()
        if norm:
            _, dims, *_ = num.size()
            den = torch.max(num, 1)[0].view(-1, 1).repeat(1, dims)
            return torch.div(num, den)
        return num

    def forward(self, X, *, nll=True):
        return self._probs(X, nll)

    def update(self, X, probs):
        for model, phi in zip(self.mixtures, probs):
            model.update(X, phi.squeeze())

    def epoc(self, X):
        y = self(X)
        self.update(X, y)


class Gaussian(nn.Module):
    def __init__(self, dims):
        """
        networks will not require gradients, because we will not use
        `.update()` to propogate updates
        """
        super().__init__()
        self.dims = dims

        self.mu = (
            nn.Parameter(
                torch.rand(dims), requires_grad=False) - 0.5) * 2.0
        self.sigma = nn.Parameter(
            torch.eye(dims),
            requires_grad=False
        )

    @property
    def epsilon(self):
        return (
            torch.rand(
                (self.dims, self.dims)
            ) - 0.5) * EPSILON

    def forward(self, X, *, nll=True):
        mm = partial(reduce, torch.mm)

        with torch.no_grad():
            dims = torch.Tensor([self.dims])
            log_two_pi_dims = dims * (
                torch.log(torch.Tensor([2.0])) +
                torch.log(torch.Tensor([math.pi]))
            )

            sigma = self.sigma + self.epsilon
            inv_sigma = torch.pinverse(sigma)

            log_det_sigma = sigma.logdet()

            log_den = 0.5 * (log_two_pi_dims + log_det_sigma)

            collect = []

            for x in X:
                diff = (x - self.mu).unsqueeze(0)
                _ = mm([-0.5 * diff, inv_sigma, diff.t()]) - log_den
                collect.append(_)
            out = torch.cat(collect)
            return -out if nll else torch.exp(out)

    def _update(self, X, likelihoods):
        """
        probs is the probability of x_i being represented by the current
        Gaussian model.
        """
        normalizer = likelihoods.sum()
        _m = (X * likelihoods).sum(axis=0) / normalizer

        diff = X - _m
        _acc = 0
        for s, l in zip(diff, likelihoods):
            _ = s.unsqueeze(0)
            _acc += torch.mm(_.t(), _) * l
        _s = _acc / normalizer
        return _m, _s

    @classmethod
    def _isdumb(cls, tensor):
        return (torch.isnan(tensor) + torch.isinf(tensor)).sum()

    def update(self, X, likelihoods):
        with torch.no_grad():
            _m, _s = self._update(X, likelihoods)
            if self._isdumb(_m) + self._isdumb(_s):
                return False

            self.mu.data = _m
            self.sigma.data = _s
            return True


class Mixture(Gaussian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi = nn.Parameter(torch.rand(1), requires_grad=False)

    def forward(self, X, *, nll=True):
        _ = super().forward(X, nll=nll)
        return -(torch.log(self.phi) - _) if nll else self.phi * _

    def _update(self, X, likelihoods):
        self.phi.data = likelihoods.mean()
        _m, _s = super()._update(X, likelihoods)
        return _m, _s


def test_gaussian(key):
    mu = means[key]
    sigma = std[key]
    d = torch.distributions.MultivariateNormal(
        torch.Tensor(mu),
        torch.diag(torch.Tensor([sigma] * dims))
    )
    N = Gaussian(dims)
    for i in tqdm(range(100)):
        X = d.sample((scale,))
        p = N(X)
        if not N.update(X, p):
            print('break')
            break
    print(N.mu)
    print(N.sigma)


def test_mixture(key):
    mu = means[key]
    sigma = std[key]
    d = torch.distributions.MultivariateNormal(
        torch.Tensor(mu),
        torch.diag(torch.Tensor([sigma] * dims))
    )
    N = Mixture(1, dims)
    X = d.sample((scale,))
    for i in tqdm(range(100)):
        p = N(X)
        if not N.update(X, p):
            break
    print(N.mu)
    print(N.sigma)
    print(N.phi)


def test_gmm():
    assert count == len(std) == len(means)
    assert dims == len(means[0])

    gmm = GMM(count, dims)

    dists = [
        torch.distributions.MultivariateNormal(
            torch.Tensor(m),
            torch.diag(torch.Tensor([s] * dims))
        ) for m, s in zip(means, std)
    ]
    samples = [d.sample((scale * i,)) for i, d in enumerate(dists, 1)]
    X = torch.cat(samples)

    for _ in tqdm(range(80)):
        score = gmm(X)
        gmm.update(X, score)

    for idx, _ in enumerate(gmm.mixtures):
        print(
            gmm.mixtures[idx].mu.data,
            torch.diag(gmm.mixtures[idx].sigma).data,
            gmm.mixtures[idx].phi.data
        )

    print('----------')
    print(*means, sep='\n')
    print('----------')
    MEANS = torch.tensor(means)
    p(MEANS)
    print(gmm._probs(torch.Tensor(means)))

    print(gmm.mixed_nll(torch.tensor(means)))


if __name__ == '__main__':
    test_gmm()
