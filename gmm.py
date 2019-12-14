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


class ExpNormalizer:
    @classmethod
    def exp_norm(cls, affiliations):
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        dims = affiliations.shape[1]
        b, _ = affiliations.max(axis=1)
        b = b.view(-1, 1).repeat(1, dims)
        num = torch.exp(affiliations - b)  # y
        den = num.sum(axis=1).view(-1, 1).repeat(1, dims)

        if False:
            idx = torch.isnan(num)
            num[idx] = 0

        out = torch.div(num, den)
        return out


class GMM(nn.Module, ExpNormalizer):
    def __init__(self, count, dims):
        super().__init__()
        self.count = count
        concentration = torch.Tensor([1/count] * count)
        Dir = torch.distributions.dirichlet.Dirichlet(concentration)
        self.mixtures = nn.ModuleList(
            [Mixture(dims, phi=phi) for phi in Dir.sample()]
        )

    @property
    def epsilon(self):
        return (torch.rand(1) - 0.5) * EPSILON

    def mixed_nll(self, X):
        affiliations = self.forward(X)
        return -affiliations.sum(axis=1)

    def forward(self, X, *, normalize=True):
        num = t(torch.stack(
            [model(X, normalize=False) for model in self.mixtures]
        )).squeeze()
        if normalize:
            out = self.exp_norm(num)
        else:
            out = num
        return out

    def update(self, X, affiliations, show=True):
        for idx, model in enumerate(self.mixtures):
            if not model.update(X, affiliations[:, idx].unsqueeze(1)):
                raise "hell"
        if show:
            _phi = torch.Tensor([model.phi for model in self.mixtures])
            print('summary:', affiliations.argmax(dim=0))
            print('phi:    ', _phi)


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

    def forward(self, X):
        mm = partial(reduce, torch.mm)

        with torch.no_grad():
            dims = torch.Tensor([self.dims])
            log_two_pi_dims = dims * (
                torch.log(torch.Tensor([2.0]))
                + torch.log(torch.Tensor([math.pi]))
            )

            sigma = self.sigma + self.epsilon
            inv_sigma = torch.pinverse(sigma)
            log_det_sigma = sigma.logdet()
            log_den = 0.5 * (log_two_pi_dims + log_det_sigma)

            collect = []
            for x in X:
                diff = (x - self.mu).unsqueeze(0)
                _ = mm([-0.5 * diff, inv_sigma, diff.t()]) - torch.exp(log_den)
                collect.append(_)
            out = torch.cat(collect)
            return out

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
            _acc += l * torch.mm(_.t(), _)
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


class Mixture(Gaussian, ExpNormalizer):
    def __init__(self, dims, phi=torch.ones((1)), *args, **kwargs):
        super().__init__(dims, *args, **kwargs)
        self.phi = nn.Parameter(phi, requires_grad=False)

    def forward(self, X, *, normalize=True):
        _ = self.phi * super().forward(X)
        return self.exp_norm(_) if normalize else _

    def _update(self, X, affiliations, normalize=True):
        self.phi.data = affiliations.mean()
        _m, _s = super()._update(X, affiliations)
        return _m, _s


def test_gaussian(scale, iters, count, dims, std, m, key):
    mu = m[key][:dims]
    sigma = std[key]
    d = torch.distributions.MultivariateNormal(
        torch.Tensor(mu),
        torch.diag(torch.Tensor([sigma] * dims))
    )
    N = Gaussian(dims)
    X = d.sample((scale,))
    for i in tqdm(range(100)):
        p = N(X)
        if not N.update(X, p):
            print('break')
            break
    print(N.mu)
    print(N.sigma)


def test_mixture(scale, iters, count, dims, std, m, key):
    mu = m[key][:dims]
    sigma = std[key]
    d = torch.distributions.MultivariateNormal(
        torch.Tensor(mu),
        torch.diag(torch.Tensor([sigma] * dims))
    )
    N = Mixture(dims)
    X = d.sample((scale,))
    for i in tqdm(range(100)):
        p = N(X)
        if not N.update(X, p):
            break
    print(N.mu)
    print(N.sigma)
    print(N.phi)


def gmm_thing(X, iters, count, dims, std, m, Type=GMM):
    gmm = Type(count, dims)

    for _ in tqdm(range(iters)):
        score = gmm(X)
        gmm.update(X, score)
        yield gmm


def test_gmm(scale, iters, count, dims, std, m, *args):
    assert count == len(std) == len(m)

    dists = [
        torch.distributions.MultivariateNormal(
            torch.Tensor(m[:dims]),
            torch.diag(torch.Tensor([s] * dims))
        ) for m, s in zip(m, std)
    ]
    samples = [d.sample((scale * i,)) for i, d in enumerate(dists, 1)]
    X = torch.cat(samples)

    for gmm in gmm_thing(X, iters, count, dims, std, m):
        pass

    for idx, _ in enumerate(gmm.mixtures):
        print(
            gmm.mixtures[idx].mu.data,
            torch.diag(gmm.mixtures[idx].sigma).data,
            gmm.mixtures[idx].phi.data
        )

    print('----------')
    print(*m, sep='\n')
    print('----------')
    S = torch.tensor(m)
    p(S)
    return gmm


def main():
    scale = 500

    iters = 80
    count, dims = 3, 2
    std = [0.1, 0.1, 0.1]
    m = [(-1.7, -1.9, -1.9), (0.0, 0.0, -0.5), (1.9, 1.7, 1.9)]
    key = 1

    return test_gmm(scale, iters, count, dims, std, m, key)


if __name__ == '__main__':
    main()
