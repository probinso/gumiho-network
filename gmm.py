from functools import reduce, partial
import math

# import numpy as np

import torch
import torch.nn as nn
# Inspired by https://github.com/RomainSabathe/dagmm/blob/master/gmm.py

from tqdm import tqdm

from ae import Print, Transpose


# torch.set_default_dtype(torch.double)
EPSILON = 1e-15
p = Print()
t = Transpose(0, 1)


class TOPROB:
    # @classmethod
    # def aff_to_prob(cls, affiliations):
    #     p = torch.exp(affiliations)
    #     idx = torch.isnan(p)
    #     drop_count = idx.sum()
    #     if drop_count:
    #         print('dropped nans:', drop_count)
    #     p[idx] = EPSILON
    #     return p
    #
    # @classmethod
    # def norm_prob(cls, probs):
    #     return probs / probs.sum(axis=1)[:, None]

    @classmethod
    def exp_normalize(cls, affiliations):
        BLOCK = torch.ones_like(affiliations)
        values, _ = affiliations.max(dim=1, keepdims=True)
        b = BLOCK * values
        num = torch.exp(affiliations - b)
        den = BLOCK * num.sum(axis=1, keepdims=True)
        norm = torch.div(num, den)
        return norm


class GMM(nn.Module, TOPROB):
    def __init__(self, count, dims):
        super().__init__()
        self.count = count
        concentration = torch.Tensor([1/count] * count)
        Dir = torch.distributions.dirichlet.Dirichlet(concentration)
        self.mixtures = nn.ModuleList(
            [_Mixture(dims, phi=phi) for phi in Dir.sample()]
        )

    @property
    def epsilon(self):
        return (torch.rand(1) - 0.5) * EPSILON

    def mixed_nll(self, X):
        ll = self(X)
        return -ll.sum(axis=1)

    def forward(self, X):
        ll = t(torch.stack([model(X) for model in self.mixtures])).squeeze()
        return ll

    def update(self, X, log_affiliations, show=False):
        # prob = self.norm_prob(self.aff_to_prob(log_affiliations))
        prob = self.exp_normalize(log_affiliations)
        for idx, model in enumerate(self.mixtures):
            p = prob[:, idx].unsqueeze(1)
            if not model.update(X, p):
                raise "hell"
        if show:
            _phi = torch.Tensor([model.phi for model in self.mixtures])
            print('summary:', prob.argmax(dim=0))
            print('phi:    ', _phi)


class Gaussian(nn.Module, TOPROB):
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

    def nll(self, X):
        return -self(X)

    def prob(self, X):
        return self.aff_to_prob(self(X))

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

            log_sqrt_den = 0.5 * (log_two_pi_dims + log_det_sigma)

            collect = []
            for x in X:
                diff = (x - self.mu).unsqueeze(0)
                log_exp_num = mm([-0.5 * diff, inv_sigma, diff.t()])
                ll = log_exp_num - log_sqrt_den
                collect.append(ll)
            out = torch.cat(collect)
            return out

    def _update_m(self, X, affiliations):
        return (X * affiliations).sum(axis=0) / affiliations.sum()

    def _update_s(self, X, _m, affiliations):
        diff = (X - _m)
        _acc = []
        for s, g in zip(diff, affiliations):
            _ = s.unsqueeze(0)  # make transposable
            _ = g * torch.mm(_.t(), _)
            _acc.append(_)
        num = torch.stack(_acc).sum(axis=0)
        den = affiliations.sum()
        _s = num / den
        return _s

    def _update(self, X, affiliations):
        """
        probs is the probability of x_i being represented by the current
        Gaussian model.
        """
        if affiliations is None:
            count = X.size(0)
            affiliations = torch.ones((count, 1))
        _m = self._update_m(X, affiliations)
        _s = self._update_s(X, _m, affiliations)
        return _m, _s

    @classmethod
    def _isdumb(cls, tensor):
        return (torch.isnan(tensor) + torch.isinf(tensor)).sum()

    def update(self, X, affiliations=None):
        with torch.no_grad():
            _m, _s = self._update(X, affiliations)
            if self._isdumb(_m) + self._isdumb(_s):
                return False

            self.mu.data = _m
            self.sigma.data = _s
            return True


class _Mixture(Gaussian):
    def __init__(self, dims, phi=torch.ones((1)), *args, **kwargs):
        super().__init__(dims, *args, **kwargs)
        self.phi = nn.Parameter(phi, requires_grad=False)

    def forward(self, X, *, normalize=True):
        liklihoods = super().forward(X)
        return torch.log(self.phi) + liklihoods

    def _update(self, X, probs):
        self.phi.data = probs.mean().data
        return super()._update(X, probs)


def test_gaussian(scale, iters, count, dims, std, m, key):
    mu = m[key][:dims]
    sigma = std[key]
    d = torch.distributions.MultivariateNormal(
        torch.Tensor(mu),
        torch.diag(torch.Tensor([sigma] * dims))
    )
    N = Gaussian(dims)
    X = d.sample((scale,))
    N.update(X)
    print(N.mu)
    print(N.sigma)


def gmm_thing(X, iters, count, dims, std, m, Type=GMM):
    gmm = Type(count, dims)
    yield gmm

    for _ in tqdm(range(iters)):
        score = gmm(X)
        gmm.update(X, score)
        yield gmm


def test_gmm(scale, iters, count, dims, std, m, *args):
    assert count == len(std) == len(m)

    dists = [
        torch.distributions.MultivariateNormal(
            torch.Tensor(mu[:dims]),
            torch.diag(torch.Tensor([s] * dims))
        ) for mu, s in zip(m[:count], std)
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
    p(gmm.mixed_nll(X))
    return gmm


def main():
    scale = 500

    iters = 80
    count, dims = 3, 3
    std = [0.1, 0.1, 0.1]
    m = [(-1.7, -1.9, -1.9), (0.0, 0.0, -0.5), (1.9, 1.7, 1.9)]
    key = 0

    return test_gmm(scale, iters, count, dims, std, m, key)


if __name__ == '__main__':
    main()
