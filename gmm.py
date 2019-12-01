import math

import numpy as np

import torch
import torch.nn as nn
# Inspired by https://github.com/RomainSabathe/dagmm/blob/master/gmm.py


class GMM(nn.Module):
    def __init__(self, count, dims):
        super().__init__()
        self.count = count
        self.mixtures = nn.ModuleList(
            [Gaussian(dims) for _ in range(count)]
        )

    def probs(self, X):
        return torch.stack([model(X) for model in self.mixtures])

    @property
    def epsilon(self):
        return torch.rand(1) * 1e-9

    def mixed_nll(self, X):
        _ = self.probs(X) + self.epsilon
        _ = torch.log(_)
        return -1 * _.sum(axis=0)

    def forward(self, X):
        out = self.probs(X)
        return out / out.sum(0)
        # return (-torch.log(out) if log else out).squeeze()

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

        self.phi = nn.Parameter(torch.rand(1), requires_grad=False)
        self.mu = nn.Parameter(torch.rand(dims), requires_grad=False)
        self.mu.data = (self.mu.data - 0.5) * 2.0
        self.sigma = nn.Parameter(torch.eye(dims), requires_grad=False)

        #  self.epsilon = torch.rand((dims, dims)) * 1e-8

    @property
    def epsilon(self):
        return torch.rand((self.dims, self.dims)) * 1e-9

    def forward(self, X):
        with torch.no_grad():
            # inv_sigma = torch.pinverse(self.sigma + self.epsilon)
            sigma = torch.diag(self.sigma)
            det_sigma = torch.prod(sigma)

            if np.isnan(det_sigma):
                print(det_sigma)
            inv_sigma = torch.diag(1/sigma)
            # _ = torch.trace(1/sigma)
            # exit(1)

            # def det(tensor):
            #     _ = np.linalg.det(tensor.data.numpy())
            #     if np.isnan(_):
            #         print(_)
            #     return torch.Tensor([_])
            #
            # def solve(A, b):
            #     return torch.from_numpy(
            #         np.linalg.solve(A.data.numpy(), b.data.numpy())
            #     )

            # det_sigma = det(self.sigma + self.epsilon)
            dims = torch.Tensor([self.dims])

            out = torch.Tensor()
            collect = []
            for x in X:
                dist = torch.abs((x - self.mu)).view(-1, 1)
                # _ = solve(self.sigma, diff)
                _ = torch.mm(inv_sigma, dist)
                _ = torch.mm(-0.5 * dist.t(), _)
                _ = self.phi * torch.exp(_)
                _ = _ / torch.sqrt(
                    torch.pow(2.0 * math.pi, dims).float() * det_sigma
                )
                collect.append(_)
            out = torch.cat(collect)
            return out

    def _update(self, X, probs):
        """
        probs is the probability of x_i being represented by the current
        Gaussian model.
        """
        self.phi.data = probs.mean()

        psum = probs.sum()

        num = 0
        for x, gamma in zip(X, probs):
            num += x * gamma
        _m = num / psum

        acc = 0
        for x, gamma in zip(X, probs):
            diff = (x - self.mu).view(-1, 1)
            acc += gamma * torch.mm(diff, diff.view(1, -1))
        _s = acc / psum

        tst = torch.isnan(_s).sum() + torch.isnan(_m).sum()
        return _m, _s, tst

    def update(self, X, probs):
        with torch.no_grad():
            # print('update')
            _m, _s, tst = self._update(X, probs)
            if bool((tst > 0.0)):
                # print(_m, flush=True)
                # print(_s, flush=True)
                # print(tst, flush=True)
                return False

            # print(_m.sum(), _s.sum())
            self.mu.data = _m
            self.sigma.data = _s
            return True


def test_gmm():
    scale = 60

    count, dims = 3, 2
    std = [0.04, 0.2, 0.01]
    means = [(-.7, -.9), (0.0, 0.2), (.5, .7)]

    gmm = GMM(count, dims)

    dists = [
        torch.distributions.MultivariateNormal(
            torch.Tensor(m),
            torch.diag(torch.Tensor([s] * dims))
        ) for m, s in zip(means, std)
    ]
    samples = [d.sample((scale,)) for i, d in enumerate(dists, 1)]
    X = torch.cat(samples)

    # print(X.data)

    for _ in range(80):
        score = gmm(X)
        gmm.update(X, score)

        # a, b = torch.max(score, 0)

        # for i in b.unique():
        #     print(*[i, ':', int((i == b).sum())])
    print(gmm.mixtures[0].mu.data, torch.diag(gmm.mixtures[0].sigma).data)
    print(gmm.mixtures[1].mu.data, torch.diag(gmm.mixtures[1].sigma).data)
    print(gmm.mixtures[2].mu.data, torch.diag(gmm.mixtures[2].sigma).data)
    print('----------')
    print(*means, sep='\n')
    print('----------')
    print(gmm.probs(torch.Tensor(means)))

    print(gmm.mixed_nll(torch.tensor(means)))


if __name__ == '__main__':
    test_gmm()
