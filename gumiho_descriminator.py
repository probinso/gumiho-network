from collections import namedtuple
from itertools import islice
from math import floor
from pathlib import Path

from tqdm import tqdm
from tqdm import trange
import torch

import matplotlib.pyplot as plt

from gmm import GMM
from ae import ConvEncoder, ConvDecoder, Identity, Print
from vae import GaussianSample
from gumiho import GumihoNetwork, data_initializer

import psutil
import ray
ray.init(
    num_cpus=psutil.cpu_count() - 1,
    memory=8*1024*1024*1024,
    object_store_memory=3*1024*1024*1024
)
p = Print()
# torch.set_default_dtype(torch.double)


@ray.remote
def identity(x):
    return x


Phi = namedtuple('Phi', ['z', 'rho'])
AScore = namedtuple('AScore', ['encoding', 'anomaly_score'])


class CondGeneratorNetwork(GumihoNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conds = {}
        self.add_cond(None, self._none_cond)

    @classmethod
    def _none_cond(cls, x):
        return True

    def _sample(self, n, *, tail=None):
        def _local_gen():
            while True:
                z = GaussianSample(self.z_size)
                if self.conds[tail](z):
                    yield z

        g = islice(_local_gen(), n)
        f = [identity.remote(_) for _ in g]
        z = ray.get(f)
        Z = torch.stack(z)
        return Z

    def add_cond(self, key, cond):
        self.conds[key] = cond

    def generate(self, n, *, cond=None, tail=None):
        z = self._sample(n, tail=cond)
        return self._generate_from_z(z, tail=tail)


class DiscriminatorNetwork(CondGeneratorNetwork):
    def __init__(self, *, rho, mixtures, DecoderType, **kwargs):
        self.remaps = set()
        super().__init__(DecoderType=DecoderType, **kwargs)

        self.rho = rho
        self.phi = Phi(0.0, float('inf'))

        self.mixtures = mixtures
        self.MM = GMM(mixtures, self.z_size)
        self.add_tail(
            'MM',
            Identity(),
            self._MMLoss,
            remap=False
        )

        self._mirror = DecoderType(
            bottle_size=self.h_size,
            data_shape=self.data_shape
        )

        self.add_cond('disc', self._discriminator_cond_z)
        self.add_tail('disc', self._mirror, self._discriminator_loss)

    def _MMLoss(self, X, *args):
        z, *_ = args
        print(z)
        p(z)
        return self.MM(z)

    def decode(self, z, *, tail=None):
        if tail in self.remaps:
            return super().decode(z, tail=tail)
        else:
            return self.tails[tail](z)

    def add_tail(self, key, network, loss, remap=True):
        if remap:
            self.remaps.add(key)
            super().add_tail(key, network, loss)
        else:
            self.tails[key] = network
            self.losses[key] = loss

    def _discriminator_cond_z(self, z):
        with torch.no_grad():
            _ = self.MM.mixed_nll([z])
            return _ >= self.phi.z

    @classmethod
    def reconstruction_error(cls, Y, X, pairs=False):
        if not pairs:
            result = cls._reconstruction_error(Y, X)
        else:
            errors = [
                cls._reconstruction_error(y, x)
                for y, x in zip(Y.unbind(dim=0), X.unbind(dim=0))
            ]
            result = torch.stack(errors)
        return result

    def _discriminator_loss(self, X, *args):
        Y, gen_x, gen_y, *_ = args
        sfunc = torch.sigmoid

        tcc = self.reconstruction_error(Y, X, pairs=True)
        FAR = sfunc(tcc - self.phi.rho).mean()  # False Atypical Rate

        acc = self.reconstruction_error(gen_y, gen_x, pairs=True)
        TAR = sfunc(acc - self.phi.rho).mean()

        return FAR - TAR

    def _get_encoding_and_anomaly_score(self, X):
        with torch.no_grad():
            h = self.encode(X)
            z, *_ = self.bottle(h)
            Y = self.decode(z)

            score = self.reconstruction_error(Y, X, pairs=True)
            return z, score

    def _update_phi(self, X):
        # get rho is false positive rate
        M = X.size()[0]
        with torch.no_grad():
            print('starts', self.phi)
            idx = floor(self.rho * M)

            encodings, scores = self._get_encoding_and_anomaly_score(X)

            scores, _ = torch.sort(scores, descending=True)

            nlls = self.MM.mixed_nll(encodings)
            nlls, _ = torch.sort(nlls, descending=True)

            while True:
                try:
                    rho = scores[idx]  # descending order
                    z = nlls.data[idx][0]  # descending order
                except IndexError:
                    raise "No viable likelyhoods"
                if z == torch.Tensor([float("Inf")]):
                    print('drop inf')
                    idx += 1
                    continue
                if z != z:
                    print('drop nan')
                    idx += 1
                    continue

                self.phi = Phi(z, rho)
                print('sets', self.phi)
                break


if __name__ == '__main__':

    mixtures = 3
    bottle_size = 2
    data_shape = [28, 28, 1]
    h_size = 64

    typical_stream, various_stream = data_initializer(
        various=50, data_shape=data_shape
    )

    rho, M = .05, 300
    model = DiscriminatorNetwork(
        rho=rho,
        mixtures=mixtures, data_shape=data_shape,
        z_size=bottle_size, h_size=h_size,
        EncoderType=ConvEncoder, DecoderType=ConvDecoder
    )

    optimizer = torch.optim.Adam(model.parameters())  # , lr=1e-3)

    eras = 10
    epocs = 500
    batch_size = 400
    steps = 1
    i = 0

    test = False
    if test:
        eras = 2
        epocs = 2
        batch_size = 100
        steps = 1

    for era in range(eras):
        modelfile = Path(f'{era}.model')
        if modelfile.exists():
            model = torch.load(modelfile)
            print(f'loaded {era}')
            continue

        t = trange(epocs*(1 + 3*(not era)), desc='phase_one')
        for epoc in t:
            next(various_stream)
            batch = various_stream.send(batch_size)
            for step in range(steps):
                optimizer.zero_grad()
                Out = model(batch, tail=None)
                J = model.loss(batch, *Out, tail=None)
                t.set_description(
                    'phase_one (loss={:.4f}) {}'.format(
                        J,
                        '+' if not step else '-'
                    )
                )
                J.backward()
                optimizer.step()

        _ = model.generate(5, cond=None, tail=None)
        for i, img in enumerate(_, i):
            plt.imsave(f'./{i}-generated.png', img.view((28, 28)).numpy())

        if not era:
            # initialize gmm
            for epoc in tqdm(range((1-test)*20)):
                next(typical_stream)
                batch = typical_stream.send(batch_size)
                for step in range(steps):
                    with torch.no_grad():
                        Out = model(batch, tail='MM')
                        J = model.loss(batch, *Out, tail='MM')
                        z, *_ = Out
                        print(J)
                        print(z)
                        model.MM.update(z, J)

        print('_phase_two')
        t = trange(epocs, desc='phase_two')
        for epoc in t:
            next(typical_stream)
            batch = typical_stream.send(M)
            with torch.no_grad():
                model._update_phi(batch)
                count = batch.size()[0]
                gen_x = model.generate(count, cond='disc')
            for step in range(steps):
                optimizer.zero_grad()
                Y, *_ = model(batch, tail='disc')
                gen_y, *_ = model(gen_x, tail='disc')
                Out = Y, gen_x, gen_y
                J = model.loss(batch, *Out, tail='disc')
                t.set_description(
                    'phase_two (loss={:.4f})'.format(J)
                )
                J.backward()
                optimizer.step()

        _ = model.generate(5, cond='disc', tail=None)
        for i, img in enumerate(_, i):
            plt.imsave(f'./{i}-anomaly.png', img.view((28, 28)).numpy())

        print('_phase_three')
        for epoc in tqdm(range((1-test)*20)):
            next(typical_stream)
            batch = typical_stream.send(batch_size)
            for step in range(steps):
                with torch.no_grad():
                    Out = model(batch, tail='MM')
                    J = model.loss(batch, *Out, tail='MM')
                    z, *_ = Out
                    model.MM.update(z, J)

        torch.save(model, modelfile)
