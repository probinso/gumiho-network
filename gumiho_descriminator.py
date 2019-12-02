from collections import namedtuple
from itertools import islice
from math import floor
from pathlib import Path

from tqdm import tqdm
import torch

from ae import ConvEncoder, ConvDecoder, Identity
from vae import GaussianSample
from gumiho import GumihoNetwork, data_initializer
from gmm import GMM

# import psutil
# import ray
# ray.init(num_cpus=psutil.cpu_count() - 1)

from tqdm import trange

Phi = namedtuple('Phi', ['z', 'rho'])
AScore = namedtuple('AScore', ['encoding', 'anomaly_score'])


# @ray.remote
# def _get_anomaly_ray(phi_z, bottle_size, MM, decoder):
#     with torch.no_grad():
#         _ = torch.Tensor([float('-inf')])
#         for steps in count():
#             if _ >= phi_z:
#                 break
#             z = GaussianSample(bottle_size)
#             _ = MM.mixed_nll([z])
#
#         return decoder(z)


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

        Z = torch.stack([_ for _ in islice(_local_gen(), n)])
        return Z

    def add_cond(self, key, cond):
        self.conds[key] = cond

    def generate(self, n, *, cond=None, tail=None):
        z = self._sample(n, tail=cond)
        return self.decode(z, tail=tail)


class DiscriminatorNetwork(CondGeneratorNetwork):
    def __init__(self, *, rho, mixtures, DecoderType, **kwargs):
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
        return self.MM(z)

    def add_tail(self, key, network, loss, remap=True):
        if remap:
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
            return super().reconstruction_error(Y, X)

        errors = [
            cls._reconstruction_error(y, x)
            for y, x in zip(Y.unbind(dim=0), X.unbind(dim=0))
        ]
        return torch.stack(errors)

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
            idx = floor(self.rho * M)
            encodings, scores = self._get_encoding_and_anomaly_score(X)

            scores = sorted(scores, reverse=True)  # descending

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

                self.phi = Phi(z, rho)
                break


if __name__ == '__main__':

    mixtures = 20
    bottle_size = 12
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
    epocs = 300
    batch_size = 400
    steps = 3

    test = True
    if test:
        eras = 2
        epocs = 2
        batch_size = 10
        steps = 1

    for era in range(eras):
        modelfile = Path(f'{era}.model')
        if modelfile.exists():
            model = torch.load(modelfile)
            print(f'loaded {era}')
            continue

        t = trange(epocs*(1 + 3*(not era)), desc='phase_one')
        maxloss = -1
        for epoc in t:
            next(various_stream)
            batch = various_stream.send(batch_size)
            for step in range(steps):
                optimizer.zero_grad()
                Out = model(batch, tail=None)
                J = model.loss(batch, *Out, tail=None)
                if J > maxloss:
                    maxloss = J
                # print(J)
                t.set_description(
                    'phase_one (loss={:.8f}) {}'.format(
                        J / maxloss,
                        '+' if not step else '-'
                    )
                )
                J.backward()

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
                        model.MM.update(z, J)

        print('_phase_two')
        t = trange(epocs, desc='phase_two')
        maxloss = -1
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
                if J > maxloss:
                    maxloss = J
                t.set_description(
                    'phase_two (loss={:.8f})'.format(J / maxloss)
                )
                J.backward()

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
