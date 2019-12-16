from gmm import GMM, gmm_thing as goer
import torch
import matplotlib.pyplot as plt


class Plotter(GMM):
    def plot(self, X):
        print(len(self.mixtures))
        ax = plt.gca()
        ax.cla()  # clear things for fresh plot
        for model, c in zip(self.mixtures, ('black', 'red', 'green', 'purple')):
            print(*model.mu, ',', torch.diag(model.sigma)[0])
            scatter = plt.scatter(*model.mu, c=c, alpha=.8)
            circle = plt.Circle(model.mu, torch.diag(model.sigma)[0], color=c, fill=False)
            ax.add_artist(circle)
            circle = plt.Circle(model.mu, torch.diag(model.sigma)[0]*2, color=c, fill=False)
            ax.add_artist(circle)
            ax.add_artist(scatter)

        plt.scatter(*X.t(), alpha=.1)
        plt.show()


if __name__ == '__main__':
    scale = 500

    iters = 80
    count, dims = 2, 2
    std = [0.1, 0.1, 0.01]
    m = [(-.8, -.8, -.4), (0.8, 0.8, -0.5), (.9, .7, .9)]

    dists = [
        torch.distributions.MultivariateNormal(
            torch.Tensor(mu[:dims]),
            torch.diag(torch.Tensor([s] * dims))
        ) for mu, s in zip(m[:count], std)
    ]
    samples = [d.sample((scale * 1,)) for i, d in enumerate(dists, 1)]
    X = torch.cat(samples)

    for gmm in goer(X, iters, count, dims, std, m, Type=Plotter):
        gmm.plot(X)
