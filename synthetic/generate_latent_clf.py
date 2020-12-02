"""
Generate data triples (x, y, z) for deterministic classification p(y | x; z)

Generative story:

    Given: n_clusters; for each cluster:
      - a cluster center (mean) center[z]
      - a linear model y=sign(w[z] * x + b[z])

    pick z from uniform Categorical(n_clusters)
    pick cluster center c = center[z]
    sample x from N(mu[z], sigma * I)
    generate deterministic y = sign(w[z] * x + b[z])

"""

# author: vlad niculae <vlad@vene.ro>
# license: bsd 2 clause

import torch


def make_latent_triples(n_samples, n_features, n_clusters, centers=None, W=None, b=None, data_std=.1,
                        cluster_std=1):
    # generate cluster centers
    if centers is None:
        centers = cluster_std * torch.randn(n_clusters, n_features)

    # generate a linear model for each cluster
    if W is None:
        W = torch.randn(n_clusters, n_features)

    # draw cluster assignments
    z = torch.randint(low=0, high=n_clusters, size=(n_samples,))

    # draw data X
    c_ = centers[z]
    X = c_ + data_std * torch.randn(n_samples, n_features)

    # choose linear model to use for each sample
    W_ = W[z]

    # compute true label y
    y_score = (W_ * X).sum(dim=-1)

    # pick a threshold for each class
    # (note: this is done like this to ensure there are always roughly balanced
    # positive and negative samples in each class)
    if b is None:
        b = torch.zeros(n_clusters)
    for c in range(n_clusters):
        b[c] = y_score[z == c].mean()

    y = torch.sign(y_score - b[z])

    return X, y, z, centers, W, b


def main():

    n_samples = 100
    n_features = 2
    n_clusters = 5

    X, y, z, centers, W, b = make_latent_triples(n_samples, n_features, n_clusters)

    import matplotlib.pyplot as plt
    Xp, zp = X[y > 0], z[y > 0]
    Xn, zn = X[y < 0], z[y < 0]

    plt.scatter(Xp[:, 0], Xp[:, 1], c=zp, marker='+')
    plt.scatter(Xn[:, 0], Xn[:, 1], c=zn, marker='.')
    plt.show()

if __name__ == '__main__':
    main()

