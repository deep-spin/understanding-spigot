import torch
import torch.nn as nn
from torch.distributions import Uniform

from ste import (ste_identity, ste_zero, ste_softmax, spigot, spigot_ce,
                 spigot_eg, spigot_zero, _one_hot_argmax, ste_fixed,
                 spigot_ce_argmax, spigot_eg_argmax)
from entmax import sparsemax


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, X, _):
        return self.classifier(X), None


class GoldLatentClassifier(nn.Module):
    def __init__(self, n_features, n_clusters, n_classes):
        super().__init__()
        self.n_clusters = n_clusters
        self.classifier = Decoder(n_features, n_clusters, n_classes)

    def forward(self, X, z):
        z_ = torch.zeros(z.shape[0], self.n_clusters, device=device)
        z_[torch.arange(z.shape[0]), z] = 1
        z_ = z_.type(torch.FloatTensor)
        if device == 'cuda':
            z_ = z_.cuda()
        return self.classifier(X, z_), z


class LatentClassifier(nn.Module):
    def __init__(self, encoder, decoder, latent_model, argmax_test_time):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_model = latent_model
        self.train = True
        self.argmax_test_time = argmax_test_time

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def forward(self, X, z=None, y=None):
        s = self.encoder(X)
        if self.train or not self.argmax_test_time:
            # This copy of the decoder will be used for update of the latent variables
            new_decoder = Decoder(self.decoder.n_features, self.decoder.n_clusters, self.decoder.n_classes)
            new_decoder.load_state_dict(self.decoder.state_dict())

            z_hat = self.latent_model(s, new_decoder, X, y)
            if device == 'cuda':
                z_hat = z_hat.cuda()
            y_hat = self.decoder(X, z_hat)
            return y_hat, torch.argmax(z_hat, dim=-1)
        else:
            z_hat = _one_hot_argmax(s)
            y_hat = self.decoder(X, z_hat)
            return y_hat, torch.argmax(z_hat, dim=-1)


class ExactSearchClassifier(nn.Module):
    """
    method: best/sum
    """
    def __init__(self, encoder, decoder, method='best'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.method = method
        if method == 'best':
            self.latent_model = LatentSTEFixed()
        self.train = True
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def forward(self, X, y, loss_function=None):
        # X.to(self.device)
        # y.to(self.device)
        # X.cuda()
        # print('X:', X)
        s = self.encoder(X)
        losses = torch.zeros(s.shape, device=device)

        if self.train:
            # Execute the decoder for each possible version of the latent variable
            for i in range(s.shape[1]):
                z_i = torch.zeros(s.shape, device=device)
                z_i[:,i] = 1
                y_i = self.decoder(X, z_i)
                loss_i = loss_function(y_i, y)
                losses[:,i] = loss_i

            # Get also the minimum loss for each instance
            z_best = torch.argmax(losses, dim=-1)

            if self.method == 'sum':
                # The possible losses are weighted by the probability of each class (i.e. softmax)
                p = torch.softmax(s, dim=-1)
                loss = torch.sum(torch.einsum('bi,bi->b', p, losses))
            elif self.method == 'best':
                # Here we perform STE with the best value for z in the forward pass
                z_best_onehot = _one_hot_argmax(losses, dim=-1)
                z_hat = self.latent_model(s, z_best_onehot)
                if device == 'cuda':
                    z_hat = z_hat.cuda()
                y_hat = self.decoder(X, z_hat)
                loss = loss_function(y_hat, y)

            return loss, z_best
        else:
            z_hat = _one_hot_argmax(s)
            y_hat = self.decoder(X, z_hat)

            return y_hat, torch.argmax(z_hat, dim=-1)



class REINFORCEClassifier(nn.Module):
    def __init__(self, encoder, decoder, use_baseline):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_baseline = use_baseline
        self.train = True

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def forward(self, X, y, loss_function=None, reward_stats=None):
        s = self.encoder(X)
        p = torch.softmax(s, dim=-1)
        if self.train:
            # Sample from the softmax
            sample = torch.multinomial(p, 1).squeeze(-1)
            z_hat = torch.zeros(s.shape, device=device)
            z_hat[torch.arange(sample.shape[0]), sample] = 1


        #     z_ = torch.zeros(z.shape[0], self.n_clusters)
        # z_[torch.arange(z.shape[0]), z] = 1
        # z_ = z_.type(torch.FloatTensor)

            # print('z_hat', z_hat)
            # print('sample', sample)

            y_hat = self.decoder(X, z_hat)

            # Loss from the downstream task, used to update the decoder parameters
            loss = loss_function(y_hat, y)

            # The reward from REINFORCE will update the encoder parameters
            reward = self.get_reinforce_reward(loss, reward_stats, self.use_baseline)
            log_p_z_given_x = s * z_hat

            total_loss = loss - (reward * log_p_z_given_x).mean()

            return total_loss, torch.argmax(z_hat, dim=-1)
        else:
            z_hat = _one_hot_argmax(s)
            y_hat = self.decoder(X, z_hat)

            return y_hat, torch.argmax(z_hat, dim=-1)

    """
    This method is taken from Goncalo Correia
    """
    def get_reinforce_reward(self, logp, reward_stats, use_baseline=True):
        reward = logp
        # baseline
        if not use_baseline:
            r_avg, r_std = reward.mean().item(), reward.std().item()
            reward = reward - reward_stats.avg
            reward = reward / reward_stats.std
            reward_stats.update(r_avg, r_std)
        reward = reward.detach()
        return reward


class Encoder(nn.Module):
    def __init__(self, n_features, n_clusters):
        super().__init__()
        self.W_s = nn.Linear(n_features, n_clusters)

    def forward(self, X):
        s = self.W_s(X)
        return s


class Decoder(nn.Module):
    def __init__(self, n_features, n_clusters, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_classes = n_classes
        self.W_y = nn.Bilinear(n_clusters, n_features, n_classes)

    def forward(self, X, z):
        # print('in decoder forward()')
        # z.requires_grad=True
        # print('X:', X.shape, 'z:', z.shape, X.requires_grad, z.requires_grad, z)
        y_hat = self.W_y(z, X)
        return y_hat


class DecoderFF(nn.Module):
    def __init__(self, n_features, n_clusters, n_classes, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(n_features+n_clusters, hidden_dim)
        self.W_y = nn.Linear(hidden_dim, n_classes)

    def forward(self, X, z):
        v = self.W_h(torch.cat((X, z), dim=1))
        v = torch.tanh(v)
        y_hat = self.W_y(v)
        return y_hat


class LatentSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, decoder, X, y):
        p = torch.softmax(s, dim=-1)
        return p


class LatentSTEIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        z = ste_identity(s, decoder, X, y, self.latent_config)
        return z


class LatentSTEFixed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z):
        z_hat = ste_fixed(s, z)
        return z_hat


class LatentSTEZero(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        z = ste_zero(s, decoder, X, y, self.latent_config)
        return z


class LatentSTESoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, decoder, X, y):
        z = ste_softmax(s)
        return z


class LatentSPIGOT(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        z = spigot(s, decoder, X, y, self.latent_config)
        # print('spigot returned value in latent model', z)
        return z


class LatentSPIGOTZero(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        z = spigot_zero(s, decoder, X, y, self.latent_config)
        return z


class LatentSPIGOTCE(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        return spigot_ce(s, decoder, X, y, self.latent_config)


class LatentSPIGOTCEArgmax(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        return spigot_ce_argmax(s, decoder, X, y, self.latent_config)


class LatentSPIGOTEG(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        return spigot_eg(s, decoder, X, y, self.latent_config)


class LatentSPIGOTEGArgmax(nn.Module):
    def __init__(self):
        super().__init__()

    def set_latent_config(self, latent_config):
        self.latent_config = latent_config

    def forward(self, s, decoder, X, y):
        return spigot_eg_argmax(s, decoder, X, y, self.latent_config)


class LatentSparsemax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, decoder, X, y):
        z = sparsemax(s, dim=-1)
        return z


class LatentGumbelSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, decoder, X, y):
        m = Uniform(0,1)
        u = m.sample(s.shape)
        g = -torch.log(-torch.log(u))
        if device == 'cuda':
            g = g.cuda()
        p = torch.softmax(s+g, dim=-1)
        return p


class LatentGumbelSTE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, decoder, X, y):
        m = Uniform(0, 1)
        u = m.sample(s.shape)
        g = -torch.log(-torch.log(u))
        if device == 'cuda':
            g = g.cuda()
        z = ste_softmax(s+g)
        return z

