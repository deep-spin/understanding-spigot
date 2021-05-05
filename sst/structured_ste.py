import torch
from sparsemap import sparsemap_batched

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_function = torch.nn.functional.binary_cross_entropy_with_logits

from torch_struct import DependencyCRF

def tree_representation(z, X):
    avg_heads = torch.einsum('bnm,bnh->bnh', z, X)
    tree_repr = torch.cat((X, avg_heads), dim=2)
    tree_repr = torch.sum(tree_repr, dim=1)
    return tree_repr


def grad_wrt_latent(z, X, y, decoder):
    # print('-----grad_wrt_latent:', z.shape, X.shape, y.shape, decoder)
    z = z.detach()
    if device == 'cuda':
            z = z.cuda()
            decoder.cuda()
    with torch.enable_grad():
        z.requires_grad_()
        decoder_input = tree_representation(z, X)
        y_hat = decoder(decoder_input)
        loss = loss_function(y_hat, y)
        grad_z = torch.autograd.grad(loss, z)[0]
    return grad_z


def ste_marginals(arc_scores):
    """
    This method does not need a backprop, a detach() trick is used instead.
    """
    dist = DependencyCRF(arc_scores)
    argmax = dist.argmax
    marginals = dist.marginals

    return (argmax-marginals).detach() + marginals


def sparsemap(arc_scores):
    return sparsemap_batched(arc_scores)


class STEIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        z = parse_argmax(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(z, X, y)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        z, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_guess = z - step_size * grad_z

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_guess, X, y, ctx.decoder)
            z_guess = z_guess - step_size * grad_z

        grad_s = z - z_guess
        return grad_s, None, None, None, None


class SPIGOT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        z = parse_argmax(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(z, X, y)
        # print('spigot forward', z.shape)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        z, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # SPIGOT update
        z_guess = sparsemap_batched(z - step_size * grad_z)[0]

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_guess, X, y, ctx.decoder)
            z_guess = sparsemap_batched(z_guess - step_size * grad_z)[0]

        grad_s = z - z_guess
        # print('spigot backward', grad_s.shape)
        return grad_s, None, None, None, None


class SPIGOTCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        mu = parse_marginals(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(mu, X, y)
        return mu

    @staticmethod
    def backward(ctx, grad_mu):
        mu, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_guess = sparsemap_batched(mu - step_size * grad_mu)[0]

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_guess, X, y, ctx.decoder)
            z_guess = sparsemap_batched(z_guess - step_size * grad_z)[0]

        # SPIGOT update with Cross-Entropy loss
        grad_s = mu - z_guess
        return grad_s, None, None, None, None


class SPIGOTEG(torch.autograd.Function):
    """
    SPIGOT for exponentiated gradient
    """
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        mu = parse_marginals(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(arc_scores, mu, X, y)
        return mu

    @staticmethod
    def backward(ctx, grad_mu):
        scores, mu_init, X, y,  = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        scores = scores - step_size * grad_mu
        mu = parse_marginals(scores)

        for i in range(1, gradient_update_steps):
            grad_mu = grad_wrt_latent(mu, X, y, ctx.decoder)
            scores = scores - step_size * grad_mu
            mu = parse_marginals(scores)

        grad_s = mu_init - mu

        # The derivative after updating the latent variable with
        # one step of exponentiated gradient
        # This is the old: - seems the same as the current one
        # grad_s = mu - parse_marginals(arc_scores - step_size * grad_mu)

        return grad_s, None, None, None, None


class SPIGOTCEArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        z = parse_argmax(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(arc_scores, X, y)
        return z

    @staticmethod
    def backward(ctx, grad_mu):
        s, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        mu = parse_marginals(s)
        z_guess = mu.clone()

        for i in range(gradient_update_steps):
            grad_z = grad_wrt_latent(z_guess, X, y, ctx.decoder)
            z_guess = sparsemap_batched(z_guess - step_size * grad_z)[0]

        # SPIGOT update with Cross-Entropy loss
        grad_s = mu - z_guess
        return grad_s, None, None, None, None


class SPIGOTEGArgmax(torch.autograd.Function):
    """
    SPIGOT for exponentiated gradient
    """
    @staticmethod
    def forward(ctx, arc_scores, decoder, X, y, config):
        z = parse_argmax(arc_scores)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(arc_scores, X, y)
        return z

    @staticmethod
    def backward(ctx, grad_mu):
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        scores, X, y,  = ctx.saved_tensors
        mu_init = parse_marginals(scores)
        mu = mu_init.clone()

        for i in range(gradient_update_steps):
            grad_mu = grad_wrt_latent(mu, X, y, ctx.decoder)
            scores = scores - step_size * grad_mu
            mu = parse_marginals(scores)

        grad_s = mu_init - mu

        # The derivative after updating the latent variable with
        # one step of exponentiated gradient
        # This is the old: - seems the same as the current one
        # grad_s = mu - parse_marginals(arc_scores - step_size * grad_mu)

        return grad_s, None, None, None, None

def parse_marginals(scores):
    return DependencyCRF(scores).marginals


def parse_argmax(scores):
    return DependencyCRF(scores).argmax.detach()



ste_identity = STEIdentity.apply
spigot = SPIGOT.apply
spigot_ce_argmax = SPIGOTCEArgmax.apply
spigot_eg_argmax = SPIGOTEGArgmax.apply



