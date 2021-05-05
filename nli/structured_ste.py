import torch

from torch_struct import NonProjectiveDependencyCRF

from sparsemap import sparsemap_batched, argmax_batched

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_function = torch.nn.functional.cross_entropy


def tree_representation(parser_output, parser_base_vectors):
    avg_heads = torch.einsum('bnm,bnh->bnh', parser_output, parser_base_vectors)
    tree_representation = torch.cat((parser_base_vectors, avg_heads), dim=2)        
    return tree_representation


def grad_wrt_latent(z_p, z_h, X_p, X_h, mask_p, mask_h, y, decoder):
    # print('-----grad_wrt_latent:', z.shape, X.shape, y.shape, decoder)
    z_p = z_p.detach()
    z_h = z_h.detach()
    if device == 'cuda':
        z_p = z_p.cuda()
        z_h = z_h.cuda()
        decoder.cuda()
    with torch.enable_grad():
        z_p.requires_grad_()
        z_h.requires_grad_()
        decoder_input_p = tree_representation(z_p, X_p)
        decoder_input_h = tree_representation(z_h, X_h)
        # print('before decoder, in structured_ste:', decoder_input_p.shape, decoder_input_h.shape, X_p.shape, X_h.shape, mask_p.shape, mask_h.shape)
        y_hat = decoder(decoder_input_p, decoder_input_h, X_p, X_h, mask_p, mask_h)

        loss = loss_function(y_hat, y)
        # print('y_hat', y_hat)
        # print('z_p', z_p)
        # print('y', y)
        # print('X_p', X_p)
        # print('decoder', decoder)
        # print('loss', loss)

        grad_z_p = torch.autograd.grad(loss, z_p, retain_graph=True)[0]
        grad_z_h = torch.autograd.grad(loss, z_h)[0]
    return grad_z_p, grad_z_h


def ste_marginals(arc_scores_p, arc_scores_h):
    """
    This method does not need a backprop, a detach() trick is used instead.
    """
    sm_p = ste_marginals_single(arc_scores_p)
    sm_h = ste_marginals_single(arc_scores_h)

    return sm_p, sm_h


def ste_marginals_single(arc_scores):
    """
    This method does not need a backprop, a detach() trick is used instead.
    """
    marginals = parse_marginals(arc_scores)
    argmax = parse_argmax(arc_scores)
    return (argmax-marginals).detach() + marginals


def sparsemap(arc_scores_p, arc_scores_h):
    z_p = sparsemap_batched(arc_scores_p)
    z_h = sparsemap_batched(arc_scores_h)
    return z_p, z_h


class STEIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y, decoder, config):
        z_p = parse_argmax(arc_scores_p)
        z_h = parse_argmax(arc_scores_h)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(z_p, z_h, X_p, X_h, mask_p, mask_h, y)
        return z_p, z_h

    @staticmethod
    def backward(ctx, grad_z_p, grad_z_h):
        z_p, z_h, X_p, X_h, mask_p, mask_h, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_guess_p = z_p - step_size * grad_z_p
        z_guess_h = z_h - step_size * grad_z_h

        # TODO: is this OK?

        # for (z_guess, X, grad_z) in [(z_guess_p, X_p, grad_z_p), (z_guess_h, X_h, grad_z_h)]:
        for i in range(1, gradient_update_steps):
            grad_z_p, grad_z_h = grad_wrt_latent(z_guess_p, z_guess_h, X_p, X_h, mask_p, mask_h, y, ctx.decoder)
            z_guess_p = z_guess_p - step_size * grad_z_p
            z_guess_h = z_guess_h - step_size * grad_z_h

        grad_s_p = z_p - z_guess_p
        grad_s_h = z_h - z_guess_h
        return grad_s_p, grad_s_h, None, None, None, None, None, None, None


class SPIGOT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y, decoder, config):
        ctx.decoder = decoder
        ctx.config = config
        z_p = parse_argmax(arc_scores_p)
        z_h = parse_argmax(arc_scores_h)
        ctx.save_for_backward(z_p, z_h, X_p, X_h, mask_p, mask_h, y)
        # print('spigot forward', z.shape)
        return z_p, z_h

    @staticmethod
    def backward(ctx, grad_z_p, grad_z_h):
        z_p, z_h, X_p, X_h, mask_p, mask_h, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # SPIGOT update
        z_guess_p = sparsemap_batched(z_p - step_size * grad_z_p)[0]
        z_guess_h = sparsemap_batched(z_h - step_size * grad_z_h)[0]

        for i in range(1, gradient_update_steps):
            grad_z_p, grad_z_h = grad_wrt_latent(z_guess_p, z_guess_h, X_p, X_h, mask_p, mask_h, y, ctx.decoder)
            # grad_z = grad_wrt_latent(z_guess, X, y, ctx.decoder)
            z_guess_p = sparsemap_batched(z_guess_p - step_size * grad_z_p)[0]
            z_guess_h = sparsemap_batched(z_guess_h - step_size * grad_z_h)[0]

        grad_s_p = z_p - z_guess_p
        grad_s_h = z_h - z_guess_h
        # print('spigot backward', grad_s.shape)
        return grad_s_p, grad_s_h, None, None, None, None, None, None, None


class SPIGOTCEArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y, decoder, config):
        ctx.decoder = decoder
        ctx.config = config
        z_p = parse_argmax(arc_scores_p)
        z_h = parse_argmax(arc_scores_h)
        ctx.save_for_backward(arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y)
        return z_p, z_h

    @staticmethod
    def backward(ctx, grad_mu_p, grad_mu_h):
        s_p, s_h, X_p, X_h, mask_p, mask_h, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        mu_p = parse_marginals(s_p)
        z_guess_p = mu_p.clone()

        mu_h = parse_marginals(s_h)
        z_guess_h = mu_h.clone()

        for i in range(gradient_update_steps):
            grad_z_p, grad_z_h = grad_wrt_latent(z_guess_p, z_guess_h, X_p, X_h, mask_p, mask_h, y, ctx.decoder)
            z_guess_p = sparsemap_batched(z_guess_p - step_size * grad_z_p)[0]
            z_guess_h = sparsemap_batched(z_guess_h - step_size * grad_z_h)[0]

        # SPIGOT update with Cross-Entropy loss
        grad_s_p = mu_p - z_guess_p
        grad_s_h = mu_h - z_guess_h
        return grad_s_p, grad_s_h, None, None, None, None, None, None, None


class SPIGOTEGArgmax(torch.autograd.Function):
    """
    SPIGOT for exponentiated gradient
    """
    @staticmethod
    def forward(ctx, arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y, decoder, config):
        z_p = parse_argmax(arc_scores_p)
        z_h = parse_argmax(arc_scores_h)
        ctx.decoder = decoder
        ctx.config = config
        ctx.save_for_backward(arc_scores_p, arc_scores_h, X_p, X_h, mask_p, mask_h, y)
        return z_p, z_h

    @staticmethod
    def backward(ctx, grad_mu_p, grad_mu_h):
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        scores_p, scores_h, X_p, X_h, mask_p, mask_h, y,  = ctx.saved_tensors
        mu_init_p = parse_marginals(scores_p)
        mu_p = mu_init_p.clone()
        mu_init_h = parse_marginals(scores_h)
        mu_h = mu_init_h.clone()

        for i in range(gradient_update_steps):
            grad_mu = grad_wrt_latent(mu_p, mu_h, X_p, X_h, mask_p, mask_h, y, ctx.decoder)
            scores_p = scores_p - step_size * grad_mu_p
            mu_p = parse_marginals(scores_p)
            scores_h = scores_h - step_size * grad_mu_h
            mu_h = parse_marginals(scores_h)

        grad_s_p = mu_init_p - mu_p
        grad_s_h = mu_init_h - mu_h

        # The derivative after updating the latent variable with
        # one step of exponentiated gradient
        # This is the old: - seems the same as the current one
        # grad_s = mu - parse_marginals(arc_scores - step_size * grad_mu)

        return grad_s_p, grad_s_h, None, None, None, None, None, None, None

def parse_marginals(scores):
    return NonProjectiveDependencyCRF(scores).marginals


def parse_argmax(scores):
    return argmax_batched(scores)


ste_identity = STEIdentity.apply
spigot = SPIGOT.apply
spigot_ce_argmax = SPIGOTCEArgmax.apply
spigot_eg_argmax = SPIGOTEGArgmax.apply



