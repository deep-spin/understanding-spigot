import copy
import torch

from entmax import sparsemax

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_function = torch.nn.CrossEntropyLoss()


def grad_wrt_latent(z, X, y, decoder):
    if device == 'cuda':
            z = z.cuda()
            decoder.cuda()
    with torch.enable_grad():
        z.requires_grad_()
        y_hat = decoder(X, z)
        loss = loss_function(y_hat, y)
        grad_z = torch.autograd.grad(loss, z)[0]
    return grad_z


def _one_hot_argmax(s, dim=-1):
    argmax = torch.argmax(s, dim=dim)
    z = torch.zeros(s.shape)
    z[torch.arange(s.shape[0]), argmax] = 1
    if device == 'cuda':
        z = z.cuda()
    return z


class STEIdentity(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Identity
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        z = _one_hot_argmax(s, dim)
        ctx.save_for_backward(s,z, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config

        return z

    @staticmethod
    def backward(ctx, grad_z):
        s,z, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_best = z - step_size * grad_z

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = z_best - step_size * grad_z

        grad_s = z - z_best
        return grad_s, None, None, None, None


class STEFixed(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Identity
    """
    @staticmethod
    def forward(ctx, s, z, dim=-1):
        return z

    @staticmethod
    def backward(ctx, grad_z):
        return grad_z, None


class STEZero(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Identity
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1, z_fixed=None):
        z = torch.zeros(s.shape, device=device)
        ctx.save_for_backward(s, z, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config
        return z

    @staticmethod
    def backward(ctx, grad_z):
        s, z, X, y, = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_best = z - step_size * grad_z

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = z_best - step_size * grad_z

        grad_s = _one_hot_argmax(s) - z_best
        return grad_s, None, None, None, None


class STESoftmax(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Softmax
    """
    @staticmethod
    def forward(ctx, s, dim=-1):
        p = torch.softmax(s, dim=dim)
        ctx.save_for_backward(s, p)
        return _one_hot_argmax(s, dim)

    @staticmethod
    def backward(ctx, grad_z):
        s, p,  = ctx.saved_tensors
        # The derivative of Softmax:
        diag = torch.eye(s.shape[1], device=device).repeat(s.shape[0], 1, 1)
        diag = torch.einsum('bij,bi->bij', diag, p)
        dpds = diag - torch.einsum('bi,bj->bij', p, p)
        # Multiply the softmax derivative by the derivative of the loss with respect to z
        grad_s = torch.einsum('bii,bi->bi', dpds, grad_z.clone())
        return grad_s


class SPIGOT(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Softmax
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1, z_fixed=None):
        if z_fixed is not None:
            z = z_fixed
        else:
            z = _one_hot_argmax(s, dim)
        ctx.save_for_backward(z, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config
        # print('spigot forward')
        return z

    @staticmethod
    def backward(ctx, grad_z):
        z, X, y,  = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_best = sparsemax(z - step_size*grad_z)

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = sparsemax(z_best - step_size*grad_z)

        grad_s = z - z_best
        return grad_s, None, None, None, None


class SPIGOTZero(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Softmax
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        z = torch.zeros(s.shape, device=device)
        ctx.save_for_backward(s, z, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config
        return z

    @staticmethod
    def backward(ctx, grad_z):
        s, z, X, y,  = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        z_best = sparsemax(z - step_size*grad_z)

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = sparsemax(z_best - step_size*grad_z)

        grad_s = _one_hot_argmax(s) - z_best
        return grad_s, None, None, None, None


class SPIGOTCE(torch.autograd.Function):
    """
    SPIGOT for Cross-entropy loss
    Forward: softmax / Backward: spigot-ce
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        p = torch.softmax(s, dim)
        ctx.save_for_backward(p, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config
        return p

    @staticmethod
    def backward(ctx, grad_z):
        p, X, y,  = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # SPIGOT update with Cross-Entropy loss
        z_best = sparsemax(p - step_size * grad_z)

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = sparsemax(z_best - step_size * grad_z)

        grad_s = p - z_best
        return grad_s, None, None, None, None


class SPIGOTCEArgmax(torch.autograd.Function):
    """
    SPIGOT for Cross-entropy loss
    Forward: argmax / Backward: spigot-ce
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        z = _one_hot_argmax(s, dim)
        ctx.save_for_backward(s, X, y)
        ctx.dim = dim
        ctx.decoder = decoder
        ctx.config = latent_config
        return z

    @staticmethod
    def backward(ctx, grad_z):
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # grad_z gets completely ignored.
        s, X, y,  = ctx.saved_tensors

        # SPIGOT update with Cross-Entropy loss
        p = torch.softmax(s, ctx.dim)
        z_best = p.clone()  # init

        for i in range(gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = sparsemax(z_best - step_size * grad_z)

        grad_s = p - z_best
        return grad_s, None, None, None, None


class SPIGOTEG(torch.autograd.Function):
    """
    SPIGOT w Exponentiated Gradient updates
    Forward: softmax / Backward: spigot-eg
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        p = torch.softmax(s, dim)
        ctx.save_for_backward(s, p, X, y)
        ctx.decoder = decoder
        ctx.config = latent_config
        return p

    @staticmethod
    def backward(ctx, grad_z):
        s, p, X, y,  = ctx.saved_tensors
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # The derivative after updating the latent variable with
        # one step of exponentiated gradient
        z_best = torch.softmax(s - step_size * grad_z, dim=-1)

        for i in range(1, gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = torch.softmax(torch.log(z_best) - step_size * grad_z, dim=-1)

        grad_s = p - z_best
        return grad_s, None, None, None, None


class SPIGOTEGArgmax(torch.autograd.Function):
    """
    SPIGOT w Exponentiated Gradient updates
    Forward: argmax / Backward: spigot-ce
    """
    @staticmethod
    def forward(ctx, s, decoder, X, y, latent_config, dim=-1):
        z = _one_hot_argmax(s, dim)
        ctx.save_for_backward(s, X, y)
        ctx.dim = dim
        ctx.decoder = decoder
        ctx.config = latent_config
        return z

    @staticmethod
    def backward(ctx, grad_z):
        step_size = ctx.config['step_size']
        gradient_update_steps = ctx.config['gradient_update_steps']

        # grad_z gets completely ignored.

        s, X, y,  = ctx.saved_tensors

        # The derivative after updating the latent variable with
        # one step of exponentiated gradient
        p = torch.softmax(s, ctx.dim)
        z_best = p.clone()  # init

        for i in range(gradient_update_steps):
            grad_z = grad_wrt_latent(z_best.detach(), X, y, ctx.decoder)
            z_best = torch.softmax(torch.log(z_best) - step_size * grad_z, dim=-1)

        grad_s = p - z_best
        return grad_s, None, None, None, None


ste_identity = STEIdentity.apply
ste_zero = STEZero.apply
ste_softmax = STESoftmax.apply
ste_fixed = STEFixed.apply
spigot = SPIGOT.apply
spigot_zero = SPIGOTZero.apply
spigot_ce = SPIGOTCE.apply
spigot_eg = SPIGOTEG.apply
spigot_ce_argmax = SPIGOTCEArgmax.apply
spigot_eg_argmax = SPIGOTEGArgmax.apply


