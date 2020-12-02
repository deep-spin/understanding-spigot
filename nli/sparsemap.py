import torch
import numpy as np

from lpsmap.ad3ext.parse import Parse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sparsemap_projection(arc_scores):
    parser = Parse(projective=False, packed=True, length=arc_scores.shape[0])
    marg_u = np.empty_like(arc_scores)
    parser.sparsemap(arc_scores.ravel('F'), marg_u.ravel(), return_active_set=False)
    return marg_u.transpose(), parser


def sparsemap_batched(arc_scores_batched):
    parsers = []
    sparsemap_projections = torch.zeros(arc_scores_batched.shape)
    for i, arc_scores in enumerate(arc_scores_batched):
        sparsemap_proj, parser = sparsemap_projection(arc_scores.cpu().detach().numpy())
        sparsemap_proj = torch.tensor(sparsemap_proj)
        sparsemap_projections[i] = sparsemap_proj
        parsers.append(parser)

    if device == 'cuda':
        sparsemap_projections = sparsemap_projections.cuda()

    return sparsemap_projections, parsers


def sparsemap_gradient(du, parser):
    sym_grad = np.empty_like(du)
    parser.jv(du.ravel('F'), sym_grad.ravel())
    return sym_grad.transpose()


def sparsemap_gradient_batched(scores_batched, parsers):
    sparsemap_grad = torch.zeros(scores_batched.shape)
    for i, scores in enumerate(scores_batched):
        grad = torch.tensor(sparsemap_gradient(scores.cpu().detach().numpy(), parsers[i]))
        sparsemap_grad[i] = grad

    if device == 'cuda':
        sparsemap_grad = sparsemap_grad.cuda()

    return sparsemap_grad


def argmax_nonproj(arc_scores):
    parser = Parse(projective=False, packed=True, length=arc_scores.shape[0])
    max_tree = np.empty_like(arc_scores)
    max_tree_score = parser.max(arc_scores.ravel('F'), max_tree.ravel())
    return max_tree.transpose()


def argmax_batched(scores_batched):
    argmax_np = torch.zeros(scores_batched.shape)
    for i, scores in enumerate(scores_batched):
        argmax = torch.tensor(argmax_nonproj(scores.cpu().detach().numpy()))
        argmax_np[i] = argmax
    if device == 'cuda':
        argmax_np = argmax_np.cuda()
    return argmax_np


def gradient(du):
    eps = 1e-5
    marg_plus = np.empty_like(du)
    marg_minus = np.empty_like(du)
    parser.sparsemap(X.ravel() + eps * du, marg_plus)
    parser.sparsemap(X.ravel() - eps * du, marg_minus)
    grad = (marg_plus - marg_minus) / (2 * eps)
    return grad





