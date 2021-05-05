import torch
# import numpy as np

from lpsmap import TorchFactorGraph, DepTree

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sparsemap_projection(arc_scores):
    fg = TorchFactorGraph()
    u = fg.variable_from(arc_scores.transpose(0,1).clone())
    fg.add(DepTree(u, packed=True, projective=False))
    fg.solve(max_iter=1, max_inner_iter=50, step_size=0)
    marg_u = u.value.transpose(0,1)
    return marg_u


def sparsemap_batched(arc_scores_batched):
    sparsemap_projections = torch.zeros(arc_scores_batched.shape)
    for i, arc_scores in enumerate(arc_scores_batched):
        sparsemap_proj = sparsemap_projection(arc_scores)
        sparsemap_proj = torch.tensor(sparsemap_proj)
        sparsemap_projections[i] = sparsemap_proj

    if device == 'cuda':
        sparsemap_projections = sparsemap_projections.cuda()

    return sparsemap_projections


def argmax_nonproj(arc_scores):
    fg = TorchFactorGraph()
    u = fg.variable_from(arc_scores.transpose(0,1).clone())
    fg.add(DepTree(u, packed=True, projective=False))
    fg.solve_map()
    marg_u = u.value.transpose(0,1)
    return marg_u


def argmax_batched(scores_batched):
    argmax_np = torch.zeros(scores_batched.shape)
    for i, scores in enumerate(scores_batched):
        argmax = torch.tensor(argmax_nonproj(scores))
        argmax_np[i] = argmax
    if device == 'cuda':
        argmax_np = argmax_np.cuda()
    return argmax_np



