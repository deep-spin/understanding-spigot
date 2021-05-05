import torch
import numpy as np

from lpsmap import TorchFactorGraph, DepTree

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sparsemap_projection(arc_scores):
    fg = TorchFactorGraph()
    u = fg.variable_from(arc_scores.transpose(0,1).clone())
    fg.add(DepTree(u, packed=True, projective=True))
    fg.solve(max_iter=1, max_inner_iter=50, step_size=0)
    marg_u = u.value.transpose(0,1)
    return marg_u


def sparsemap_batched(arc_scores_batched):
    sparsemap_projections = torch.zeros(arc_scores_batched.shape)
    for i, arc_scores in enumerate(arc_scores_batched):
        sparsemap_proj = sparsemap_projection(arc_scores)
        sparsemap_proj = sparsemap_proj.clone().detach().requires_grad_(True) 
        sparsemap_projections[i] = sparsemap_proj

    if device == 'cuda':
        sparsemap_projections = sparsemap_projections.cuda()

    return sparsemap_projections 

    



