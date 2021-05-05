import torch
import numpy as np

from lpsmap import TorchFactorGraph, DepTree
from torch_struct import DependencyCRF

from sparsemap import sparsemap_batched

def test_sparsemap():
    print('TEST test_sparsemap')
    n = 6
   
    arc_scores = torch.randn(n, n, requires_grad=True)

    with torch.no_grad():
        torch.autograd.set_detect_anomaly(True)
        print('Calculate with torch struct')
        print(arc_scores.shape)
        dist = DependencyCRF(arc_scores.unsqueeze(0), multiroot=False)
        pystruct_tree = dist.argmax.detach()[0]
        print('Torch-struct tree:')
        print(pystruct_tree)
        print(pystruct_tree.shape)
        
        print('arc_scores before transpose:', arc_scores.shape)
        arc_scores = arc_scores.transpose(0,1).clone()
        print('arc_scores transposed:', arc_scores.shape)

        # exit()

        # MAP
        fg = TorchFactorGraph()
        v = fg.variable_from(arc_scores)
        fg.add(DepTree(v, packed=True, projective=True))
        fg.solve_map()
        max_tree = v.value.transpose(0,1)
        print('MAX TREE with solve_map:')
        print(max_tree)
        print(max_tree.shape)

        assert max_tree.allclose(pystruct_tree), "The trees form torch-struct and lp-sparsemap.solve_map do not match"

        # exit()

        fg = TorchFactorGraph()
        v = fg.variable_from(arc_scores*100)
        fg.add(DepTree(v, packed=True, projective=True))
        fg.solve(max_iter=1, max_inner_iter=50, step_size=0)
        max_tree = v.value
        print('MAX TREE 2 with solve(scores*100):')
        print(max_tree)
        print(arc_scores.shape)

        # SparseMAP marginals
        u = fg.variable_from(arc_scores)
        fg.add(DepTree(u, packed=True, projective=True))
        fg.solve(max_iter=1, max_inner_iter=50, step_size=0)
        marg_u = u.value
        print('SPARSE MARGINALS:')
        print(marg_u)
        print('SPARSE MARGINALS grad:')
        print(marg_u.grad)


def test_batched_sparsemap():
    print('TEST test_batched_sparsemap')
    b = 5
    n = 5
    arc_scores_batched = torch.rand(b, n, n, requires_grad=True)

    sparsemap_proj = sparsemap_batched(arc_scores_batched)

    print('Scores', arc_scores_batched.shape)
    print('Sparsemap batched projection:', sparsemap_proj.shape)
    print(sparsemap_proj)

    print('Testing the grad:')
    sparsemap_deriv = sparsemap_proj.grad #sparsemap_gradient_batched(arc_scores_batched)
    print('Sparsemap batched derivative:', sparsemap_deriv)



if __name__ == '__main__':
    test_batched_sparsemap()

    test_sparsemap()






