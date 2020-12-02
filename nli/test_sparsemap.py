import torch
import numpy as np

from lpsmap.ad3ext.parse import Parse

from torch_struct import NonProjectiveDependencyCRF

from sparsemap import sparsemap_batched, sparsemap_gradient_batched, argmax_batched, argmax_nonproj


def test_sparsemap():
    print('TODO...')
    n = 4
    temp = 100

    arc_scores = np.random.random((n, n))

    # pytorch-struct parser:
    arc_scores_torch = torch.tensor(arc_scores).unsqueeze(0)
    print(arc_scores_torch.shape)
    # pystruct_marginals = NonProjectiveDependencyCRF(arc_scores_torch).marginals
    pystruct_argmax = NonProjectiveDependencyCRF(arc_scores_torch*temp).marginals

    print('torch-struct:')
    # print('marginals:', pystruct_marginals)
    print(pystruct_argmax.numpy())

    parser = Parse(projective=False, packed=True, length=arc_scores.shape[0])
    marg_u = np.empty_like(arc_scores)
    max_tree = np.empty_like(arc_scores)

    print("SparseMAP Maximizing")
    max_tree_score = parser.max(arc_scores.ravel('F'), max_tree.ravel())
    print(max_tree.T)

    print('From method:')
    print(argmax_nonproj(arc_scores))


def test_sparsemap_batched():
    b = 3
    n = 4
    temp = 20
    arc_scores_batched = torch.rand(b, n, n)

    argmax_np = argmax_batched(arc_scores_batched)

    print('Scores', arc_scores_batched.shape)
    print('Argmax batched:', argmax_np.shape)
    print(argmax_np)

    print('Pytorch struct:')
    pystruct_argmax = NonProjectiveDependencyCRF(arc_scores_batched*temp).marginals
    print(pystruct_argmax)


if __name__ == '__main__':
    test_sparsemap()
    test_sparsemap_batched()




