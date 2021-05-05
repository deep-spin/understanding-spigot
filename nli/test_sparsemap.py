import torch

from lpsmap import TorchFactorGraph, DepTree

from torch_struct import NonProjectiveDependencyCRF

from sparsemap import sparsemap_batched, argmax_batched, argmax_nonproj


def test_sparsemap():
    print('----------------------------')
    print('TEST test_sparsemap')
    n = 4

    arc_scores = torch.randn(n, n, requires_grad=True)

    # pytorch-struct parser:
    # arc_scores_torch = torch.tensor(arc_scores).unsqueeze(0)
    print(arc_scores.shape)
    # pystruct_marginals = NonProjectiveDependencyCRF(arc_scores_torch).marginals
    pystruct_marg = NonProjectiveDependencyCRF(arc_scores.unsqueeze(0), multiroot=True).marginals
    print('torch-struct marginals:')
    print(pystruct_marg)

    pystruct_map = NonProjectiveDependencyCRF(arc_scores.unsqueeze(0)*100, multiroot=True).marginals
    print('torch-struct MAP:')
    print(pystruct_map)

    # SparseMAP marginals
    fg = TorchFactorGraph()
    u = fg.variable_from(arc_scores.transpose(0,1).clone())
    fg.add(DepTree(u, packed=True, projective=False))
    fg.solve(max_iter=1, max_inner_iter=50, step_size=0)
    marg_u = u.value.transpose(0,1)

    # MAP
    fg = TorchFactorGraph()
    u = fg.variable_from(arc_scores.transpose(0,1).clone())
    fg.add(DepTree(u, packed=True, projective=False))
    fg.solve_map()
    max_tree = u.value.transpose(0,1)


    print("SparseMAP Maximizing")
    print(max_tree)

    print('From method:')
    print(argmax_nonproj(arc_scores))


def test_sparsemap_batched():
    print('----------------------------')
    print('TEST test_sparsemap_batched')
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




