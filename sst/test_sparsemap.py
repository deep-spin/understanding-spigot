import torch
import numpy as np

from lpsmap.ad3ext.parse import Parse

from torch_struct import DependencyCRF

from sparsemap import sparsemap_batched, sparsemap_gradient_batched


def test_sparsemap():
    print('TODO...')
    n = 6
    # arc_scores = np.random.random((n,n))
    # arc_scores = torch.zeros((3, 3))

    # # 1->* and 2->* are equal
    # arc_scores[0, 0] += 10
    # arc_scores[1, 1] += 10
    # arc_scores[1, 0] += 1  # 2->1
    # arc_scores[2, 0] += 1  # 3->1

    arc_scores = np.random.random((n, n))

    # sparsemap_tree, sparsemap_tree_score, sparsemap_marginals, sparsemap_trees = test_sparsemap(arc_scores.cpu().numpy())

    # pytorch-struct parser:
    arc_scores_torch = torch.tensor(arc_scores).unsqueeze(0)
    print(arc_scores_torch.shape)
    dist = DependencyCRF(arc_scores_torch)
    pystruct_tree = dist.argmax.detach()[0]

    # print('SparseMAP tree:')
    # print(sparsemap_tree)

    print('torch-struct tree:')
    print(pystruct_tree)

    parser = Parse(projective=True, packed=True, length=arc_scores.shape[0])
    marg_u = np.empty_like(arc_scores)
    max_tree = np.empty_like(arc_scores)

    print("SaprseMAP Maximizing")
    max_tree_score = parser.max(arc_scores.ravel('F'), max_tree.ravel())
    print(max_tree.T)
    exit()

    print()
    print("SparseMAP-projecting")
    p = parser.sparsemap(arc_scores.ravel(), marg_u.ravel(), return_active_set=True)
    print(marg_u)
    
    # print(np.dot(scores, marg_u))
    print()
    print("SparseMAP-selected trees:")
    print(p)

    return max_tree.transpose(), max_tree_score, marg_u.transpose(), p


def test_batched_sparsemap():
    b = 5
    n = 5
    arc_scores_batched = torch.rand(b, n, n)

    sparsemap_proj = sparsemap_batched(arc_scores_batched)

    print('Scores', arc_scores_batched.shape)
    print('Sparsemap batched projection:', sparsemap_proj.shape)
    print(sparsemap_proj)

    print('Testing the grad:')
    sparsemap_deriv = sparsemap_gradient_batched(arc_scores_batched)
    print('Sparsemap batched derivative:', sparsemap_deriv.shape)
    print(sparsemap_deriv)



if __name__ == '__main__':
    # test_batched_sparsemap()

    test_sparsemap()






