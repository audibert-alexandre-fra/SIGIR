#Import Library
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import tensorly as tl
import scipy.linalg
import autograd.numpy as np_opt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from functools import reduce
from  tensorly.random import random_cp
import copy


def createRandomEstimation(shapeZ: tuple ,rank: int):
    """[Function to create random Estimation of Z]

    Args:
        shapeZ (tuple): [dimension of the tensor]
        rank (int): [rank for cp decomposition]

    Returns:
        [list]: [list of numpy]
    """
    A_list = []
    for shape in shapeZ:
        A = np.random.normal(size= (shape, rank))
        A_list.append(A)
    return A_list


def outProduct(A_list: list):
    """[compute cp decompostion]

    Args:
        A_list (list): [list of numpy which contains list of array A_i]

    Returns:
        [numpy]: [Tensor compute thanks to cp decomposition]
    """
    weights = np.ones(len(A_list[0][0]))
    return tl.kruskal_to_tensor((weights,A_list))



def kroneckerProduct(A_list: list, n: int):
    """[Kronecker product between a_N^r,...,a_1^r whitout a_n^r]

    Args:
        A_list (list): [list of numpy which contains list of array A_i]
        n (int): [dimension to not use and compute]

    Returns:
        [np.array]: [computation of kronecker]
    """
    init = None
    for index, mat in enumerate(reversed(A_list)):
        if (len(A_list) - (index + 1))  != n:
            if init is None:
                init = mat
            else:
                init = scipy.linalg.khatri_rao(init, mat)
    return init


def unfold(tensor: np.ndarray, mode: int=0):
    """[Function used to unfold in mode ax]

    Args:
        tensor ([type]): [description]
        mode (int, optional): [description]. Defaults to 0.

    Returns:
        [np.ndarray]: [unfold tensor in the great ax]
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), 
                      (tensor.shape[mode], -1), order='F')
    


def put_in_place(A_list: list, lamb: list):
    """[Nurmalize order]

    Args:
        A_list (list): [list of numpy which contains list of array A_i]
        lamb (list): [list of normalisation]
    """
    dimension = len(A_list)
    rank = len(A_list[0][0])
    for dim in range(dimension):
        for r in range(rank):
            A_list[dim][:, r] *= lamb[r]**(1/dimension)
            

def ortho_update(Z: np.ndarray, A_list: list, dim: int) :
    """[Ortogonal find for the order speciliazed as input]

    Args:
        Z (np.ndarray): [original tensor]
        A_list (list): [approximation of Z with CP decomposition]
        dim (int): [int which need orthogonal update]

    Returns:
        [np.ndarray]: [something]
    """
    unfolding = unfold(Z, dim)
    katri_ro = kroneckerProduct(A_list, dim).T
    manifold = Stiefel(A_list[dim].shape[0], A_list[dim].shape[1])
    def cost(X): return np_opt.sum((unfolding - np_opt.dot(X, katri_ro))**2)
    problem = Problem(manifold=manifold, cost=cost, verbosity=False)
    solver = SteepestDescent()
    Xopt = solver.solve(problem)
    return Xopt


# CP-decompostion function 
# Criteria of stop : 500 iterations
def cpDecomposition_help(Z: np.ndarray, dim_ortho: int =1, rank: int=10, max_iter: int=100):
    """[CP-decomposition with orthognal contraint]

    Args:
        Z (np.ndarray): [original tensor to decompose]
        dim_ortho (int, optional): [dim where imply orthogonal contraint]. Defaults to 1.
        rank (int, optional): [hyperparameter rank ]. Defaults to 10.
        max_iter (int, optional): [nb of iteration before to stop algorithm]. Defaults to 100.

    Returns:
        [list]: [list of array obtain by cp decomposition with orthogonal contraint on the
        dimension dim]
    """
    A_list = createRandomEstimation(Z.shape, rank)
    med = np.median(Z)
    criteria = 0
    best_score = None
    order = len(Z.shape)
    save = copy.deepcopy(A_list)
    
    while(criteria < max_iter):
        for dim in range(order):
            lam = np.linalg.norm(A_list[dim], axis=0)
            if dim != dim_ortho:
                non_An = [A_list[m] for m in range(order) if m != dim]
                V = reduce(np.multiply, map((lambda a: np.dot(a.T, a)), non_An))
                new = np.dot(unfold(Z, dim),np.dot(kroneckerProduct(A_list, dim), np.linalg.pinv(V)))
            else:
                new = ortho_update(Z, A_list, dim)
            A_list[dim] = new/lam
            put_in_place(A_list, lam)
        if np.linalg.norm(Z-outProduct(A_list))< 1e-7:
            print("End iteration{}/{}".format(criteria, max_iter))
        save = copy.deepcopy(A_list)
        criteria += 1
    return A_list