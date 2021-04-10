from random import random

from scipy.stats import norm
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig

def reachMatrix(Graph, labels):
    """
    compute the reachability matrix
    :param labels:
    :param Graph: source graph
    :param N: node number
    :return: reachability matrix
    """
    N = len(labels)
    reachableMatrix = np.zeros((N, N))

    M = Graph
    for i in range(N - 1):
        M = np.dot(M, Graph)
        reachableMatrix += M

    return reachableMatrix

# 偏相关系数
def partial_corr(C, x, y, S):
    """
    compute Partial correlation between A and B while S, learn more: https://en.wikipedia.org/wiki/Partial_correlation
    :param C: correlation
    :param x: sample of A
    :param y: sample of B
    :param S: set
    :return: Partial correlation result
    """
    # S中没有点。0阶情况
    if len(S) == 0:
        return C[x, y]

    # S中只有一个点，一阶偏相关系数
    elif len(S) == 1:
        return (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[x, S], 2)) * (1 - math.pow(C[y, S], 2)))

    # 高阶情况，迭代计算
    else:
        Rxy = partial_corr(C, x, y, S[1:len(S)])
        Rxz0 = partial_corr(C, x, S[0], S[1:len(S)])
        Rz0y = partial_corr(C, y, S[0], S[1:len(S)])
        return (Rxy - Rxz0 * Rz0y) / math.sqrt(1 - math.pow(Rxz0, 2) * (1 - math.pow(Rz0y, 2)))


# 条件独立性检验
def gaussCItest(suffstat, x, y, S):
    """
    execute the conditional independence test
    :param suffstat: a
    :param x:
    :param y:
    :param S:
    :return:
    """
    C = suffstat["C"]
    n = suffstat["n"]

    # 偏相关系数
    r = partial_corr(C, x, y, S)

    # Fisher’s z-transform
    res = math.sqrt(n - len(S) - 3) * .5 * math.log((1 + r) / (1 - r))

    # h >= Φ^{-1}(1-α/2) -> 2(1 - Φ(h)) >= α
    return 2 * (1 - norm.cdf(abs(res)))


def Draw(graph, labels):
    """
    draw and save the graph
    :param graph: graph you want to draw and save, type dataFrame will be fine
    :param labels: labels in the graph
    :return: no return
    """
    # 创建空有向图
    G = nx.DiGraph()

    for i in range(len(graph)):
        G.add_node(labels[i])
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edges_from([(labels[i], labels[j])])

    nx.draw(G, with_labels=True)
    # 保存DAG图
    plt.savefig("dataWithTenNodeByMyCode.png")
    plt.show()


def generate_data_linear_DAG(N, T):
    """
    :param N: Node Number
    :param T: Sample size
    :return: simulated data, edge matirx(true graph), weight matirx
    """
    # 初始化各节点的噪声值
    # data = np.random.randn(T, N)
    data = np.random.uniform(0, 0.3, [T, N])
    # 生成邻接矩阵,上三角
    edge_mat = np.triu((np.random.uniform(0, 1, [N, N]) < 0.3).astype('int'), 1)

    # 这里的权重用的都是正数，为避免出现违背faithfulness assumption的情况
    weight_mat = np.random.uniform(0.3, 1, [N, N]) * edge_mat

    for i in tqdm(range(N)):
        data[:, i] = data[:, i] + np.dot(data, weight_mat[:, i])

    data = pd.DataFrame(data)
    edge_mat = pd.DataFrame(edge_mat)
    return data, edge_mat, weight_mat

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    Others: I find this method from "notears". It is a great job done by professor Zheng.
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

def set_seed(seed):
    """
    normal tips for doing experiments
    :param seed: random int, you can set it by yourself
    :return: nothing
    """
    # for np
    np.random.seed(seed)
