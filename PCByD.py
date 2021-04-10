import itertools
import numpy as np
import utils.utils as ut


def UCG(labels):
    """
    init a undirected graph base on labels
    :param labels: labels
    :return: undirected graph
    """
    m = len(labels)
    G = np.array(np.ones((m, m)))
    return G


def neighborsFind(Graph, x, y):
    """
    find out Adj(G, x) / {y}
    :param Graph: graph
    :param x: this is x
    :param y: this is y
    :return: Adj(G, x) / {y}
    """

    G1 = Graph.copy()
    if Graph[x][y] == 1:
        # 除去x--y
        G1[x][y] = 0
        neighborsSet = []
        for j in range(len(Graph[x])):
            if G1[x][j] == 1:
                neighborsSet.append(j)
    return neighborsSet


def undirectNodeBrother(G):
    """
    find out all undirected pairwise in G
    :param G: graph
    :return: undirected pairwise
    """

    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == 1 and G[j][i] == 1:
                ind.append((i, j))
    return ind


def directNodeBrother(G):
    """
    find out all directed pairwise in G
    :param G: graph
    :return: directed pairwise
    """
    ind = []
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] == 1 and G[j][i] == 0:
                ind.append((i, j))
    return ind


def skeleton(suffStat, alpha, labels):
    """
    estimate the skeleton, the first stage in PC algorithm
    :param labels:
    :param suffStat: Dictionary contains data and nodenumber
    :param alpha: significance level, we set it as .05 usually
    :param N: node number
    :return: a undirected graph(skeleton)
    """
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]
    Sep = set()

    # 完全无向图
    G = UCG(labels)

    for i in range(len(labels)):
        # 不需要检验i--i
        G[i][i] = 0

    # done flag
    done = False

    ord = 0

    while not done:

        done = True
        # 相邻点对，ind记录相邻点对的下标
        ind = undirectNodeBrother(G)

        for x, y in ind:
            if G[x][y]:
                neighbors = neighborsFind(G, x, y)

                if len(neighbors) >= ord:

                    # |adj(C, x) / {y}|>ord
                    if len(neighbors) > ord:
                        done = False

                    if ord <= 1:
                        # 遍历全部基数为1的子集
                        sep = []
                        for neighbors_S in set(itertools.combinations(neighbors, ord)):
                            pval = ut.gaussCItest(suffStat, x, y, list(neighbors_S))

                            # 条件独立
                            if pval >= alpha:
                                G[x][y] = G[y][x] = 0
                                print("pval between %d - %d = %f" % (x, y, pval))
                                print("%d and %d is conditional independent by %s, so delete edge %d - %d" % (
                                    x, y, neighbors_S, x, y))
                                print(neighbors_S)
                                # 把neighbors_S加入分离集

                                sep += list(neighbors_S)
                        sepset[x][y] = sep

                    else:
                        # |adj(C, x) / {y}|=ord
                        for neighbors_S in set(itertools.combinations(neighbors, ord)):

                            # x,y是否被ord阶neighbors_S而条件独立
                            # 条件独立性检验，返回p-value
                            pval = ut.gaussCItest(suffStat, x, y, list(neighbors_S))

                            # 条件独立
                            if pval >= alpha:
                                G[x][y] = 0
                                G[y][x] = 0
                                print("pval between %d - %d = %f" % (x, y, pval))

                                print("%d and %d is conditional independent by %s, so delete edge %d - %d" % (
                                    x, y, neighbors_S, x, y))
                                # 把neighbors_S加入分离集
                                sep = list(neighbors_S)
                                sepset[x][y] = sep
                                # Sep.add(neighbors_S)
                                break

        ord += 1

    ind = undirectNodeBrother(G)
    return {'sk': np.array(G), 'sepset': sepset, 'ind': ind}


def rule1(Graph, labels):
    """
    execute the rule1 in PC: x -> y - z , if "x - z" is False, then orient y - z => y -> z (y <-> z => y -> z)
    :param Graph: graph with v-structure step finished
    :param N: node number
    :return: graph with rule1 finished
    """
    while True:
        ind = directNodeBrother(Graph)
        changed = False
        for x, y in ind:
            # 找到与 y 相连且与 x 不相连的 z，存入集合Z_S
            for z in range(len(labels)):
                if Graph[y][z] == 1 and Graph[z][y] == 1 and Graph[x][z] == 0 and Graph[z][x] == 0 and z != x:
                    Graph[y][z] = 1
                    Graph[z][y] = 0
                    changed = True
                if changed:
                    break
        if not changed:
            break
    print("rule1 finished.")
    return Graph


def rule2(Graph, labels):
    """
    execute the rule2 in PC: If there is a path like x -> z -> y，then orient x - y => x -> y(x <-> y => x -> y)
    :param Graph: graph with rule1 step finished
    :param N: node number
    :return: graph with rule2 step finished
    """

    # 1. 判断 x 到 y 是否存在有向路径，使用可达矩阵
    G_temp = Graph.copy()
    # 将为定向边 <-> 改为 - ，便于后续的可达矩阵计算
    for i in range(len(G_temp)):
        for j in range(i, len(G_temp[i])):
            if G_temp[i][j] == 1 and G_temp[j][i] == 1:
                G_temp[i][j] = G_temp[j][i] = 0

    # 找到所有的 a - b 序列对
    # changed 判断dag是否出现了新的定向边
    while True:
        changed = False
        ind = undirectNodeBrother(G_temp)
        M = ut.reachMatrix(G_temp, labels)

        # sorted对ind进行排序
        for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
            if M[x][y] > 0 and Graph[x][y] == 1 and Graph[y][x] == 1:
                # x - y可达且未定向
                # Graph[x][y] = True
                Graph[x][y] = 1
                changed = 1
            if changed is True:
                break
        if changed is False:
            break
    print("rule2 finished.")
    return Graph


def extend_cpdag(Graph, labels):
    """
    execute the rule1 and rule2 in PC algorithm
    :param Graph: graph that confirms v-structure
    :param labels: labels
    :return: cpdag
    """
    pdag = Graph.copy()
    pdag_rule1 = rule1(pdag, labels)
    pdag_rule2 = rule2(pdag_rule1, labels)

    return pdag_rule2


def pc(suffStat, alpha, labels):
    """
    Peter-Clark Algorithm
    :param suffStat: Dictionary contain data and node number
    :param alpha: significance level
    :param labels: labels
    :return: estimate graph by pc
    """
    # 骨架
    skeletonGraph = skeleton(suffStat, alpha, labels=labels)

    # V-结构
    Ved_Graph = V_Structure(skeletonGraph, labels)

    # CPDAG
    CPDAG = extend_cpdag(Ved_Graph, labels)
    return CPDAG


def V_Structure(Graph, labels):
    """
    excute the v - Structure step in PC Algorithm
    :param Graph: skeleton graph
    :param labels: labels
    :return: graph that confirms v - structure
    """
    # 砍边后的V-结构学习，若x - y - z，且 y 不在Sep[x][z]中，变为 x -> y <- z
    ind = Graph['ind']
    G = Graph['sk']
    Sep = Graph['sepset']

    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        # 存放所有与 y 邻接且与 x 不邻接的 z
        Z_S = []

        for z in range(len(labels)):
            if G[y][z] == 1 and G[z][y] == 1 and G[z][x] == 0 and G[x][z] == 0 and z != x:
                Z_S.append(z)

        # 定向
        for z in Z_S:
            if Sep[x][z] != None and Sep[z][x] != None and not (y in Sep[x][z] or y in Sep[z][x]):
                print("v-structure %d - %d - %d, sep[x][z]: %s" % (x, y, z, Sep[x][z]))
                G[x][y] = G[z][y] = 1
                G[y][x] = G[y][z] = 0

    return np.array(G)
