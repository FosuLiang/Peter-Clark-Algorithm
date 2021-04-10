#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Kevin D
# Description: test

import utils.utils as ut
from PCByD import pc

# 设置生成图的节点数以及sample size，emmm...别忘了设置种子
N = 5
T = 800
seed = 20

ut.set_seed(seed)
message = ut.generate_data_linear_DAG(N, T)
data = message[0]
true_graph = message[1]

row_count = data.shape[1]
# 节点
labels = [str(i) for i in range(row_count)]

# 运行PC算法，独立性检验使用高斯检验
est_graph = pc(suffStat={"C": data.corr().values, "n": T}, alpha=.01, labels=labels)

print("true graph:")
print(true_graph)

print("est_graph:")
print(est_graph)

# 画图
# ut.Draw(est_graph, labels)
result = ut.count_accuracy(true_graph, est_graph)

tpr = result['tpr']
fpr = result['fpr']
shd = result['shd']

print("'tpr': %f, 'fpr': %f, 'shd': %f," % (tpr, fpr, shd))
