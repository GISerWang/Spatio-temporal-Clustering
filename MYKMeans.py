import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# 计算两个矩阵的距离矩阵
def compute_distances_no_loops(A, B):
    return cdist(A,B,metric='euclidean')

# 显示簇集，如果簇集类别大于6类，需要增加colorMark的内容
def plotFeature(data, labels_):
    clusterNum=len(set(labels_))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown','#BC8F8F','#8B4513','#FFF5EE']
    ax = fig.add_subplot(111)
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1], c=colorSytle, s=20)
    plt.show()

# 聚类算法的实现
# 需要聚类的数据data
# K 聚类的个数
# tol 聚类的容差，即ΔJ
# 聚类迭代都最大次数N
def K_means(data,K,tol,N):
    #一共有多少条数据
    n = np.shape(data)[0]
    # 从n条数据中随机选择K条，作为初始中心向量
    # centerId是初始中心向量的索引坐标
    centerId = random.sample(range(0, n), K)
    # 获得初始中心向量,k个
    centerPoints = data[centerId]
    # 计算data到centerPoints的距离矩阵
    # dist[i][:],是i个点到三个中心点的距离
    dist = compute_distances_no_loops(data, centerPoints)
    # axis=1寻找每一行中最小值都索引
    # getA()是将mat转为ndarray
    # squeeze()是将label压缩成一个列表
    labels = np.argmin(dist, axis=1).squeeze()
    # 初始化old J
    oldVar = -0.0001
    # data - centerPoint[labels]，获得每个向量与中心向量之差
    # np.sqrt(np.sum(np.power(data - centerPoint[labels], 2)，获得每个向量与中心向量距离
    # 计算new J
    newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
    # 迭代次数
    count=0
    # 当ΔJ大于容差且循环次数小于迭代次数，一直迭代。负责结束聚类
    # abs(newVar - oldVar) >= tol:
    while count<N and abs(newVar - oldVar) > tol:
        oldVar = newVar
        for i in range(K):
            # 重新计算每一个类别都中心向量
            centerPoints[i] = np.mean(data[np.where(labels == i)], 0)
        # 重新计算距离矩阵
        dist = compute_distances_no_loops(data, centerPoints)
        # 重新分类
        labels = np.argmin(dist, axis=1).squeeze()
        # 重新计算new J
        newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
        # 迭代次数加1
        count+=1
    # 返回类别标识，中心坐标
    return labels,centerPoints
starttime = time.clock()
data = np.loadtxt("data/cluster.csv", delimiter=",")
labels,_=K_means(data,3,0.01,100)
endtime = time.clock()
print(endtime - starttime)
plotFeature(data, labels)