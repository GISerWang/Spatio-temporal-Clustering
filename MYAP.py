# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
def plotFeature(data, labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', '#BC8F8F', '#8B4513', 'brown']
    ax = fig.add_subplot(111)
    for i in range(0, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels == i)]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=20)
    plt.show()
# data:参与聚类的数据集
# preference：每一个数据点的参考度
# damping：阻尼系数，用户控制数据的收敛速度
# max_iter：最大迭代的次数
# convergence_iter:比较多少次聚类中心不变之后停止迭代
# similarity:相似度，可以指定，也可以自动计算
def MYAffinityPropagation(data,preference,damping,max_iter,convergence_iter,similarity = None,precomputed=False):
    # 获得数据一共有多少行
    n,m = data.shape
    if precomputed == False:
        # 如果没有指定相似度矩阵，采用负的欧氏距离作为相似度的度量
        # 相似度越高，越容易聚在一起
        similarity = - squareform(pdist(data,metric='euclidean'))
    # similarity.flat将矩阵压缩成一维
    # [::(n+1)]代表切片，所有的元素，每隔(n+1)进行赋值，简单来说就是给矩阵的对角线赋值
    similarity.flat[::(n+1)] = preference
    # 用于控制中途停止的变量，当连续convergence_iter次聚类中心不发生变化，聚类停止
    stop = np.zeros((n, convergence_iter))
    # 初始化吸引度矩阵Responsibility
    responsibility = np.zeros((n, n))
    # 初始化归属度矩阵Availability
    availability = np.zeros((n, n))
    # 获得一个列表[0,1,2,3,4,...,n-1]
    # 此列表主要用于修改指定元素
    ind = np.arange(n)
    # 开始迭代，寻找exemplars
    for idx in range(max_iter):
        # 表示公式a[i,k]+s[i,k]
        a_add_s = np.add(availability, similarity)
        # 获得每一行，最大的值的索引,及其最大值
        maxIdx,maxY = np.argmax(a_add_s, axis=1), np.max(a_add_s, axis=1)
        # 将每一行最大值索引赋值为“负无穷”，然后求每一行的第二大的值
        a_add_s[ind, maxIdx] = - np.inf
        maxSecondY = np.max(a_add_s, axis=1)
        # maxY.reshape(-1,1) 将做大的值转置为n行，1列
        # 然后采用np.tile()方法，将数据变成n行n列，注意是横向复制
        # 采用similarity减去最大值，但是有一点不同，本应是最大值的那个数，应该是减去第二大的值
        oldResponsibility = similarity - np.tile(maxY.reshape(-1,1), (1,n))
        # 修改本应减去第二大值的那个数值
        oldResponsibility[ind, maxIdx] = similarity[ind, maxIdx] - maxSecondY
        # 迭代更新responsibility矩阵
        responsibility = responsibility*damping + (1-damping)*oldResponsibility
        # np.maximum比较responsibility与0的值，如果0大，那么保留0，否则等于responsibility原来的值
        # 此方法等于temp[i,k]=max(0,r[i,k])
        zero_max_r = np.maximum(responsibility, 0)
        # 将responsibility对角线的值，赋值给zero_max_r的对角线
        zero_max_r.flat[::n + 1] = responsibility.flat[::n + 1]
        # 将每一列的值，加起来,注意这里，其实多加了一个数，即等于i的时候
        # 公式等于:r[k,k]+ max(0,r[i`,k]) i` 不等于k，但是这里可以等于i
        # 这里减去zero_max_r等于，去除了i这个条件
        zero_max_r = np.sum(zero_max_r, axis=0)-zero_max_r
        # r[k,k]+ max(0,r[i`,k]) i` 不等于i
        dA = np.diag(zero_max_r).copy()
        # np.minimum：取两个矩阵中的最小值
        oldAvailability = np.minimum(0, zero_max_r)
        # 给矩阵的对角线赋值，即a[k,k]的赋值
        oldAvailability.flat[::(n + 1)] = dA
        # 迭代更新availability矩阵
        availability = availability * damping + (1 - damping) * oldAvailability
        # 检查迭代是否停止，即连续convergence_iter 中心不发生改变
        # 当两个矩阵的对角线大于0时，这个点为聚类中心
        E = (np.diag(availability) + np.diag(responsibility)) > 0
        stop[:, idx % convergence_iter] = E
        # 识别了多少个聚类中心点:K个
        K = np.sum(E, axis=0)
        # 当迭代次数大于convergence_iter，才有可能存在停止的可能性
        if idx >= convergence_iter:
            # 将stop按照行叠加，可以知道每一个点，在convergence_iter迭代中，做了多少次聚类中心
            se = np.sum(stop, axis=1)
            # 求的不变的轨迹点，判断是否等于m，如果是，那么聚类结束
            # 不变的轨迹点有两种：
            #           一种是连续在convergence_iter迭代中都是聚类中心，
            #           一种是convergence_iter都不是聚类中心
            converged = np.sum((se == convergence_iter) + (se == 0)) == n
            if converged and (K > 0):
                break
    # 聚类结束，求聚类中心
    exemplars = np.where(np.diag(availability + responsibility) > 0)[0]
    # 返回exemplars的索引,及相似性矩阵
    return exemplars,similarity
# 根据相似性矩阵即聚类中心提取labels
def extract_cluster(exemplars, similarity):
    # 距离哪一个点相似度大，就属于哪一个类别
    # 注意，核心点会分类错误，因为自己距离自己距离点都比较小，为0
    labels = np.argmax(similarity[:,exemplars],axis=1)
    for index,exemplar in enumerate(exemplars):
        labels[exemplar]=index
    return labels
data = np.loadtxt("data/cluster.csv", delimiter=",")
starttime = time.clock()
exemplars,similarity = MYAffinityPropagation(data,preference=-60,damping=0.9,max_iter=500,convergence_iter=15)
endtime = time.clock()
print(endtime - starttime)
labels = extract_cluster(exemplars,similarity)
plotFeature(data, labels)