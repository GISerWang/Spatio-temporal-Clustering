# -*- coding: UTF-8 -*-
# 此算法还可以进一步改进（邻域搜索的时候也要保证沿着时间轴前进-尤其是下一个簇集的第一个点的邻域搜素）
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time
from mpl_toolkits.mplot3d import Axes3D
# 判断pt点是否为核心点
# ptIndex为点pt的索引
# floorIds每一个轨迹点的楼层Id
# disMat距离矩阵
# timeDisMat时间距离矩阵
# eps1为pt的空间半径
# eps2为pt的时间半径
# minPts为pt的圆中的点个数（每个点都不一样）
def isCorePoint(ptIndex,floorIds,xyArr,timeArr,eps1,eps2,minPts):
    # 获得索引为ptIndex轨迹点楼层Id
    currentFloorId = floorIds[ptIndex]
    xyDisArr = compute_xy_dis(ptIndex,xyArr)
    timeDisArr = compute_time_dis(ptIndex,timeArr)
    # 满足三个条件为核心点：在一定“时间”、“空间”邻域中包含“同楼层”轨迹点的个数大于minPts
    flag = (xyDisArr <= eps1) & (timeDisArr <= eps2) & (floorIds == currentFloorId)
    if np.sum(np.where(flag, 1, 0)) >= minPts:
        return True
    return False
# 计算X矩阵的距离矩阵
def compute_xy_dis(ptIndex,xyArr):
    xyDisArr = cdist(xyArr[ptIndex].reshape(-1,2), xyArr, metric='euclidean')
    return xyDisArr[0]
def compute_time_dis(ptIndex,timeArr):
    timeDisArr = cdist(timeArr[ptIndex].reshape(-1, 1), timeArr.reshape(-1, 1), metric='euclidean')
    return timeDisArr[0]
# 改进的地方
def updateSeeds(seeds,neighbours,labels,start):
    # 簇集的扩展，延时间轴扩展，即：neighbour >= start
    for neighbour in neighbours:
        if labels[neighbour] == -1 and neighbour >= start:
            seeds.add(neighbour)
    return seeds
# data 的第一列是unix时间戳，第二、三列是空间X,Y坐标，第四列是楼层ID
# eps1 空间邻域
# # eps2 时间邻域
# # minPts 满足双邻域的最少点的个数
def Indoor_STDBSCAN(data,eps1,eps2,minPts):
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 获得每一个轨迹点的时间数组
    timeArr=data[:,0]
    # 获得空间距离矩阵
    xyArr = data[:,1:3]
    # 获得每一个轨迹点的楼层Id
    floorIds = data[:,3]
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    # 遍历所有轨迹点寻找簇集
    clusterId = 0
    start = 0
    for ptIndex in range(n):
        if ptIndex < start:
            continue
        if labels[ptIndex] != -1:
            continue
        # 如果某轨迹点不是核心点，直接continue(因为簇集的产生是由核心点控制)
        if not isCorePoint(ptIndex, floorIds, xyArr, timeArr, eps1, eps2, minPts):
            continue
        # 进入到这里说明ptIndex是核心点
        # 首先将点pointId标记为当前类别(即标识为已操作)
        labels[ptIndex] = clusterId
        # 然后寻找种子点的“eps1”、“eps2”邻域且“没有被分类”的“同楼层”点，将其放入种子集合
        # 获得索引为ptIndex轨迹点楼层Id
        currentFloorId = floorIds[ptIndex]
        # 获得同一个楼层，时间邻域下面的
        neighbours = np.where((compute_xy_dis(ptIndex,xyArr) <= eps1) & (compute_time_dis(ptIndex,timeArr) <= eps2) & (labels == -1) & ((floorIds == currentFloorId)))[0]
        seeds = set()
        seeds = updateSeeds(seeds, neighbours, labels, start)
        while len(seeds) > 0:
            # 弹出一个新种子点
            newPoint = seeds.pop()
            # 将newPoint标记为当前类
            labels[newPoint] = clusterId
            # 寻找newPoint种子点eps邻域（包含自己）
            queryResults = np.where((compute_xy_dis(newPoint,xyArr) <= eps1) & (compute_time_dis(newPoint,timeArr) <= eps2) & (floorIds == floorIds[newPoint]))[0]
            if len(queryResults) >= minPts:
                # 将邻域内且没有被分类的点压入种子集合
                seeds = updateSeeds(seeds, queryResults, labels, start)
        # 簇集生长完毕，寻找到一个类别
        clusterId = clusterId + 1
        # 簇集的扩展，延时间轴扩展，此处获得当前聚类的最后一个点的索引
        start = np.argwhere(labels != -1)[-1][0]
    return labels
def plotFeature(data, labels_):
    clusterNum=len(set(labels_))
    ax = plt.figure().add_subplot(111, projection='3d')
    scatterColors = ['red', 'blue', 'orange', 'orchid', 'black', '#E0E3E6', '#E0E0E6','#E5E0E6','#E6E0E4','#E6E0E4','brown']
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1],subCluster[:,2], c=colorSytle, s=20)
    plt.show()
data = np.loadtxt("./data/cluster_unix_time_indoor.csv", delimiter=",")
start = time.clock()
labels=Indoor_STDBSCAN(data,3,100,10)
end = time.clock()
print('finish all in %s' % str(end - start))
plotFeature(data[:,1:], labels)