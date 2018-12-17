# -*- coding: utf-8 -*-
# Clustering by fast search and find of density peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import collections
# 解决中文乱码问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

def showDenDisAndDataSet(den, dis, ds):
    # 密度和距离图显示在面板
    plt.figure(num=1, figsize=(15, 9))
    ax1 = plt.subplot(121)
    plt.scatter(x=den, y=dis, c='k', marker='o', s=15)
    plt.xlabel('密度')
    plt.ylabel('距离')
    plt.title('决策图')
    plt.sca(ax1)
    # 将数据集显示在图形面板
    ax2 = plt.subplot(122)
    plt.scatter(x=ds[:, 0], y=ds[:, 1], marker='o', c='k', s=8)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('数据集')
    plt.sca(ax2)
    plt.show()
# 确定类别点,计算每点的密度值与最小距离值的乘积，并画出决策图，以供选择将数据共分为几个类别
def show_nodes_for_chosing_mainly_leaders(gamma):
    plt.figure(num=2, figsize=(15, 10))
    # -np.sort(-gamma) 将gamma从大到小排序
    plt.scatter(x=range(len(gamma)), y=-np.sort(-gamma), c='k', marker='o', s=-np.sort(-gamma) * 100)
    plt.xlabel('点的数量')
    plt.ylabel('每个点的评分')
    plt.title('递减顺序排列的γ')
    plt.show()
def show_result(labels, data, corePoints):
    # 画最终聚类效果图
    plt.figure(num=3, figsize=(15, 10))
    # 一共有多少类别
    clusterNum = len(set(labels))
    scatterColors = [
              '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
              '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
              '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    # 绘制分类数据
    for i in range(clusterNum):
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        # 选择该类别的所有Node
        subCluster = data[np.where(labels == i)]
        plt.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=5, marker='o', alpha=0.66)
    # 绘制每一个类别的聚类中心
    plt.scatter(x=data[corePoints, 0], y=data[corePoints, 1], marker='+', s=100, c='K', alpha=0.8)
    plt.title('聚类结果图')
    plt.show()
# 确定每点的最终分类
# densitySortArr密度从小到大的索引排序
# closestNodeIdArr是比自己密度大，距离自己最近点的id
# n代表一共有多少条数据
def extract_cluster(densitySortArr,closestNodeIdArr, classNum,gamma):
    n=densitySortArr.shape[0]
    # 初始化每一个点的类别
    labels=np.full((n,),-1)
    corePoints =  np.argsort(-gamma)[: classNum]  # 选择
    # 将选择的聚类中心赋予类别
    labels[corePoints]=range(len(corePoints))
    # 将ndarrar数组转为list集合
    densitySortList=densitySortArr.tolist()
    # 将集合元素反转，即密度从大到小的排序索引
    densitySortList.reverse()
    # 循环赋值每一个元素的label
    for nodeId in densitySortList:
        if(labels[nodeId]==-1):
            # 如果nodeId节点没有类别
            # 首先获得closestNodeIdArr[nodeId] 比自己密度大，且距离自己最近的点的索引
            # 将比自己密度大，且距离自己最近的点的类别复制给nodeId
            labels[nodeId]=labels[closestNodeIdArr[nodeId]]
    return corePoints,labels

def CFSFDP(data,dc):
    n,m=data.shape
    # 制作任意两点之间的距离矩阵
    disMat = squareform(pdist(data,metric='euclidean'))
    # 计算每一个点的密度（在dc的园中包含几个点）
    densityArr = np.where(disMat < dc, 1, 0).sum(axis=1)
    # 将数据点按照密度大小进行排序（从小到大）
    densitySortArr=np.argsort(densityArr)
    # 初始化，比自己密度大的且最近的距离
    closestDisOverSelfDensity = np.zeros((n,))
    # 初始化，比自己密度大的且最近的距离对应的节点id
    closestNodeIdArr = np.zeros((n,), dtype=np.int32)
    # 从密度最小的点开始遍历
    for index,nodeId in enumerate(densitySortArr):
        #  点密度大于当前点的点集合
        nodeIdArr = densitySortArr[index+1:]
        # 如果不是密度最大的点
        if nodeIdArr.size != 0:
            # 计算比自己密度大的点距离nodeId的距离集合
            largerDistArr = disMat[nodeId][nodeIdArr]
            # 寻找到比自己密度大，且最小的距离节点
            closestDisOverSelfDensity[nodeId] = np.min(largerDistArr)
            # 寻找到最小值的索引，索引实在largerdist里面的索引（确保是比nodeId）节点大
            # 如果存在多个最近的节点，取第一个
            # 注意，这里是largerDistArr里面的索引
            min_distance_index = np.argwhere(largerDistArr == closestDisOverSelfDensity[nodeId])[0][0]
            # 获得整个数据中的索引值
            closestNodeIdArr[nodeId] = nodeIdArr[min_distance_index]
        else:
            # 如果是密度最大的点，距离设置为最大，且其对应的ID设置为本身
            closestDisOverSelfDensity[nodeId] = np.max(closestDisOverSelfDensity)
            closestNodeIdArr[nodeId] = nodeId
    #  由于密度和最短距离两个属性的数量级可能不一样，分别对两者做归一化使结果更平滑
    normal_den = (densityArr - np.min(densityArr)) / (np.max(densityArr) - np.min(densityArr))
    normal_dis = (closestDisOverSelfDensity - np.min(closestDisOverSelfDensity)) / (
                np.max(closestDisOverSelfDensity) - np.min(closestDisOverSelfDensity))
    gamma = normal_den * normal_dis
    # densityArr：里面保存了每一个点的密度大小
    # densitySortArr：数组中保存了每一个点排序
    # closestDisOverSelfDensity：数组保存了比本身密度大，且最近的距离
    # closestNodeIdArr:数组保存了比本身密度大，且最近的节点id
    # gamma:每个点的评分(综合距离和密度)
    return densityArr,densitySortArr,closestDisOverSelfDensity,closestNodeIdArr,gamma
data = np.loadtxt("data/cluster.csv", delimiter=",")
# 执行聚类算法
densityArr,densitySortArr,closestDisOverSelfDensity,closestNodeIdArr,gamma= CFSFDP(data,2)
showDenDisAndDataSet(densityArr, closestDisOverSelfDensity, data)
show_nodes_for_chosing_mainly_leaders(gamma)
# 根据决策图提取类别
classNum = int(input('请输入簇集的个数：'))
# corePoints是聚类中心的索引id
# labels是一个点的聚类类别
corePoints,labels = extract_cluster(densitySortArr, closestNodeIdArr, classNum, gamma)
show_result(labels, data, corePoints)
