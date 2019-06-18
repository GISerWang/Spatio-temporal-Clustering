# -*- coding: utf-8 -*-
# Clustering by fast search and find of density peaks
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
# 计算任意两点之间的欧氏距离,并返回矩阵为矩阵
def compute_squared_EDM_method(X):
  return squareform(pdist(X,metric='euclidean'))
# 绘制密度和时间距离图
def showDenDisAndDataSet(den, dis, ds):
    # 密度和时间距离图显示在面板
    plt.figure(num=1, figsize=(15, 9))
    ax1 = plt.subplot(121)
    plt.scatter(x=den, y=dis, c='k', marker='o', s=15)
    plt.xlabel('密度')
    plt.ylabel('时间距离')
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
    y=-np.sort(-gamma)
    plt.scatter(x=range(len(gamma)), y=y, c='k', marker='o', s=y * 100)
    plt.xlabel('n',fontsize=20)
    plt.ylabel('γ',fontsize=20)
    # plt.title('递减顺序排列的γ')
    plt.show()
# 显示聚类结果
def show_result1(labels, data, corePointIds):
    plt.figure(num=3, figsize=(15, 10))
    plt.scatter(data[corePointIds, 1],data[corePointIds, 2],marker='x',s=100,color='red')
    return plt
def show_result(plt,labels, data, corePoints):
    # 画最终聚类效果图
    # 一共有多少类别
    clusterNum = len(corePoints)
    scatterColors = ['black',
              'blue', 'green', 'yellow', '#67C6A', '#228B22',
              '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
              '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    # 绘制分类数据
    for i in range(0,clusterNum):
        # 为i类别选择颜色
        colorSytle = scatterColors[i % len(scatterColors)]
        # 选择该类别的所有Node
        subCluster = data[np.where(labels == i)]
        plt.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=15, marker='o', alpha=0.66)
    # # 绘制每一个类别的聚类中心

    plt.title('')
    plt.show()
# 确定每一个点的分类
def extract_clustering(data,corePointIds,xyDisMat):
    # 获取数据一共行n
    n=data.shape[0]
    # 初始化每一个点的类别
    labels=np.full((n,),-1)
    # 或许一共有多少类别
    classNum = len(corePointIds)
    # 赋予聚类中心类别
    labels[corePointIds] = range(classNum)
    for index in range(classNum):
        # 延时间轴 第一个聚类中心的前面所有点 都是第0类
        if(index==0):
            labels[:corePointIds[index]]=0
        # 延时间轴 最后一个聚类中心的后面所有点 都是最后一类
        elif(index==classNum-1):
            labels[corePointIds[index]:] = classNum-1
        else:
            # 获得当前聚类中心和前一个聚类中心的索引
            first=corePointIds[index-1]
            last=corePointIds[index]
            # 开始遍历 （注意从first+1开始。从前向后循环）
            x=first+1
            while(x<last and labels[x]==-1):
                # 如果距离前一个聚类中心大于后一个聚类中心，循环停止
                if(xyDisMat[x][first]>xyDisMat[x][last]):
                    break
                # 否则赋予x类别
                labels[x] = labels[first]
                x=x+1
            # 开始遍历 （注意从last-1开始，从后往前循环）
            x = last-1
            while (x > first and labels[x]==-1):
                # 如果距离前一个聚类中心小于后一个聚类中心，循环停止
                if (xyDisMat[x][first] < xyDisMat[x][last]):
                    break
                # 否则赋予x类别
                labels[x] = labels[last]
                x = x - 1
    return labels
# 确定最终的核心点索引
# num:需要选择的潜在的聚类个数
# disThreh:距离阈值，当相邻的核心坐标小于disThreh时应该合并
# gamma：代表时间距离与密度的乘积
def extract_corePoints(num, disThreh,xyDisMat,gamma):
    # 选取前num个点，（num并不是class的个数，还需要合并）
    corePointIds = np.argsort(-gamma)[: num]  # 选择
    plt=show_result1(None,data,corePointIds)
    # 将点索引按照时间顺序排序
    corePointIds=np.sort(corePointIds)
    # result时最终的分类id
    result=[]
    # 遍历计算延时间轴相邻两点之间的距离，如果小于一定阈值就合并
    for index,pointId in enumerate(corePointIds):
        if (index == (len(corePointIds) - 1)):
            break
        # 第一个点，一定是分类点
        if (index == 0):
            result.append(pointId)
        # 遍历计算相邻两点之间的距离
        if(xyDisMat[corePointIds[index]][corePointIds[index-1]]>disThreh and index>0):
            result.append(pointId)
    return result,plt
# 1.5 100
# 2 90
# data 的第一列是unix时间戳，剩余列是空间坐标数据
# disThreh 空间邻域
# timeThreh 时间邻域
def ST_CFSFDP(data,disThreh=2,timeThreh=200):
    # 获得数据的行和列
    n,m = data.shape
    # 获得数据的时间矩阵
    timeDisMat = compute_squared_EDM_method(data[:, 0].reshape(n, 1))
    # 获得空间距离矩阵
    xyDisMat = compute_squared_EDM_method(data[:, 1:])
    # 统计每一个点的密度（在距离阈值和时间阈值同时的作用下）
    densityArr=np.where((xyDisMat < disThreh) & (timeDisMat < timeThreh), 1, 0).sum(axis=1)
    # 将数据点按照密度大小进行排序（从小到大）
    densitySortArr = np.argsort(densityArr)
    # 初始化，比自己密度大的且最近的时间距离
    closestDisOverSelfDensity = np.zeros((n,))
    # 从密度最小的点开始遍历
    for index, nodeId in enumerate(densitySortArr):
        # 点密度大于当前点的点集合
        nodeIdArr = densitySortArr[index + 1:]
        # 如果不是密度最大的点
        if nodeIdArr.size != 0:
            # 计算比自己密度大的点距离nodeId的时间距离集合
            largerDistArr = timeDisMat[nodeId][nodeIdArr]
            # 寻找到比自己密度大，且最小的时间距离节点
            closestDisOverSelfDensity[nodeId] = np.min(largerDistArr)
        else:
            # 如果是密度最大的点，距离设置为最大，且其对应的ID设置为本身
            closestDisOverSelfDensity[nodeId] = np.max(closestDisOverSelfDensity)
    #  由于密度和最短距离两个属性的数量级可能不一样，分别对两者做归一化使结果更平滑
    normal_den = densityArr / np.max(densityArr)
    normal_dis = closestDisOverSelfDensity / np.max(closestDisOverSelfDensity)
    gamma = normal_den * normal_dis
    # densityArr：里面保存了每一个点的密度大小
    # closestDisOverSelfDensity：数组保存了比本身密度大，且最近的距离
    # gamma:每个点的评分(综合距离和密度)
    # xyDisMat：距离矩阵
    return densityArr, closestDisOverSelfDensity, gamma,xyDisMat
# 加载数据
data = np.loadtxt("data/cluster_unix_time.csv", delimiter=",")
# 执行聚类算法
disThreh=1
timeThreh=200
start=time.clock()
densityArr,closestDisOverSelfDensity,gamma,xyDisMat= ST_CFSFDP(data,disThreh,timeThreh)
end=time.clock()
showDenDisAndDataSet(densityArr, closestDisOverSelfDensity, data[:,1:])
show_nodes_for_chosing_mainly_leaders(gamma)
# 根据决策图提取类别
num = int(input('请输入潜在簇集的个数：'))
corePointIds,plt = extract_corePoints(num,disThreh,xyDisMat,gamma)
labels = extract_clustering(data,corePointIds,xyDisMat)
show_result(plt,labels,data[:,1:],corePointIds)
print('finish all in %s' % str(end - start))