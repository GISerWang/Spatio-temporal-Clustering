# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import Axes3D
# 绘制不同的点类别
def plotFeature(data, clusterArr):
    clusterNum = len(clusterArr)
    ax = plt.figure().add_subplot(111, projection='3d')
    scatterColors = ['red', 'blue', 'orange', 'orchid', 'black', '#E0E3E6', '#E0E0E6', '#E5E0E6', '#E6E0E4', '#E6E0E4',
                     'brown']
    for i in range(clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[clusterArr[i].indexSet]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], subCluster[:, 2], c=colorSytle, s=20)
    plt.show()

# 凝聚层次聚类算法的簇集
class AGNESCluster(object):
    def __init__(self, center, startId, endId, indexSet):
        # 中心坐标
        self.center = center
        # 起索引（start）
        self.startId = startId
        # 始索引（end）
        self.endId = endId
        # 该簇包含的点索引集合
        self.indexSet = indexSet
    def __repr__(self):
        return "------------\ncenter:%s\nstartId:%s\nendId:%s\nindedxSet:%s\n"%(self.center,self.startId,self.endId,self.indexSet)
# 计算两个簇集的时间距离
def computerTime(cluster1,cluster2,timeArr):
    # 如果两个簇集相离，时间距离为计算
    if(cluster1.endId<cluster2.startId):
        return timeArr[cluster2.startId]-timeArr[cluster1.endId]
    elif(cluster1.startId > cluster2.endId):
        return timeArr[cluster1.startId] - timeArr[cluster2.endId]
    # 如果两个首尾都相等，那就是一个簇集
    elif (cluster1.startId == cluster2.startId) and (cluster1.endId == cluster2.endId):
        return 0
    else:
    # 如果两个簇集相交，时间距离为-1
        return -1
# 更新簇集更新矩阵
# clusterArr：簇集
# clusterDisMat：当前的簇集空间距离矩阵
# clusterTimeMat：当前的簇集时间距离矩阵
# timeArr：每一个点的时间
# needHBId：需要被替换的距离数组索引
# deleteClusterId：需要被删除的簇集索引
def updateClusterMat(clusterArr,clusterDisMat,clusterTimeMat, timeArr,needHBId,deleteClusterId):
    # 一共有n个簇集
    n = len(clusterArr)
    # 删除第deleteClusterId空间距离，主要为第deleteClusterId行和deleteClusterId列
    clusterDisMat = np.delete(clusterDisMat, deleteClusterId, axis=1)
    clusterDisMat = np.delete(clusterDisMat, deleteClusterId, axis=0)
    # 删除第deleteClusterId时间距离，主要为第deleteClusterId行和deleteClusterId列
    clusterTimeMat = np.delete(clusterTimeMat, deleteClusterId, axis=1)
    clusterTimeMat = np.delete(clusterTimeMat, deleteClusterId, axis=0)
    # 需要被更新的时间距离与空间距离
    upDisArr = []
    upTimeArr = []
    for i in range(n):
        d = np.linalg.norm(clusterArr[i].center - clusterArr[needHBId].center)
        upDisArr.append(d)
        upTimeArr.append(computerTime(clusterArr[i], clusterArr[needHBId], timeArr))
    # 更新时间距离与空间距离
    clusterDisMat[needHBId] = upDisArr
    clusterDisMat[:, needHBId] = upDisArr
    clusterTimeMat[needHBId] = upTimeArr
    clusterTimeMat[:, needHBId] = upTimeArr
    return clusterDisMat, clusterTimeMat
# 簇集合并函数
def getNewCluster(cluster1, cluster2, xyArr):
    # 第一个簇集包含的点集
    list1 = cluster1.indexSet
    # 第二个簇集包含的点集
    list2 = cluster2.indexSet
    # 新簇集所包含的点集,注意：两个簇集中不可能包含重复的点集，所以不考虑去重的现象
    newList = np.concatenate([list1,list2])
    # 计算新簇集中心点的坐标
    newCenter = np.mean(np.concatenate([xyArr[list1],xyArr[list2]]),axis=0)
    # 计算新簇集的起点
    newStartId = np.min(newList)
    # 计算新簇集的终点
    newEndId = np.max(newList)
    # 返回新簇集
    return AGNESCluster(newCenter,newStartId,newEndId,newList)
# 获得每一个簇集最近的簇集，以及最近簇集对应的距离
# clusterTimeMat簇集之间的时间距离矩阵
# clusterDisMat簇集之间的空间距离矩阵
# timeThreh簇集之间的时间阈值
# disThreh簇集之间的时间阈值
# clusterFloorId每个簇集对应的楼层
# 对比getClosestCluster的实现（与Indoor_STAGNES_DIS对比，此算法的实现慢一倍）
def getClosestCluster(clusterTimeMat,clusterDisMat,timeThreh,clusterFloorIds):
    # 返回的最近的簇集ID
    closedClusterId = []
    # 返回的最近的簇集距离
    closedClusterDis = []
    # 获得一共含有多少簇集
    n = clusterTimeMat.shape[0]
    for i in range(n):
        # 满足四个条件：时间阈值小于timeThreh（但是不能为0），空间阈值小于disThreh，楼层ID等于clusterFloorIds[i]
        # 如果满足条件，返回本身的距离值，否则返回np.inf（这里是一个技巧）
        disArr = np.where(((clusterTimeMat[i] < timeThreh) & (clusterTimeMat[i] != 0) & (clusterFloorIds == clusterFloorIds[i])), clusterDisMat[i],np.inf)
        closedClusterId.append(np.argmin(disArr))
        closedClusterDis.append(np.min(disArr))
    return np.array(closedClusterId), np.array(closedClusterDis)
# data 的第一列是unix时间戳，第二、三列是空间X,Y坐标，第四列是楼层ID
# disThreh 空间阈值
# timeThreh 时间阈值
def Indoor_STAGNES(data,classNum,timeThreh):
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 获得每一个轨迹点的时间数组
    timeArr=data[:,0]
    # 获得初始的簇集时间距离矩阵（此矩阵不断发生变化）
    clusterTimeMat = squareform(pdist(timeArr.reshape(-1, 1), metric='euclidean'))
    # 获得空间距离矩阵
    xyArr = data[:,1:3]
    # 获得初始的簇集空间距离矩阵（此矩阵不断发生变化）
    clusterDisMat = squareform(pdist(xyArr,metric='euclidean'))
    # 获得每一个簇集所在的楼层，Indoor的关键所在
    clusterFloorIds = data[:,3]
    # 将每一个点都初始化为一个簇集
    clusterArr = []
    for i in range(n):
        # 起初每个点都是一个簇，起始索引都是自己本身，且簇只包含自己本身
        clusterArr.append(AGNESCluster(xyArr[i], i, i, [i]))
    # 获得每一个簇集最近的"簇集"，以及相应的距离
    closedClusterId, closedClusterDis = getClosestCluster(clusterTimeMat,clusterDisMat,timeThreh,clusterFloorIds)
    while len(clusterArr) > classNum:
        # 获得需要合并的簇集ID
        needHBId = np.argmin(closedClusterDis)
        # 需要合并的两个簇
        currentCluster1 = clusterArr[needHBId]
        currentCluster2 = clusterArr[closedClusterId[needHBId]]
        # 合并后的新的簇集
        newCluster = getNewCluster(currentCluster1, currentCluster2, xyArr)
        # 赋予新的簇集
        clusterArr[needHBId] = newCluster
        # 删除被合并掉的簇集及所在的楼层
        del clusterArr[closedClusterId[needHBId]]
        clusterFloorIds = np.delete(clusterFloorIds, closedClusterId[needHBId])
        # 更新簇集距离矩阵
        clusterDisMat, clusterTimeMat = updateClusterMat(clusterArr, clusterDisMat, clusterTimeMat, timeArr, needHBId,
                                                         closedClusterId[needHBId])
        closedClusterId, closedClusterDis = getClosestCluster(clusterTimeMat, clusterDisMat, timeThreh,clusterFloorIds)
    return clusterArr
data = np.loadtxt("./data/cluster_unix_time_indoor.csv", delimiter=",")
start = time.clock()
clusterArr = Indoor_STAGNES(data,5,30)
end = time.clock()
print('finish all in %s' % str(end - start))
plotFeature(data[:,1:], clusterArr)
