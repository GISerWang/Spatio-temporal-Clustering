import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
import operator

# 计算X矩阵的距离矩阵
def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))
# 显示决策图
def plotReachability(data,eps):
    plt.figure()
    plt.plot(range(0,len(data)), data)
    plt.plot([0, len(data)], [eps, eps])
    plt.show()
# 显示分类的类别
def plotFeature(data,labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels == i)]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)
    plt.show()
def updateSeeds(seeds,core_PointId,neighbours,reach_dists,disMat,isProcess,minPts):
    # 计算core_PointId的核心距离
    tempDisMat=disMat[core_PointId,neighbours]
    # print(tempDisMat)
    core_dist=tempDisMat[np.argsort(tempDisMat)[minPts-1]]
    # 遍历core_PointId 的每一个邻居点
    for neighbour in neighbours:
        # 如果neighbour没有被处理过，计算改核心距离
        if(isProcess[neighbour]==-1):
            # 首先计算改点的针对core_PointId的可达距离
            new_reach_dist = max(core_dist, disMat[core_PointId][neighbour])
            if(np.isnan(reach_dists[neighbour])):
                reach_dists[neighbour]=new_reach_dist
                seeds[neighbour] = new_reach_dist
            elif(new_reach_dist<reach_dists[neighbour]):
                reach_dists[neighbour] = new_reach_dist
                seeds[neighbour] = new_reach_dist
    return seeds
def extract_dbscan(data,orders, reach_dists, eps):
    # 获得原始数据的行和列
    n,m=data.shape
    # reach_dists[orders] 将每个点的可达距离，按照有序列表排序（即输出顺序）
    # np.where(reach_dists[orders] <= eps)[0]，找到有序列表中小于eps的点的索引，即对应有序列表的索引
    reach_distIds=np.where(reach_dists[orders] <= eps)[0]
    # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
    pre=reach_distIds[0]-1
    clusterId=0
    labels=np.full((n,),-1)
    for current in reach_distIds:
        # 正常来说：current的值的值应该比pre的值多一个索引。如果大于一个索引就说明不是一个类别
        if(current-pre!=1):
            # 类别+1
            clusterId=clusterId+1
        labels[orders[current]]=clusterId
        pre=current
    return labels
# data 的第一列是unix时间戳，剩余列是空间坐标数据
# eps1 空间邻域
# eps2 时间邻域
# minPts 满足双邻域的最少点的个数
def ST_OPTICS(data,eps1,eps2,minPts):
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 计算时间距离矩阵
    timeDisMat = compute_squared_EDM(data[:,0].reshape(n, 1))
    # 获得距离矩阵
    orders = []
    disMat = compute_squared_EDM(data[:,1:])
    # 将每一个点的可达距离未定义
    reach_dists= np.full((n,), np.nan)
    # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
    core_points_index = np.where(np.sum(np.where((disMat <= eps1) & (timeDisMat<=eps2), 1, 0), axis=1) >= minPts)[0]
    # 初始化类别，-1代表未分类。
    isProcess = np.full((n,), -1)
    # 遍历所有的核心点
    for pointId in core_points_index:
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if (isProcess[pointId] == -1):
            # 将点pointId标记为当前类别(即标识为已操作)
            isProcess[pointId] = 0
            orders.append(pointId)
            # 寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbours = np.where((disMat[:, pointId] <= eps1) & (timeDisMat[:, pointId]<=eps2) & (isProcess == -1))[0]
            seeds = dict()
            seeds=updateSeeds(seeds,pointId,neighbours,reach_dists,disMat,isProcess,minPts)
            while len(seeds)>0:
                nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]
                del seeds[nextId]
                isProcess[nextId] = 0
                orders.append(nextId)
                # 寻找newPoint种子点eps邻域（包含自己）
                # 这里没有加约束isProcess == -1，是因为如果加了，本是核心点的，可能就变成了非和核心点
                queryResults = np.where((disMat[:, nextId] <= eps1) & (timeDisMat[:, nextId]<=eps2))[0]
                if len(queryResults) >= minPts:
                    seeds=updateSeeds(seeds,nextId,queryResults,reach_dists,disMat,isProcess,minPts)
    return orders,reach_dists

data = np.loadtxt("data/cluster_unix_time.csv", delimiter=",")
start = time.clock()
orders,reach_dists=ST_OPTICS(data,np.inf,500,30)
end = time.clock()
print('finish all in %s' % str(end - start))
plotReachability(reach_dists[orders],3)
labels=extract_dbscan(data,orders,reach_dists,3)
plotFeature(data[:,1:],labels)
