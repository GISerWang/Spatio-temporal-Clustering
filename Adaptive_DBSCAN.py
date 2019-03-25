import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
# 计算距离矩阵
def compute_squared_EDM(X):
  return squareform(pdist(X,metric='euclidean'))

# 判断pt点是否为核心点
# ptIndex为点pt的索引
# disMat距离矩阵
# eps为pt的半径（每个点都不一样）
# minPts为pt的圆中的点个数（每个点都不一样）
def isCorePoint(ptIndex,disMat,eps,minPts):
    if np.sum(np.where(disMat[ptIndex] <= eps, 1, 0)) >= minPts:
        return True
    return False

# Adaptive_DBSCAN算法核心过程
def Adaptive_DBSCAN(data,epsArr,minPtsArr):
    # 获得距离矩阵
    disMat = compute_squared_EDM(data)
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    clusterId = 0
    # 遍历所有轨迹点寻找簇集
    for ptIndex in range(n):
        # 如果某轨迹点已经有类别，直接continue
        if labels[ptIndex] != -1:
            continue
        # 如果某轨迹点不是核心点，直接continue(因为簇集的产生是由核心点控制)
        if not isCorePoint(ptIndex,disMat,epsArr[ptIndex],minPtsArr[ptIndex]):
            continue
        # 首先将点pointId标记为当前类别(即标识为已操作)
        labels[ptIndex] = clusterId
        # 然后寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
        neighbour = np.where((disMat[:, ptIndex] <= epsArr[ptIndex]) & (labels == -1))[0]
        seeds = set(neighbour)
        while len(seeds) > 0:
            # 弹出一个新种子点
            newPoint = seeds.pop()
            # 将newPoint标记为当前类
            labels[newPoint] = clusterId
            # 寻找newPoint种子点eps邻域（包含自己）
            queryResults = np.where(disMat[:,newPoint] <= epsArr[newPoint])[0]
            if len(queryResults) >= minPtsArr[newPoint]:
                # 将邻域内且没有被分类的点压入种子集合
                for resultPoint in queryResults:
                    if labels[resultPoint] == -1:
                        seeds.add(resultPoint)
        # 簇集生长完毕，寻找到一个类别
        clusterId = clusterId + 1
    return labels

# 将分类后的数据可视化显示
def plotFeature(data, labels_):
    clusterNum=len(set(labels_))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1], c=colorSytle, s=12)
    plt.show()

# 加载数据
data = np.loadtxt("data/cluster.csv", delimiter=",")
start = time.clock()
# 获得一共具有多少条数据
n = data.shape[0]
# 构造每一个轨迹点的半径及包含的轨迹点个数
epsArr = np.full((n),2)
minPtsArr = np.full((n),15)
# Adaptive_DBSCAN聚类并返回标识；ϵ=2，且MinPts=15
# 第一个参数是每一个轨迹点的坐标
# 第二个参数是每一个轨迹点自己的半径
# 第三个参数是每一个轨迹点自己的最小包含点
labels=Adaptive_DBSCAN(data,epsArr,minPtsArr)
end = time.clock()
print('finish all in %s' % str(end - start))
plotFeature(data, labels)