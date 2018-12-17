import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time

# 计算X矩阵的距离矩阵
def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))
# data 的第一列是unix时间戳，剩余列是空间坐标数据
# eps1 空间邻域
# eps2 时间邻域
# minPts 满足双邻域的最少点的个数
def ST_DBSCAN(data,eps1,eps2,minPts):
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 计算时间距离矩阵
    timeDisMat=compute_squared_EDM(data[:,0].reshape(n, 1))
    # 获得空间距离矩阵
    disMat = compute_squared_EDM(data[:,1:])
    # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
    # 注意：np.where()的两种用法（搜索、替换功能）
    core_points_index = np.where(np.sum(np.where((disMat <= eps1) &(timeDisMat<=eps2), 1, 0), axis=1) >= minPts)[0]
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    clusterId = 0
    # 遍历所有的核心点
    for pointId in core_points_index:
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if (labels[pointId] == -1):
            # 将点pointId标记为当前类别(即标识为已操作)
            labels[pointId] = clusterId
            # 寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbour=np.where((disMat[:, pointId] <= eps1) & (timeDisMat[:, pointId] <= eps2) & (labels==-1))[0]
            seeds = set(neighbour)
            # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
            while len(seeds) > 0:
                # 弹出一个新种子点
                newPoint = seeds.pop()
                # 将newPoint标记为当前类
                labels[newPoint] = clusterId
                # 寻找newPoint种子点eps邻域（包含自己）
                queryResults = set(np.where((disMat[:,newPoint]<=eps1) & (timeDisMat[:, newPoint] <= eps2) )[0])
                # 如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
                if len(queryResults) >= minPts:
                    # 将邻域内且没有被分类的点压入种子集合
                    for resultPoint in queryResults:
                        if labels[resultPoint] == -1:
                            seeds.add(resultPoint)
            # 簇集生长完毕，寻找到一个类别
            clusterId = clusterId + 1
    return labels
def plotFeature(data, labels_):
    clusterNum=len(set(labels_))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange','#BC8F8F','#8B4513','brown']
    ax = fig.add_subplot(111)
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1], c=colorSytle, s=20)
    plt.show()
data = np.loadtxt("data/cluster_unix_time.csv", delimiter=",")
start = time.clock()
labels=ST_DBSCAN(data,3,500,30)
end = time.clock()
print('finish all in %s' % str(end - start))
plotFeature(data[:,1:], labels)