import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
def compute_squared_EDM(X):
  return squareform(pdist(X,metric='euclidean'))
def plotFeature(data, labels_):
    clusterNum=len(set(labels_))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown','#BC8F8F','#8B4513','#FFF5EE']
    ax = fig.add_subplot(111)
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1], c=colorSytle, s=12)
    plt.show()
# 获得新簇集的起始索引
def getNewCluster(classStart,mindisIndex,dataLength,disLength):
    # classStart[mindisIndex]是新簇集的开始坐标
    # 如果距离最小值索引是距离集合的最后一个
    # 新簇集结束坐标是data的最后一个值
    if (mindisIndex == disLength - 1):
        return classStart[mindisIndex],dataLength
    # 如果距离最小值索引不是距离集合的最后一个
    # 新簇集结束坐标为mindisIndex + 2，因为Python切片不包含索引为mindisIndex + 2的数据
    # Python切片，包含前面，不包含后面
    return classStart[mindisIndex],classStart[mindisIndex + 2]
# 获得新簇集的上一个簇集的起始索引
def getPreCluster(classStart,mindisIndex):
    # 如果距离最小值索引是距离集合的第0个元素
    # 新簇集不存在上一个集合，因此返回None
    if (mindisIndex == 0):
        return None
    # 如果距离最小值索引不是距离集合的第0个元素
    # 新簇集即为：classStart[mindisIndex - 1]:classStart[mindisIndex]
    return classStart[mindisIndex - 1],classStart[mindisIndex]
# 获得新簇集的下一个簇集的起始索引
def getNextCluster(classStart,mindisIndex,dataLength,disLength):
    # 如果距离最小值索引是距离集合的最后一个元素
    # 新簇集不存在下一个集合，因此返回None
    if (mindisIndex == disLength - 1):
        return None
    # 如果距离最小值索引是距离集合的倒数第二个元素
    # 新簇集的下一个集合应该为簇集集合中的最后一个簇集
    elif (mindisIndex == disLength - 2):
        return classStart[mindisIndex + 2],dataLength
    # 如果距离最小值索引不是距离集合的倒数第二个元素，也不是倒数第一个
    # 新簇集即为：classStart[mindisIndex + 2]:classStart[mindisIndex + 3]
    else:
        return classStart[mindisIndex + 2],classStart[mindisIndex + 3]
def updateNextCluster(dataDisMat,classDis,classStart,mindisIndex,dataLength,disLength):
    # 获得新产生的簇集集合
    newCluster = getNewCluster(classStart,mindisIndex,dataLength,disLength)
    # 获得与新簇集相邻的下一个簇集
    nextCluster = getNextCluster(classStart,mindisIndex,dataLength,disLength)
    # 如果preCluster为None，说明新簇集是最后一个簇集，不存在下一个簇集
    if(nextCluster is None):
        return classDis
    # 如果preCluster不为None，查找新簇集与后一个簇集的距离矩阵
    nextDisMat = dataDisMat[newCluster[0]:newCluster[1],nextCluster[0]:nextCluster[1]]
    n, m = nextDisMat.shape
    # 计算两个簇集的平均距离（这里也可以是最小距离，也可以是最大距离）
    nextDis = np.sum(nextDisMat) / (n * m)
    # 将新簇集与后一个簇集的距离赋值给mindisIndex + 1的位置上
    classDis[mindisIndex + 1] = nextDis
    return classDis
def updatePreCluster(dataDisMat,classDis,classStart,mindisIndex,dataLength,disLength):
    # 获得新产生的簇集集合
    newCluster = getNewCluster(classStart,mindisIndex,dataLength, disLength)
    # 获得新簇集的前一个集合
    preCluster = getPreCluster(classStart,mindisIndex)
    # 如果preCluster为None，说明新簇集就是第一个簇集
    if (preCluster is None):
        return classDis
    # 如果preCluster不为None，查找新簇集与前一个簇集的距离矩阵
    preDisMat = dataDisMat[preCluster[0]:preCluster[1],newCluster[0]:newCluster[1]]
    n, m = preDisMat.shape
    # 计算两个簇集的平均距离（这里也可以是最小距离，也可以是最大距离）
    preDis = np.sum(preDisMat) / (n * m)
    # 将新簇集与前一个簇集的距离赋值给mindisIndex - 1的位置上
    classDis[mindisIndex - 1] = preDis
    return classDis
# 时空凝聚聚类算法
def ST_AGNES(data,dThred):
    # 将data沿着y轴滚动一次
    rollData = np.roll(data, 1, axis=0)
    # 计算向量之间的距离矩阵
    dataDisMat = compute_squared_EDM(data)
    # 计算一共有多少条数据
    dataLength=len(data)
    # la.norm(rollData - data, axis=1) 求相邻数据之间的欧式距离，默认为2范数
    # np.delete()删除距离数组中的第一个元素
    # classDis中的第i个元素代表着第i与i+1类别之间的距离
    classDis = np.delete(la.norm(rollData - data, axis=1), 0)
    # 记录每一个簇集起始点的索引
    # classStart数组的长度一定比classDis数组的长度多1
    # classStart数组的长度等于簇集的个数
    # 初始化簇集起始点的索引，起始点，每一个点都是一个簇集
    classStart = np.arange(0, len(data))
    # 获取类距离集合中最小数值
    minDis = np.min(classDis)
    # 当簇集中任意两个距离大于dThred时，聚类停止
    while (minDis < dThred):
        # 寻找到最小距离的id
        # 需要合并的簇集mindisIndex 和mindisIndex + 1
        mindisIndex = np.argmin(classDis)
        disLength=len(classDis)
        # 簇集向前扩展，即合并后的簇集，重新计算新簇集与前一个簇集的距离
        classDis=updatePreCluster(dataDisMat,classDis,classStart,mindisIndex,dataLength,disLength)
        # 簇集向后扩展，即合并后的簇集，重新计算新簇集与后一个簇集的距离并
        classDis=updateNextCluster(dataDisMat,classDis,classStart,mindisIndex,dataLength,disLength)
        # 在簇集合并当中mindisIndex + 1的簇集被mindisIndex的簇集合并掉
        # 因此删掉第mindisIndex + 1簇集的初始索引
        classStart = np.delete(classStart, mindisIndex + 1)
        # 在簇集合并当中mindisIndex + 1的簇集被mindisIndex的簇集合并掉
        # 并将新的簇集距离赋值给mindisIndex - 1 和 mindisIndex + 1
        # 因此去掉索引为mindisIndex的距离
        classDis = np.delete(classDis, mindisIndex)
        # 当只有一个簇集的时候合并结束
        if(len(classStart)==1):
            break
        # 重新计算最小的距离
        minDis = np.min(classDis)
    return classStart
# 此方法通过classStart 生成相应的簇集
# classStart的长度标志着簇集的个数
# classStart中每一个元素都标志着簇集的开始点坐标
def extract_clusters(data,classStart):
    # 将每一个簇集的类别初始化为-1
    labels=np.full((len(data),),-1)
    # 循环给簇集的每一个元素赋值
    for i in range(len(classStart)):
        if(i==(len(classStart)-1)):
            labels[classStart[i]:] = i
        else:
            labels[classStart[i]:classStart[i+1]]=i
    return labels
# 加载聚类数据
data = np.loadtxt("data/cluster.csv", delimiter=",")
# 执行聚类算法，得到聚类类别的初始点坐标索引
start = time.clock()
classStart=ST_AGNES(data,8)
end = time.clock()
# 获得每一个点的聚类类别
labels=extract_clusters(data,classStart)
# 显示图形
print('finish all in %s' % str(end - start))
plotFeature(data,labels)