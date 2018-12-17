# -*- coding: utf-8 -*-
# Warped K-Means
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import time
# 显示类别
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

# 将样本X从簇i移动到簇j时产生的增量的变化
# 如果增量为负，说明将i移动到簇j是有益的
# x是样本X的索引
# i是第i个簇集的索引
# j是第j个簇集的索引
# data是序列集合
# ns是簇集的个数集合,ns[i]代表第i个簇集具有多少元素
# us是簇集的平均点坐标的集合,us[i]代表第i个簇集中心坐标是多少
def computeJ(data,ns,us,x,i,j):
    # 计算增量deltaJ的变化情况
    return (ns[j] / (ns[j] + 1)) * (LA.norm(data[x] - us[j], 1) ** 2) - (ns[i] / (ns[i] + 1)) * (
                LA.norm(data[int(i)] - us[i], 1) ** 2)
# 将样本X从簇i移动到簇j时,假设SOE增量为负，ns，us，J的变化
# ns是簇集的个数集合,ns[i]代表第i个簇集具有多少元素
# us是簇集的平均点坐标的集合,us[i]代表第i个簇集中心坐标是多少
# J 是现有的J
# x是样本X的索引
# i是第i个簇集的索引
# j是第j个簇集的索引
def updateUNJ(data,us,ns,J,deltaJ,x,i,j):
    # 第i个簇集的均值变化
    us[i] = us[i] - (data[x] - us[i]) / (ns[i] - 1)
    # 第j个簇集的均值变化
    us[j] = us[j] + (data[x] - us[j]) / (ns[j] + 1)
    # 第i个簇集的个数减1
    ns[i] = ns[i] - 1
    # 第j个簇集的个数加1
    ns[j] = ns[j] + 1
    J=J+deltaJ
    return us,ns,J
# 初始化类别
# data是数据集
# k是初始化为k个类别
def TS(data,k,n):
    # 初始化边界对象
    bs=np.zeros((k,),dtype=np.int)
    # ls 是一个累加，第i个数被前i-1个数作用
    ls=np.zeros((n,))
    # 初始化labels
    labels=np.zeros((n,),dtype=np.int)
    # 迭代计算ls
    for i in range(1,n):
        ls[i]=ls[i-1]+LA.norm(data[i]-data[i-1],1)
    r=ls[n-1]/k
    i=0
    # 迭代计算边界对象
    for j in range(1,k+1):
        while(r*(j-1)>ls[i]):
            i=i+1
        bs[j-1]=i
    # 计算每一个点的label
    for j in range(k-1):
       labels[bs[j]:bs[j+1]]=j
    labels[bs[k-1]:] = k-1
    return bs,labels
# WKM算法的主函数，
# data是按顺序排列的数据集
# k：指定具有多少个类别
# rho：此参数一般指定为0，用于控制迭代的数量，0:迭代全部，1:只迭代两端
def WKM(data,k,rho):
    # 获得数据具有多少行和多少列
    n,m = data.shape
    # 初始化边界对象，和相应的labels
    bs,labels= TS(data,k,n)
    # 初始化每个类别的us和ns
    us = np.zeros((k, 2))
    ns = np.zeros((k,))
    for j in range(0,k-1):
        ns[j]=bs[j+1]-bs[j]
        us[j]=np.average(data[bs[j]:bs[j+1],:],axis=0)
    ns[k-1]=n-bs[k-1]
    us[k-1] = np.average(data[bs[k-1]:n, :], axis=0)
    # 计算误差
    error=np.sum(np.sqrt(np.sum(np.power(data-us[labels], 2), axis=1)))
    # 迭代聚类，WKM算法
    for j in range(k):
        # 如果当前类别不是第0类
        if j>0:
            # 获得该类的的起始索引
            first=bs[j]
            # 获得该类别的终止索引（这个循环只迭代一半）
            last=first+ int(ns[j]/2*(1-rho))
            # 开始迭代，从x开始迭代
            x=first
            while(x<=last):
                # 计算deltaJ
                deltaJ=computeJ(data,ns,us,x,j,j-1)
                # 如果类别个数小于1，或者deltaJ>0
                # 跳出循环
                if((ns[j]<=1) or deltaJ>=0):
                    break
                # 更新边界索引集合
                bs[j] = bs[j] + 1
                # 更新us, ns, error
                us, ns, error = updateUNJ(data,us, ns, error, deltaJ, x, j, j - 1)
                x = x + 1
        if j<k-1:
            # 获得该类的的终止索引
            last = bs[j+1]-1
            # 获得该类别的起始索引（这个循环只迭代一半）
            first = last-int(ns[j]/2*(1-rho))
            # 开始迭代，从x开始迭代
            x = last
            while (x >= first):
                # 计算deltaJ
                deltaJ = computeJ(data,ns,us,x,j,j+1)
                # 如果类别个数小于1，或者deltaJ>0
                # 跳出循环
                if ((ns[j] <= 1) or deltaJ >= 0):
                    break
                # 更新边界索引集合
                bs[j+1] = bs[j+1] - 1
                # 更新us, ns, error
                us, ns, error = updateUNJ(data,us, ns, error, deltaJ, x, j, j + 1)
                x=x-1
    return bs
# 通过边界对象提取聚类类别
def extract_clustering(bs,n):
    labels = np.zeros((n,), dtype=np.int)
    # 获取一共有多少类别
    k=len(bs)
    # 循环给labels复制，赋予每一个类别
    for j in range(k - 1):
        labels[bs[j]:bs[j + 1]] = j
    labels[bs[k - 1]:] = k - 1
    return labels
data = np.loadtxt("data/cluster.csv", delimiter=",")
start = time.clock()
bs= WKM(data,4,0)
end = time.clock()
print('finish all in %s' % str(end - start))
labels=extract_clustering(bs,data.shape[0])
plotFeature(data,labels)