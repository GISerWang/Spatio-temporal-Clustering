# Spatio-temporal Clustering

## 1代码介绍
### 1.1 数据介绍
* data/cluster_time：按时间顺序排列的用户行为轨迹
* data/cluster_unix_time：按时间顺序(时间已经转换为时间戳)排列的用户行为轨迹
* data/cluster_unix_time：按时间顺序(时间约束隐含，没有时间字段)排列的用户行为轨迹

### 1.2 聚类算法

* MYDBSCAN：基于密度的聚类DBSCAN（Density-Based Spatial Clustering of Applications with Noise）算法的底层实现
* MYOPTICS：基于密度的聚类OPTICS（Ordering points to identify the clustering structure）算法的底层实现
* MYKMeans：基于划分的聚类KMeans算法的底层实现
* MYCFSFDP：基于划分和密度的聚类CFSFDP（Clustering by fast search and find of density peaks）算法的底层实现


### 1.3 时空聚类算法

* ST-DBSCAN：基于DBSCAN改造的时空聚类算法
* ST-OPTICS：基于OPTICS改造的时空聚类算法
* ST-AGNES_DIS：基于凝聚层次聚类（AGNES）改造的时空聚类算法（用距离做阈值，自动生成聚类个数）
* ST-AGNES_SUM：基于凝聚层次聚类（AGNES）改造的时空聚类算法（使用聚类个数做阈值）
* WKM：WKM（Warped K-Means）基于K-Means改造的时空聚类算法（使用聚类个数做阈值）