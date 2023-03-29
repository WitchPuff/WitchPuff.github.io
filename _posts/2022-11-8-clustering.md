---
layout: post
title:  "Machine Learning: Clustering"
date:   2022-12-18 01:36:30
categories: Courses
tag: Machine Learning,AI,K-Means,GMM,Clustering

---

* TOC
{:toc}
# 机器学习：K-Means&GMM学习笔记

## 1、K-Means

K-Means算法是最常用的聚类算法，主要思想是：在给定K值和K个初始类簇中心点的情况下，把每个点(即样本数据)分到离其最近的类簇中心点所代表的类簇中，所有点分配完毕之后，根据一个类簇内的所有点重新计算该类簇的中心点(取平均值)，然后再迭代的进行分配点和更新类簇中心点的步骤，直至类簇中心点的变化很小，或者达到指定的迭代次数。

### 1）算法思路

1. 初始化中心点；

2. 在第j次迭代中，对于每个样本点，使用欧几里得距离计算样本间的距离，选取最近的中心点，归为该类；

3. 更新中心点为每类的均值；

4. 重复(2)(3)迭代更新，直至误差小到某个值或者到达一定的迭代步数，误差不变。


空间复杂度：o(N)，时间复杂度：o(I*K\*N)，其中，N为样本点个数，K为中心点个数，I为迭代次数。在解决MNIST聚类问题时，由于标签为10个数字，K默认为10。

### 2）初始化中心点

#### 1. 随机选取k个中心点

##### 2. 最大距离选取中心点

随机初始化质心可能导致算法迭代很慢，K-means++是对K-mean随机初始化质心的一个优化，具体步骤如下：

1. 随机选取一个点作为第一个聚类中心。
2. 计算所有样本与第一个聚类中心的距离。
3. 选择出上一步中距离最大的点作为第二个聚类中心。
4. 迭代：计算所有点到与之最近的聚类中心的距离，选取最大距离的点作为新的聚类中心。
5. 终止条件：直到选出了这k个中心。

### 3）关键代码展示

实现了Kmeans类。

```python
class KMEANS:
    def __init__(self, n_clusters=10, max_iter=20,device = torch.device("cuda:0"),is_random=True):
        """init

        Args:
            n_clusters (int, optional): the number of clusters. Defaults to 10.
            max_iter (int, optional): the maximum iterations. Defaults to 20.
            device (torch.device, optional): use cuda or gpu. Defaults to torch.device("cuda:0").
            is_random (bool, optional): randomly initialize the center points or not. Defaults to True.
        """
        self.n_clusters = n_clusters
        self.labels = None # the labels of input data
        self.centers = None    # the center points
        self.max_iter = max_iter
        self.device = device
        self.is_random = is_random
        
    def acc(self,y_true:np.array, y_pred:np.array):
        """
        参考: https://blog.csdn.net/qq_42887760/article/details/105720735
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.array(ind).T
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
    def initial(self,x):
        """initialize the center points

        Args:
            x (torch.Tensor): the data of trainset, trainx, shape(60000,784)
        """
        if self.is_random:
            # 随机选择初始中心点
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
            self.centers = x[init_row].to(self.device)
        else:
            # 最大距离初始化
            # 只随机选择一个初始中心点
            first = torch.randint(0,x.shape[0],(1,))
            cen = x[first]
            centers = torch.empty((0, x.shape[1])).to(self.device)
            centers = torch.cat([centers, cen], (0))
            dists = torch.empty((x.shape[0],0)).long().to(self.device) # 记录每个样本与中心点的距离
            for i in range(1, self.n_clusters): # 要选n个中心点
                for cen in centers: 
                    # 对于每个样本点，计算其与每个中心点的距离
                    dist = torch.sum(torch.mul(x-cen,x-cen),(1))
                    dists = torch.cat([dists,dist.unsqueeze(1)],(1))
                # 选出离最近中心点距离最大的样本点，作为下一个中心点
                cen = x[torch.argmax(torch.min(dists,dim=1).values)]
                centers = torch.cat([centers, cen.unsqueeze(0)], (0))
            self.centers = centers

        
        
    def fit(self, x, y, testx,testy):
        """fit the model

        Args:
            x (Tensor): trainx
            y (Tensor): trainy, the labels of trainset
            testx (Tensor): testx
            testy (Tensor): testy, the labels of testset
        """
        for i in range(self.max_iter):
            print(f"Epoch {i+1}")
            # 分类
            self.labels = self.classify(x)
            # 更新中心点
            self.update_center(x)
            # 计算trainset与testset的精度
            accu_train.append(self.acc(np.array(y.cpu()), np.array(self.labels.cpu())))
            print(f"Accuracy of train data: {accu_train[-1]}", end=', ')
            accu_test.append(self.acc(np.array(testy.cpu()), np.array(self.classify(testx).cpu())))
            print(f"Accuracy of test data: {accu_test[-1]}")



    def classify(self, x):
        """classify

        Args:
            x (tensor): train x, shape(60000,784)
        """
        dists = torch.empty((x.shape[0],0)).long().to(self.device) # 对应每个样本与各中心点的距离
        for cen in self.centers:
            dist = torch.sum(torch.mul(x-cen,x-cen),(1))
            dists = torch.cat([dists,dist.unsqueeze(1)],(1))
        return torch.argmin(dists,dim=1)


    def update_center(self, x):
        """update the center points

        Args:
            x (tensor): train x, shape (60000,784)
        """
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            cluster_samples = x[self.labels == i] # 该族样本点
            # 取该族所有样本点的均值，取代当前中心点
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

```



## 2、EM算法与GMM模型

### 1）EM算法

EM（Expectation Maximization）算法是一种常用的数据挖掘和机器学习方法，用于估计带隐变量的概率模型的参数。它的基本思想是通过迭代的方式不断优化模型的参数，以达到模型对数据的最佳拟合。

EM 算法主要包括两个步骤：期望（Expectation）步骤和极大化（Maximization）步骤。在期望步骤中，通过计算数据点在隐变量的某种可能状态下的期望值来估计隐变量的分布。在极大化步骤中，通过极大化隐变量的期望值来优化模型的参数。

EM 算法可以用来解决多种问题，例如高斯混合模型的参数估计、图像分割和语音识别等。它的优点在于可以用于处理带隐变量的模型，并且可以通过迭代的方式不断优化模型的参数。

### 2）GMM模型

GMM（高斯混合模型）是一种概率模型，用于对一个数据集进行拟合。它假设数据是由若干个高斯分布组成的混合体，并使用最大似然估计或最大后验概率估计来估计每个分布的参数。

在数学上，一个由 $K$ 个高斯分布组成的 GMM 可以表示为：

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$
其中 $\pi_k$ 表示第 $k$ 个高斯分布在混合中的权重，$\mathcal{N}(x | \mu_k, \Sigma_k)$ 表示第 $k$ 个高斯分布的概率密度函数，$\mu_k$ 和 $\Sigma_k$ 分别表示第 $k$ 个高斯分布。

### 3）算法思路

EM算法是用来训练混合模型（例如GMM）的一种常见方法。具体来说，EM算法通过迭代地执行下面两个步骤来训练GMM模型：

1. **E步骤（期望步骤）**：对于给定的模型参数，计算每个观测值属于每个混合成分的概率（预测标签）。
2. **M步骤（极大似然步骤）**：使用计算出的每个观测值属于每个混合成分的概率来更新模型参数（均值$\mu$，协方差矩阵$\Sigma$、权重$\pi$），使得模型对观测数据的似然最大。
3. 检查模型是否收敛，计算当前的模型参数与上一次迭代的模型参数之间的差异，如果差异小于某个阈值，则认为已经收敛。收敛则结束迭代，反之继续从E步骤开始迭代。

#### 1. E Step

在E步骤中，我们需要计算每个观测值属于每个混合成分的概率。这通常使用给定的模型参数（例如每个混合成分的均值、协方差和混合系数）以及观测数据的高斯分布概率密度函数来实现。

具体来说，对于每个观测值x，我们需要计算其属于每个混合成分的概率。这可以使用下面的公式计算：
$$
p(z_i = j | x_i, θ) = π_j * N(x_i | μ_j, Σ_j)
$$
其中，$z_i$表示观测值$x_i$属于的混合成分的编号，θ表示模型参数（包括每个混合成分的均值、协方差和混合系数），$π_j$表示混合成分j的混合系数，$N(x_i | μ_j, Σ_j)$表示观测值$x_i$的高斯分布概率密度函数，其中$μ_j$和$Σ_j$分别表示混合成分j的均值和协方差。

#### 2. M Step

在M步骤中，我们使用极大似然估计来更新模型参数。具体来说，我们需要求解下面的优化问题：
$$
maximize L(θ) = ∑_{i=1}^{n} log ∑_{j=1}^{m} π_j * N(x_i | μ_j, Σ_j)
$$
其中，$n$表示观测数据的数量，$m$表示混合成分的数量，$θ$表示模型参数（包括每个混合成分的均值、协方差和混合系数），$π_j$表示混合成分j的混合系数，$N(x_i | μ_j, Σ_j)$表示观测值$x_i$的高斯分布概率密度函数，其中$μ_j$和$Σ_j$分别表示混合成分j的均值和协方差。

我们可以使用拉格朗日乘数法来求解上述优化问题。具体来说，对于每个混合成分$j$，我们需要更新其均值$μ_j$、协方差$Σ_j$和混合系数$π_j$，使得似然函数$L(θ)$最大。

1. 对于每个混合成分$j$，更新其均值$μ_j$：
   $$
   μ_j = ∑_{i=1}^{n} p(z_i = j | x_i, θ) * x_i / ∑_{i=1}^{n} p(z_i = j | x_i, θ)
   $$
   其中，$n$表示观测数据的数量，$p(z_i = j | x_i, θ)$表示观测值$x_i$属于混合成分$j$的概率，$x_i$表示第$i$个观测值。

2. 对于每个混合成分$j$，更新其协方差$Σ_j$：
   $$
   Σ_j = ∑_{i=1}^{n} p(z_i = j | x_i, θ) * (x_i - μ_j)(x_i - μ_j)^T / ∑_{i=1}^{n} p(z_i = j | x_i, θ)
   $$

3. 对于每个混合成分$j$，更新其混合系数$π_j$：
   $$
   π_j = ∑_{i=1}^{n} p(z_i = j | x_i, θ) / n
   $$

### 4）初始化模型参数

在使用EM算法训练GMM模型时，需要初始化模型参数。常用的方法包括：

1. 随机初始化：将模型参数随机初始化，并开始迭代训练。
2. K-means聚类：使用K-means聚类算法将数据分成若干类，然后将每一类的均值作为GMM的均值，协方差设为协方差，并将每一类的权重设为相同值。
3. 统计量初始化：使用数据的均值和协方差矩阵作为GMM的均值和方差，并将每一类的权重设为相同值。

### 5）协方差矩阵格式

GMM模型允许每个混合成分使用不同的协方差矩阵。具体来说，GMM模型可以使用以下三种协方差矩阵格式：

1. 全局协方差（full covariance）：每个混合成分使用相同的协方差矩阵。这种协方差矩阵格式在多维情况下可以很好地捕捉数据的复杂关系，但是它的计算代价比较大。
2. 对角协方差（diagonal covariance）：每个混合成分使用一个对角矩阵作为协方差矩阵，即只有对角线上的元素不为零。这种协方差矩阵格式计算代价较小，但是可能不能很好地捕捉数据的复杂关系。
3. 共轭协方差（spherical covariance）：每个混合成分使用一个单位矩阵乘以一个单独的方差值作为协方差矩阵。这种协方差矩阵格式计算代价最小，但是对于多维数据可能不能很好地捕捉其复杂关系。

