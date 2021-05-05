# Support Vector Machine

## 1 导论
### 1.1基本概念：
支持向量机（support vector machines, SVM）是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；SVM还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

### 1.2 算法原理：

SVM学习的基本想法是基于训练集D在样本空间中找到一个能够将训练样本分开，并且几何间隔最大的划分超平面。对于线性可分的数据集来说，这样的超平面有无穷多个（即感知机），但是几何间隔最大的分离超平面却是唯一的。这样的超平面对训练样本局部扰动的"容忍"最好，产生的分类结果是最鲁棒的(Robust)，泛化能力最强。

![划分超平面](https://github.com/wangchuan-hub/Support-Vector-Machine/blob/main/svm%E5%88%92%E5%88%86%E8%B6%85%E5%B9%B3%E9%9D%A2%202021-04-19%20181503.png)

支持向量机可分为：线性可分支持向量机，线性支持向量机，非线性支持向量机。

* 当训练数据线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机。
* 当训练数据近似线性可分时，通过软间隔最大化，学习一个线性分类器，即线性支持向量机。
* 当训练数据线性不可分时，通过使用核技巧和软间隔最大化，学习非线性支持向量机。



## 2 线性可分支持向量机与硬间隔最大化

### 2.1 模型构建：

在样本空间中，划分超平面由 $w^Tx + b = 0$表示，其中 w 为法向量,决定超平面的方向，b 为位移项，决定超平面与原点的距离。
* 第一步
$$
\underset{w,b}{max}\ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i+b|\\\\
s.t.\ \ \ w^Tx_i + b > 0,y_i = +1\\\\
w^Tx_i + b < 0,y_i = -1\\\\
(i=1,2……N)
$$
其中 $ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i + b|$ 表示训练样本中与超平面最近的点 到 超平面的距离 
$ \underset {w,b}{max}\ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i + b| $ 则表示目标函数为使这个距离 也就是 "间隔" 最大 
约束条件表示样本能够正确分类

* 第二步：
$$
\underset{w,b}{max}\frac{1}{||w||} \underset{x_i}{min}\ y_i(w^Tx_i + b) \\\\
s.t.\ \ y_i(w^Tx_i + b) > 0 \ \ \ i = 1,2...N
$$

存在 r > 0 使得 $ \underset{x_i}{min}\ y_i(w^Tx_i + b)=r $ 并可以将 r 取为1 则 $\underset{w,b}{max}\frac{1}{||w||} \underset{x_i}{min}\ y_i(w^Tx_i + b)$ 可写为 $ \underset{w,b}{max} \frac{1}{||w||}$

约束条件变为  $y_i(w^Tx_i + b) ≥ 1$, 得到最终的约束优化问题(原问题):
$$
\underset{w,b}{min}\frac{1}{2}||w||^2 \\\\\
s.t. \ \  y_i(w^Tx_i + b） ≥ 1 \\\\ 
(i =1, 2……N)
$$

上式本身是一个个凸二次规划 (convex quadratic programming) 问题，能直接用现成的优化计算包求解，但我们可

以有更高效的办法.即转化为对偶问题。

### 2.2  原问题与对偶问题

* 由原问题构建拉格朗日函数:


$$
L(w,b,\lambda) = \frac{1}{2}||w||^2 + \sum_{i=1}^{N}\lambda_i[1-y_i(w^Tx_i + b)]\\\\\
其中\ \lambda_i≥0，1-y_i(w^Tx_i + b)≤0
$$

* 将带约束形式的原问题写为无约束形式:
$$
\underset{w,b}{min}\underset{\lambda}{max}L(w,b,\lambda) \\\\\
s.t. \ \ \lambda_i ≥ 0
$$

* 转化为对偶问题：
$$
\underset{\lambda}{max}\ \underset{w,b}{min}\ L(w,b,\lambda) \\\\\
s.t.\ \ \lambda_i ≥ 0
$$

  先求 $ \underset{w,b}{min}\ L(w,b,\lambda)$ :

$$
对b求偏导:\frac{\partial L}{\partial b} = 0 \rightarrow \sum_{i=1}^{N} \lambda_iy_i=0\ (1)\\\\
(1)代入L(w,b,\lambda)=\frac{1}{2}w^Tw + \sum_{i=1}^{N} \lambda_i- \sum_{i=1}^{N}\ \lambda_iy_iw^Tx_i\\\\\
对w求偏导:\frac{\partial L}{\partial w}=0 \rightarrow w=\sum_{i=1}^{N}\lambda_iy_ix_i\ (2)\\\\\
(2)代入L(w,b,\lambda)=\sum_{i=1}^{N}\lambda_i-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_i^Tx_j
$$

* 对偶问题的等价形式:
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\lambda_i)\\\\
s.t. \ \ \sum_{i=1}^{N}\lambda_iy_i=0,\\\\
\lambda_i≥0,(i=1,2……N)
$$



### 2.3  KKT 条件

$$
\left\{
\begin{matrix}
\lambda_i& ≥ 0\\\\
1-y_i(w^T x_i+b)& ≤ 0\\\\
\lambda_i\ [1-y_i(w^Tx_i+b)]& = 0
\end{matrix}
\right.
$$

$$
下面根据KKT条件求:  w^ *, b^ * \\\\
w^ * 上面已经得到 w^ * = \sum_{i=1}^{N}\lambda_iy_ix_i( 可见w^ *是数据data的线性组合)
$$

下面来求 $b^*$:

![松弛互补](https://github.com/wangchuan-hub/Support-Vector-Machine/blob/main/%E6%9D%BE%E5%BC%9B%E4%BA%92%E8%A1%A5.png)

$$
\exists(x_k,y_k) 使得 1-y_k(w^Tx_k+b)=0\\\\
y_k(w^Tx_k+b)=1 两边同乘  \ y_k\\\\
(w^Tx_k+b)=y_k\\\\
b=y_k-w^Tx_k, 将w^* 代入 \\\\
b^*=y_k-\sum_{i=1}^{N}\lambda_iy_ix_i^Tx_k
$$

划分超平面:$w^ *x +b^*=0$

**注:由KKT条件可以看到 $1-y_i(w^Tx_i+b)<0$ 的所有样本其$\lambda_i$都为0，也就是只有在最大间隔边界上的点才会起作用，这些样本点被称为支持向量，不在最大间隔边界上的点训练完成后都不需保留。**



## 3 线性支持向量机与软间隔最大化 

硬间隔假定训练样本在样本空间或特征空间中是线性可分的，即存在一个超平面能将不同类的样本完全分开。但在现实任务中往往很难确定合适的超平面将不同类的样本完全分开；退一步讲，即使可以找到某个超平面使训练集在特征空间上线性可分，也很难断定这个结果不是由于过拟合造成的。

缓解该问题的方法是允许支持向量机在一些样本上出错，为此，引入"软间隔"概念

<img src="C:\Users\15735\Desktop\硕士文件\软间隔.png" alt="软间隔示意图" style="zoom: 67%;" />

线性不可分意味着某些样本点不满足 间隔$y_i(w^Tx_i + b） ≥ 1$，为解决这个问题，引入松弛变量$\xi_i≥0$，使得间隔加上松弛变量大于等于1，则约束条件变为：
$$
y_i(w^Tx_i + b） ≥ 1-\xi_i
$$
再对每一个松弛变量 $\xi_i$ 支付一个代价 $\xi_i$ 目标函数变为：
$$
\frac{1}{2}||w||^2+C\sum_{i=1}^{N}\xi_i
$$

这里C>0为惩罚参数，C值大时对误分类的惩罚增大，最小化目标函数包含两层含义：

* 使$\frac{1}{2}||w||^2$尽量小，即间隔尽量大
* 使误分类的点的个数尽量少，C为调和两者的系数

软间隔的约束优化问题(原问题)：
$$
\underset{w,b}{min}\frac{1}{2}||w||^2+C\sum_{i}^{N}\xi_i \\\\
s.t. \ \ \ y_i(w^Tx_i + b） ≥ 1-\xi_i \\\\
\xi_i≥0\ (i=1,2……N)
$$

拉格朗日函数：
$$
L(w,b,\lambda,\xi,\mu)=\frac{1}{2}||w||^2+C\sum_{i=1}^{N}\xi_i+\\\\
\sum_{i=1}^{N}\lambda_i(1-\xi_i-y_i(w^Tx_i+b))-\sum_{i=1}^{N}\mu_i\xi_i
$$
令$L(w,b,\lambda,\xi,\mu)对w,b,\xi偏导为零$可得：
$$
w=\sum_{i=1}^{N}\lambda_iy_ix_i\\\\
0=\sum_{i=1}^{N}\lambda_iy_i\\\\
C=\lambda_i+\mu_i
$$
对偶问题：
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\lambda_i)\\\\
s.t. \ \ \sum_{i=1}^{N}\lambda_iy_i=0,\\\\
0≤\lambda_i≤C,(i=1,2……N)
$$




## 4非线性支持向量机与核函数

### 4.1 非线性问题与核技巧

<img src="C:\Users\15735\Desktop\硕士文件\屏幕截图 2021-05-04 111626.png" alt="非线性分类问题" style="zoom: 50%;" />

<img src="C:\Users\15735\Desktop\硕士文件\异或问题.png" alt="异或问题" style="zoom:67%;" />

非线性问题不好求解，可以采取非线性变换，将非线性问题转化为线性问题，通过解变换后的线性问题的方法求解原来的非线性问题。



令$\phi(x)表示将x$映射后的向量，则特征空间中划分超平面所对应的模型:
$$
f(x)=w^T\phi(x)+b
$$
特征空间中的对偶问题：
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_j\phi(x_i)^T\phi(x_j)-\sum_{i=1}^{N}\lambda_i)\\\\
s.t.\ \ \sum_{i=1}^{N}\lambda_iy_i=0,\\\\
\lambda_i≥0,(i=1,2……N)
$$
特征空间维数很高甚至是无穷维，直接求解 $\phi(x_i)^T\phi(x_j)$ 很困难，故设想这样一个函数 ：
$$
K(x_i，x_j)=\phi(x_i)^T\phi(x_j)
$$
则特征空间中的对偶问题可重写为：
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jK(x_i,x_j)-\sum_{i=1}^{N}\lambda_i)\\\\
s.t.\ \ \sum_{i=1}^{N}\lambda_iy_i=0,\\\\
\lambda_i≥0,(i=1,2……N)
$$

### 4.2 核函数

设 $\chi$ 是输入空间 ，H 为特征空间，如果存在一个从$\chi$到 H 的映射 $\phi(x):\chi \rightarrow$ H​，使得对所有 $x_i，x_j \in\chi$

函数$K(x_i，x_j)$满足条件：
$$
K(x_i，x_j)=\phi(x_i)\cdot\phi(x_j)
$$


则称$K(x_i,x_j)为核函数，\phi(x)为映射函数，\phi(x_i)\cdot\phi(x_j)为内积$

* 常用核函数：

  

![常用核函数](https://github.com/quantitiveFund/Support-Vector-Machine1/blob/main/%E5%B8%B8%E7%94%A8%E6%A0%B8%E5%87%BD%E6%95%B0.png)



## 5 序列最小优化算法 (SMO)

从以上的推导可以看出，支持向量机的学习问题可以形式化为求解凸二次规划问题，这样的凸二次规划问题有全局最优解。许多算法可以用于这一问题的求解，但是在训练样本容量很大时，这些算法往往变得很低效，这里介绍高效的支持向量机的学习算法：序列最小优化算法 SMO(sequential minimal optimization)

SMO算法要求解的凸二次规划的对偶问题
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jK(x_i,x_j)-\sum_{i=1}^{N}\lambda_i)\\\\s.t.\ \ \sum_{i=1}^{N}\lambda_iy_i=0,\\\\
0≤\lambda_i≤C,(i=1,2……N)
$$
在这个问题中变量是拉格朗日乘子，一个变量 $\lambda_i$ 对应于一个样本点$(x_i,y_i)$,变量总数等于训练样本容量 N。

