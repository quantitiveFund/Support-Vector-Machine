<div align='center'><font size ='70'>  Support Vector Machine </font></div>

## 1 hard-margin SVM
### 1.1基本概念：
支持向量机（support vector machines, SVM）是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；SVM还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。
### 1.2 算法原理：

SVM学习的基本想法是基于训练集D在样本空间中找到一个能够将训练样本分开，并且几何间隔最大的划分超平面。对于线性可分的数据集来说，这样的超平面有无穷多个（即感知机），但是几何间隔最大的分离超平面却是唯一   的。这样的超平面对训练样本局部扰动的"容忍"最好，产生的分类结果是最鲁棒的(Robust)，泛化能力最强。

![SVM划分超平面](C:\Users\15735\Desktop\硕士文件\svm划分超平面 2021-04-19 181503.png)

### 1.3 模型构建：

在样本空间中，划分超平面由 $w^Tx + b = 0$表示，其中 w 为法向量,决定超平面的方向，b 为位移项，决定超平面与原点的距离。
* 第一步
$$
\underset{w,b}{max}\ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i+b|\\\\
s.t.\ w^Tx_i + b > 0,y_i = +1\\\\
w^Tx_i + b < 0,y_i = -1，i=1,2...N
$$
其中 $ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i + b|$ 表示训练样本中与超平面最近的点 到 超平面的距离； 
$ \underset {w,b}{max}\ \underset{x_i}{min} \frac{1}{||w||}|w^Tx_i + b| $ 则表示目标函数为使这个距离 也就是 "间隔" 最大；约束条件表示样本能够正确分类



* 第二步：
$$
\underset{w,b}{max}\frac{1}{||w||} \underset{x_i}{min}\ y_i(w^Tx_i + b) \\\\
s.t.\ \ y_i(w^Tx_i + b) > 0 \ \ \ i = 1,2...N
$$

存在 r > 0 使得 $ \underset{x_i}{min}\ y_i(w^Tx_i + b)=r $ 并可以将 r 缩放为1 则 $\underset{w,b}{max}\frac{1}{||w||} \underset{x_i}{min}\ y_i(w^Tx_i + b)$ 可写为 $ \underset{w,b}{max} \frac{1}{||w||}$

约束条件变为  $y_i(w^Tx_i + b) ≥ 1$, 得到最终的约束优化问题(原问题):
$$
\underset{w,b}{min}\frac{1}{2}w^Tw \\
s.t. \ \  y_i(w^Tx_i + b） ≥ 1 \ \ i =1, 2……N
$$

### 1.4  原问题与对偶问题

* 由原问题构建拉格朗日函数:


$$
L(w,b,\lambda) = \frac{1}{2}w^Tw + \sum_{i=1}^{i=N}\lambda_i[1-y_i(w^Tx_i + b)]\\ 其中\ \lambda_i≥0，1-y_i(w^Tx_i + b)≤0
$$

* 将带约束形式的原问题写为无约束形式:
$$
\underset{w,b}{min}\ \underset{\lambda}{max} \ L(w,b,\lambda) \\s.t. \ \ \lambda_i ≥ 0
$$

* 转化为对偶问题：
$$
\underset{\lambda}{max}\ \underset{w,b}{min}\ L(w,b,\lambda) \\s.t.\ \ \lambda_i ≥ 0
$$

  先求 $ \underset{w,b}{min}\ L(w,b,\lambda)$ :

$$
对b求偏导:\frac{\partial L}{\partial b} = 0 \rightarrow \sum_{i=1}^{N} \lambda_iy_i=0\ (1)\\\\
(1)代入L(w,b,\lambda)=\frac{1}{2}w^Tw + \sum_{i=1}^{N} \lambda_i- \sum_{i=1}^{N}\ \lambda_iy_iw^Tx_i\\\\\
对w求偏导:\frac{\partial L}{\partial w}=0 \rightarrow w=\sum_{i=1}^{N}\lambda_iy_ix_i\ (2)\\\\\
(2)代入L(w,b,\lambda)=\sum_{i=1}^{N}\lambda_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_i^Tx_j
$$

* 对偶问题的等价形式:
$$
\underset{\lambda}{min}\ \ \ (\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\lambda_i\lambda_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\lambda_i)
$$



### 1.5  KKT 条件

$$
\left\{\begin{aligned}\lambda_i≥0\\
1-y_i(w^Tx_i+b)≤0\\
\lambda_i[1-y_i(w^Tx_i+b)]=0\end{aligned}\right.
$$

根据KKT条件求 $w^*,b^*$:其中$w^*$ 上面已得到  $w^*=\sum_{i=1}^{N}\lambda_iy_ix_i$ ; 
可见$ w^*$ 是数据data的线性组合

下面来求 $b^*$:

<img src="C:\Users\15735\Desktop\硕士文件\松弛互补.png" alt="松弛互补" style="zoom: 67%;" />
$$
\exists(x_k,y_k)使得1-y_k(w^Tx_k+b)=0\\
y_k(w^Tx_k+b)=1两边同乘\ y_k\\
(w^Tx_k+b)=y_k\\
b=y_k-w^Tx_k,将w^*代入\\
b^*=y_k-\sum_{i=1}^{N}\lambda_iy_ix_i^Tx_k
$$
划分超平面：  $w^*x+b^*$

**注:由KKT条件可以看到 $1-y_i(w^Tx_i+b)<0$ 的所有样本其$\lambda_i$都为0，也就是只有在最大间隔边界上的点才会起作用，这些样本点被称为支持向量，不在最大间隔边界上的点训练完成后都不需保留。**

## 2 soft-margin SVM 

硬间隔假定训练样本在样本空间或特征空间中是线性可分的，即存在一个超平面能将不同类的样本完全分开。但在现实任务中往往很难确定合适的超平面将不同类的样本完全分开；退一步讲，即使可以找到某个超平面使训练集在特征空间上线性可分，也很难断定这个结果不是由于过拟合造成的。

缓解该问题的方法是允许支持向量机在一些样本上出错，为此，引入"软间隔"概念
