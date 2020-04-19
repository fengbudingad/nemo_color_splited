# 基于贝叶斯决策的nemo鱼颜色分割

## 框架

#### 整个颜色分割模型框架如下：

![](F:\Desktop\框架.jpg)



#### 整体上分为参数估计和非参数估计两类：

##### （1）非参数估计：

对于一维的灰度图情况，使用了直方图法，kn近邻法和parzen窗法。其中kn近邻法分别设置了kn=2,10,100的情况，parzen窗法使用了高斯核和立方体核。（因为对于一维情况立方体核和超球核基本等价，所以只使用了其中一种）

##### （2）参数估计：

假设类条件概率密度服从正态分布，根据正态分布下贝叶斯决策的知识计算出决策面方程。



#### 注：nemo_figures文件夹内为以上各种情况下的颜色分割结果图以及一维条件下绘制的类条件概率密度函数图形