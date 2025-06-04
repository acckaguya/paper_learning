# 1 Normalization

Normalization：规范化或便准化，把输入数据进行伸缩和变化，将X的分布规范化地固定在区间范围的标准分布。

**基本框架：**

$$h = f\left(\mathbf{g} \cdot \frac{\mathbf{x}-\mu}{\sigma} + \mathbf{b}\right)$$

- h 是输出
- f 是激活函数
- g 是权重向量
- x 是输入数据
- μ (mu) 是均值
- σ (sigma) 是标准差（标准化参数）
- b 是偏置项

Normalization把数据拉回正态分布，保证了网络的稳定性。

Normalizaiton根据标准化操作的维度分为Batch Normalization和Layer Normalization。前者通过batch size这个维度来进行归一化，后者通过hidden size这个维度。

每一层网络单独看成一个分类器，对上一层的输出数据进行分类，每一层的数据分布不一样，会导致Internal Covariate Shift（内部协变量偏移）。随着网络层数的增大，误差会不断积累，最终导致效果欠佳。此时只对数据进行预处理只能解决第一层的问题，之后需要Normalization等方法来解决。

# Batch Normalization

针对单个神经元进行，利用网络训练时的一个mini-batch的数据来计算该神经元的均值和方差。

归一化操作，加入小常数$\epsilon$防止除0：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

可学习参数变换：
$$y = \gamma \cdot \hat{x} + \beta$$

训练与推理的差异：

![[Pasted image 20250526152439.png]]

移动平均更新公式：
$$\mu_{moving} = \rho \cdot \mu_{moving} + (1-\rho) \cdot \mu_B$$ $$\sigma_{moving}^2 = \rho \cdot \sigma_{moving}^2 + (1-\rho) \cdot \sigma_B^2$$
![[Pasted image 20250526154126.png]]


# Layer Normalization

综合考虑一层所有维度的输入，计算该层平均输入值和输入方差，然后用同一个规范化操作来转换各个维度的输入。

统计量计算：
$$\mu = \frac{1}{d}\sum_{i=1}^d x_i$$ $$\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 + \epsilon}$$

通过学习缩放因子g和b来恢复表达能力。




# BN和LN对比

Batch Normalization (BN)：  
  
归一化维度：对同一批数据中相同特征维进行归一化。即对所有样本在同一特征维上计算均值和方差，然后对该特征维的所有样本值做归一化。  
  
Layer Normalization (LN)：  
  
归一化维度：对每个样本的特征维进行归一化，即对一个样本的所有特征一起计算均值和方差，将该单样本内部的特征进行标准化。不会跨样本求统计量，每个样本独立计算自己的均值和方差。




| 特性      | Layer Norm      | Batch Norm   |
| ------- | --------------- | ------------ |
| 统计量计算维度 | 特征维度（单样本）       | 批量维度（同特征跨样本） |
| 适用场景    | RNN/Transformer | CNN/全连接网络    |
| 小批量数据表现 | 稳定              | 性能下降         |
| 推理阶段处理  | 无需存储统计量         | 依赖训练统计量      |