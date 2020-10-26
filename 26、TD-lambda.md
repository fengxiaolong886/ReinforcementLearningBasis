# 强化学习基础篇（二十六）$TD({\lambda})$预测

## 1、平均n-Step回报

从在上一篇中我们考虑了n-Step回报，在每个n的选择都有着相应的回报（Reward）。我们如果把不同的n-step回报都做一个如下的平均，例如对2-step和4-step回报可以这样：
$$
\frac{1}{2} G^{(2)}+\frac{1}{2} G^{(4)}
$$
这样我们得到的回报信息就结合了2个不同的时间步的结果。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-87b10105db2c5a7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 2、$\lambda-return$

如果我们考虑结合所有的n-step的回报$G_t^{(n)}$，就可以得到$\lambda-return$，即$G_t^{\lambda}$。其定义为：
$$
G_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t}^{(n)}
$$
这里是把$TD(\lambda)$算法视为平均n步更新的一种特例。这里的平均值包括了所有可能的n步更新，每一个都按照比例$\lambda^{n-1}$进行加权，其中$\lambda \in [0,1]$，最后乘上正则项$1-\lambda$保证权值和为1。可视化如下：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-744803262982f2a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$\lambda-return$中每一个n步回报的权重如下所示：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-e2a177f7d7bdaf9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中单步回报获得了最大的权值$(1-\lambda)$，两步回报为$(1-\lambda)\lambda$，三步回报为$(1-\lambda)\lambda^2$，以此类推进行衰减。

所以前向的$TD(\lambda)$算法使用$G_t^{\lambda}$替换target，得到价值函数的更新为：
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\lambda}-V\left(S_{t}\right)\right)
$$

## 3、$TD(\lambda)$的前向视图

之前我们介绍的所有算法，理论上都是前向的(Forward-view)。对于访问的每一个状态，我们向前（未来的方向）探索所有可能的收益并决定如何将他们提供的信息进行有效结合利用。如下所示：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-cc4d5db7ce030e54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以想象处于一个状态流之中，从每一个状态向前看并决定如何更新这个状态。每次更新完一个状态，移动到下一个状态并且不再更新以前经过的路径的状态。在另一个方面，未来的状态会从之前的位置被重复地观测并被处理。

引入了$\lambda$之后，会发现要更新一个状态的状态价值$V(S_t)$，必须要走完整个episode获得每一个状态的即时奖励以及最终状态获得的即时奖励。这和MC算法的要求一样，因此$TD(\lambda)$算法有着和MC方法一样的劣势。

* TD更新为：
  $$
  \begin{aligned}
  &V \left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\lambda}-V\left(S_{t}\right)\right)\\
  &G_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t}^{(n)}\\
  &G_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} V\left(S_{t+n}\right)
  \end{aligned}
  $$

* MC更新为：
  $$
  V \left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}-V\left(S_{t}\right)\right)
  $$



## 4、资格迹（Eligibility Traces）

考虑如下一个问题，老鼠在连续接受了3次响铃和1次亮灯信号后遭到了电击，那么在分析遭电击的原因时，到底是响铃的因素较重要还是亮灯的因素更重要呢？

![image.png](https://upload-images.jianshu.io/upload_images/15463866-0f643c09a5a2171a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

问题的归因可以考虑两种情况：

* 频率启发（Frequency heuristic）：将原因归因于出现频率最高的状态，所以老鼠被点击的主要原因会是铃铛。
* 就近启发 （Recency heuristic）：将原因归因于较近的几次状态，这种考虑之下，老鼠被点击的主要原因会是灯。

结合频率启发（Frequency heuristic）与就近启发 （Recency heuristic）两种思想，可以引出资格迹（Eligibility Traces）。

定义如下：
$$
\begin{array}{l}
E_{0}(s)=0 \\
E_{t}(s)=\gamma \lambda E_{t-1}(s)+\mathbf{1}\left(S_{t}=s\right)
\end{array}
$$
其中$\mathbf{1}\left(S_{t}=s\right)$是一个条件判断，可以改写$E_t(s)$为：
$$
E_{t}(s)=\left\{\begin{array}{ll}
\gamma \lambda E_{t-1} & \text { if } S_{t} \neq s \\
\gamma \lambda E_{t-1}+1 & \text { if } S_{t}=s
\end{array}\right.
$$
直观上如下图：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-a4e6f6ecaf7116ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

该图横坐标是时间，横坐标下有竖线的位置代表当前进入了状态$s$，纵坐标是资格迹$E$，可以看出当某一状态连续出现， $E$值会在一定衰减的基础上有一个单位数值的提高，此时将增加该状态对于最终收获贡献的比重，因而在更新该状态价值的时候可以较多地考虑最终收获的影响。同时如果该状态距离最终状态较远，则其对最终收获的贡献越小，在更新该状态时也不需要太多的考虑最终收获。

资格迹的提出是基于一个**信度分配（Credit Assignment）**问题的，打个比方，最后我们去跟别人下围棋，最后输了，那到底该中间我们下的哪一步负责？或者说，每一步对于最后我们输掉比赛这个结果，分别承担多少责任？这就是一个信度分配问题。对于小鼠问题，小鼠先听到三次铃声，然后看见灯亮，接着就被电击了，小鼠很生气，它仔细想，究竟是铃声导致的它被电击，还是灯亮导致的呢？如果按照事件的发生频率来看，是铃声导致的，如果按照最近发生原则来看，那就是灯亮导致的，但是，更合理的想法是，这二者共同导致小鼠被电击了，于是小鼠为这两个事件分别分配了权重，如果某个事件$s$发生，那么$s$对应的资格迹的值就加1，如果在某一段时间$s$未发生，则按照某个衰减因子进行衰减，这也就是上面的资格迹的计算公式了。

资格迹$E$值并不需要等到完整的episode结束才能计算出来，它可以每经过一个时刻就得到更新。 

## 5、$TD(\lambda)$的后向视图

后向视角使用了我们刚刚定义的资格迹，每个状态$s$都保存了一个资格迹。我们可以将资格迹理解为一个权重，状态$s$被访问的时间离现在越久远，其对于值函数的影响就越小，状态$s$被访问的次数越少，其对于值函数的影响也越小。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-ca229c19466c8d9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$TD(\lambda)$的后向视角解释：有个人坐在状态流上，手里拿着话筒，面朝着已经经历过的状态获得当前回报并利用下一个状态的值函数得到TD偏差之后，此人会向已经经历过的状态喊话告诉这些已经经历过的状态处的值函数需要利用当前时刻的TD偏差进行更新。此时过往的每个状态值函数更新的大小应该跟距离当前状态的步数有关。

假设当前状态为$s$，TD偏差为$\delta_t$，那么$s_{t-1}$处的值函数更新应该乘以一个衰减因子$\gamma\lambda$，状态 $s_{t-2}$处的值函数更新应该乘以 $(\gamma\lambda)^2$，以此类推。状态价值的更新为：
$$
\begin{array}{l}
\delta_{t}=R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right) \\
V(s) \leftarrow V(s)+\alpha \delta_{t} E_{t}(s) \\
\text { 其中 } E_{t}(s)=\left\{\begin{array}{ll}
\gamma \lambda E_{t-1} & \text { if } S_{t} \neq s \\
\gamma \lambda E_{t-1}+1 & \text { if } S_{t}=s
\end{array}, \quad E_{0}(s)=0\right.
\end{array}
$$

## 6、$TD(\lambda)$的前向视图与后向视图的关系

### $TD(\lambda)$与$TD（0)$

当$\lambda =0$的时候，$\gamma\lambda=0$。只有当前状态会得到更新，资格迹只会记录脉冲信号。其等价于$TD(0)$算法：
$$
\begin{array}{l}
E_{t}(s)=\mathbf{1}\left(S_{t}=s\right) \\
V(s) \leftarrow V(s)+\alpha \delta_{t} E_{t}(s) \quad \left(其中，\delta_{t}=R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right) \\
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha \delta_{t}\\
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha (R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right))
\end{array}
$$

### $TD(1)$与$MC$

当$\lambda =1$的时候，信度分配只会在episode结束的时候才会被定义，$TD(\lambda)$与$MC$将会等价。

理论上，对于状态$s$的总更新量前向视角和后向视角是等价的，如下等式的左边为后向视角的总更新量，等式右边为前向视角的总更新量。
$$
\sum_{t=1}^{T} \alpha \delta_{t} E_{t}(s)=\sum_{t=1}^{T} \alpha\left(G_{t}^{\lambda}-V\left(S_{t}\right)\right) \mathbf{1}\left(S_{t}=s\right)
$$
我们假设处在某个episode中，状态$s$在$k$时刻被访问了一次，那么$TD(1)$的资格迹会随时间进行衰减（在$k$时刻之前，资格迹E为0，自$k$时刻开始衰减）：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-ff050819dad214c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$TD(1)$的在线更新过程中，他的累积误差可以表述为：
$$
\sum_{t=1}^{T-1} \alpha \delta_{t} E_{t}(s)=\alpha \sum_{t=k}^{T-1} \gamma^{t-k} \delta_{t}=\alpha\left(G_{k}-V\left(S_{k}\right)\right)
$$
在episode介绍的时候总的累积误差是：
$$
\delta_{k}+\gamma \delta_{k+1}+\gamma^{2} \delta_{k+2}+\ldots+\gamma^{T-1-k} \delta_{T-1}
$$
当$\lambda=1$的时候，总的累积误差与MC误差的关系可以进行如下关联：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-85504add2bd6ef6a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上式推导过程中，只是简单展开后删除中间项，这样的结果中$G_t$相当于MC方法的总的回报，$V(S_t)$为当前的状态值函数，$G_{t}-V\left(S_{t}\right)$相当于蒙特卡洛的更新量。所以当$\lambda=1$时，TD总的累积误差会缩小为MC误差。

所以简单总结下：

* $TD(1)$和每次访问的蒙特卡洛方法是大致是等价的，不过也有区别。
* $TD(1)$是在线对误差进行累积，每步都会更新。
* $TD(1)$如果也等到episode结束后离线更新，那么$TD(1)$和MC就完全等价。

### $TD(\lambda)$的$\lambda-error$

对于一般的$\lambda$，不是极端的0与1之外的情况，我们也可以证明总误差等价于$G_{t}^{\lambda}-V\left(S_{t}\right)$。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-e62d41eafdfa3af0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 前向视角和后向视角的$TD(\lambda)$

假设在某个episode中，状态$s$在$k$时刻被访问了一次，那么$TD(\lambda)$的资格迹会随时间进行衰减（在$k$时刻之前，资格迹E为0，自$k$时刻开始衰减）

![image.png](https://upload-images.jianshu.io/upload_images/15463866-470451d5891f0c0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

后向视角的$TD(\lambda)$在这个过程中不断累积误差：
$$
\sum_{t=1}^{T} \alpha \delta_{t} E_{t}(s)=\alpha \sum_{t=k}^{T}(\gamma \lambda)^{t-k} \delta_{t}=\alpha\left(G_{k}^{\lambda}-V\left(S_{k}\right)\right)
$$
当整个片段完成时，后向视角方法对于值函数$V(s)$ 的增量等于$\lambda-return$；如果状态$s$被访问了多次，那么资格迹就会累积，从而相当于累积了更多的$V(s)$ 的增量。这直观地解释了前向视角和后向视角的等价性。

### 总结

![image.png](https://upload-images.jianshu.io/upload_images/15463866-897833b4f013dbf0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)