# 强化学习基础篇（二十七）Model-free控制

终于推进到控制部分了，控制的问题才是核心。

## 1、预测与控制

预测与控制的区别在于：

* 预测问题中是输入一个MDP $(S,A,P,R,\gamma)$以及一个策略$\pi$，然后输出基于当前策略$\pi$的价值函数$V_{\pi}$。
* 控制问题是MDP $(S,A,P,R,\gamma)$，然后输出最优价值函数$V_*$以及最优策略$\pi_*$。

之前的内容主要讲了MC，TD，$TD(\lambda)$算法，这三个算法都是为了在给定策略下去估计价值函数$V(s)$。其区别在于MC需要得到一个完整的episode才能进行一次价值函数的更新，而TD方法则可以没走一步就更新一次价值函数。但是我们的目标是要得到最优的策略，所以我们需要通过控制问题，在有MDP的情况下，通过价值函数去改进策略。不断得进行迭代改进，最终收敛到最优策略和最优价值函数。

控制的例子很多，比如控制一个机器人行走，让智能体学会玩围棋，开发一个交易管理的智能体等等。这些实际的问题在显示中可能有着两类问题：

* 一是，MDP模型未知，但是我们可以从现实中很容易进行采样收集数据。
* 二是，MDP模型已知，但是问题的规模太大了，完全没办法进行高效的计算，所以必须使用采样的方法。

Model-free控制就是专注于解决这类问题。

## 2、同轨策略与异轨策略（On and Off Policy）

同轨策略学习（On policy learning）就是智能体已经有了一个策略$\pi$，并且基于该策略$\pi$进行采样，以得到的经验轨迹集合来更新价值函数。虽有采用策略评估和策略改进对给定策略进行优化，以获得最优策略。由于需要优化的策略$\pi$基于当前给定的策略$\pi$，所以称之为On Policy。

异轨策略学习（Off policy learning）是智能体虽然有一个策略$\pi$，但是并不基于该策略$\pi$进行采样，而是基于另一个策略$\mu$进行采样。另一个策略$\mu$可以是人类专家制定的策略等一些较为成熟的策略方法。由于优化的策略不完全基于当前策略，所以称为Off Policy。

## 3、同轨蒙特卡洛控制（On-policy Monte Carlo Control）

### 广义策略迭代（GPI）

回顾一下之前提到的广义策略迭代（Generalized Policy Iteration，GPI）模型是指让策略评估和策略改善交互的一般概念，它不依赖于两个过程的粒度（granularity）和其他细节。几乎所有强化学习方法都可以被很好地描述为GPI。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-fa0dac1fcc144570.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果评估过程和改善过程都稳定下来，即不再发生变化，那么价值函数和策略必须都是最优的，如上图（右）所示。

人们还可以用两个目标来考虑GPI中评估和改善过程的相互作用，如上图（左）所示，上面的线代表目标价值函数$V=V^{\pi}$，下面的线代表目标$\pi=greedy(v)$。目标会发生相互作用，因为两条线不是平行的。从一个策略$\pi$和一个价值函数$v$开始，每一次箭头向上代表着利用当前策略进行价值函数的更新，每一次箭头向下代表着根据更新的价值函数贪婪地选择新的策略，说它是贪婪的，是因为每次都采取转移到可能的、状态函数最高的新状态的行为。最终将收敛至最优策略和最优价值函数。

### 基于MC的GPI

在之前的GPI方法中，策略评估用到的是贝尔曼方程，策略改进使用的是贝尔曼最优方程：
$$
\begin{aligned}
v_{*}(s) &=\max _{a} \mathbb{E}\left[R_{t+1}+\gamma v_{*}\left(S_{t+1} \mid S_{t}=s, A_{t}=a\right)\right] \\
&=\max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{*}\left(s^{\prime}\right)\right]
\end{aligned}
$$
但是使用动态规划算法来改善策略是需要知道某一状态的所有后续状态及状态间转移概率，即：
$$
\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}}\left(\mathcal{R}_{s}^{a}+\mathcal{P}_{s s^{\prime}}^{a} V\left(s^{\prime}\right)\right)
$$
但是如果后续的转移概率未知，则在策略估计中就不能使用贝尔曼期望方程，而是变为sample方法，比如MC方法或TD方法。

在模型未知的时候，首先应该用状态行为对的价值$Q(s,a)$来代替状态价值$V(s)$：
$$
\pi'(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}}Q(s,a)
$$
这样做的目的是可以改善策略而不用知道整个模型，只需要知道在某个状态下采取什么样的行为价值最大即可。

所以基于$Q(s,a)$的GPI可以为如下形式。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-84016aa8c85ed805.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

即使这样，至少还存在一个问题，即当我们每次都使用贪婪算法来改善策略的时候，将很有可能由于没有足够的采样经验而导致产生一个并不是最优的策略，我们需要不时的尝试一些新的行为，这就是探索（Exploration）。

### 探索的例子

![image.png](https://upload-images.jianshu.io/upload_images/15463866-ce4c9184c11c2e79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如上图，在你面前有两扇门，考虑如下的行为、奖励并使用贪婪算法改善策略：

* 你打开左侧门得到奖励为0：$V(left)=0$
* 你打开右侧门得到奖励为1：$V(right)=+1$
* 使用greedy策略，会继续去打开右侧门，而不会打开左侧门，假设得到奖励为+3：$V(right)=\frac{1+3}{2}=+2$
* 继续greedy策略，打开右侧门，假设得到奖励+2:$V(right)=\frac{1+3+2}{3}=+2$
* 如此一直循环下去，会一直打开右侧门。

这种情况下，打开右侧门是否就一定是最好的选择呢？

答案显而易见是否定的。因此完全使用贪婪算法改善策略通常不能得到最优策略。为了解决这一问题，我们需要引入一个随机机制，以一定的概率选择当前最好的策略，同时给其它可能的行为一定的几率，这就是$\epsilon-greedy$探索。

### $\epsilon-greedy$探索策略

 $\epsilon-greedy$策略是一个最简单的探索策略，其假设所有$m$个动作都有着非0的概率被执行，在策略选择中以$1-\epsilon$的概率去选择贪婪动作，并以$\epsilon$的概率选择随机动作。其数学表达式为：
$$
\pi(a \mid s)=\left\{\begin{array}{ll}
\epsilon / m+1-\epsilon & \text { if } a^{*}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(s, a) \\
\epsilon / m & \text { otherwise }
\end{array}\right.
$$

如果我们在GPI的策略改进部分使用$\epsilon-greedy$探索策略，那么我们会有理论证明保障改进的策略可以可以单调递增的。

即对于任意的$\epsilon-greedy$策略$\pi$，使用相应的$q_{\pi}$得到的$\epsilon-greedy$策略$\pi'$是在$\pi$上的一次策略提升，即$v_{\pi'}(s) \ge v_{\pi}(s)$。

证明过程如下：
$$
\begin{aligned} q_{\pi}\left(s, \pi^{\prime}(s)\right) &=\sum_{a \in \mathcal{A}} \pi^{\prime}(a \mid s) q_{\pi}(s, a) \\ &=\epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \max _{a \in \mathcal{A}} q_{\pi}(s, a) \\ & \geq \epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a \mid s)-\epsilon / m}{1-\epsilon} q_{\pi}(s, a) \\ &=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_{\pi}(s, a)=v_{\pi}(s) \end{aligned}
$$
所以完整的MC策略迭代过程在引入了$\epsilon-greedy$之后如下所示：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-f414698a4f1ca513.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和之前讲到的策略迭代方法不一样，MC策略迭代在估计中用的是Q函数，在策略改进中用的是$\epsilon-greedy$方法，在实际应用中，我们称之为蒙特卡洛控制，且更确切地给出其迭代示意图：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-88cb2eb841a09cb6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

MC控制使用Ｑ函数进行策略评估，使用$\epsilon-greedy$探索改善策略。该方法最终可以收敛至最优策略。

图中每一个向上或向下的箭头都对应着多个episode。也就是说我们一般在经历了多个episode之后才进行依据Ｑ函数更新或策略改善。

实际上我们也可以在每经历一个episode之后就更新Ｑ函数或改善策略。但不管使用那种方式，在使用$\epsilon-greedy$探索下我们始终只能得到基于某一策略下的近似Ｑ函数，且该算法没有一个终止条件，因为它一直在进行探索。

### GLIE

我们希望得到一个这样的学习方法：

* 1、在学习开始时有足够的探索:
  $$
  \lim _{k \rightarrow \infty} N_{k}(s, a)=\infty
  $$
  
* 2、最终得到的策略没有探索，是一个确定性的策略。
  $$
  \lim _{k \rightarrow \infty} \pi_{k}(a \mid s)=\mathbf{1}\left(a=\underset{a^{\prime} \in \mathcal{A}}{\operatorname{argmax}} Q_{k}\left(s, a^{\prime}\right)\right)
  $$
  

为此引入了另一个理论概念：GLIE(Greedy in the Limit with infinite Exploration), 如果$\epsilon-greedy$策略能够使得$\epsilon$在$\epsilon_k=\frac{1}{k}$时候降低到0，那么$\epsilon-greedy$策略也是一个GLIE策略。其算法流程如下所示：

* 1. 首先从环境中使用策略$\pi$采样k个episode: $S_1,A_1,R_2,...,S_T \sim \pi$

* 2. 对于在episode中的每个状态$S_t$以及动作$A_t$，进行如下增量更新：
     $$
     N\left(S_{t}, A_{t}\right) \leftarrow N\left(S_{t}, A_{t}\right)+1 \\
     Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\frac{1}{N\left(S_{t}, A_{t}\right)}\left(G_{t}-Q\left(S_{t}, A_{t}\right)\right)
     $$

* 3. 基于新得到的Q函数更新策略：
     $$
     \epsilon \leftarrow 1/k \\
     \pi \leftarrow \epsilon-greedy(Q)
     $$

在理论上GLIE MC控制的方法是可以收敛到最优动作值函数，$Q(s,a) \rightarrow q_*(s,a)$。

## 4、同轨时序差分（On-policy TD）学习

### MC和TD控制的差别

时序差分（Temporal-difference,TD）学习方法相比于MC的方法有着几个优势：

* 低方差（low variance）
* 可以在线实学习
* 可以学习不完整的episode

因此可以很自然的想到在控制的迭代中去使用TD方法代替MC方法。也就是下面降到的Sarsa算法。

### Sarsa算法

![image.png](https://upload-images.jianshu.io/upload_images/15463866-03c22c0c1dfb349a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Sarsa算法的名字就是来源于上图这个过程，在智能体处在某个状态$S$的时候按$\epsilon-greedy$执行动作$A$，会得到一个即时奖励$R$，并在与环境交互中转移到下一个状态$S'$，再一次基于$\epsilon-greedy$策略选择动作$A'$。

这个时候智能体不会去执行$A'$，而是通过自身当前的状态行为价值函数得到该$(S',A')$状态行为对的价值$Q(S',A')$，同时结合$(S,A)$获得的奖励$R$来更新$Q(S,A)$。所以通过公式很容易可以看出SARSA的更新规则：
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)
$$
同轨策略的Sarsa算法描述如下：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-66f08f6275ebfe5d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/15463866-32780ab100156968.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Sarsa的收敛性

Sarsa的收敛性是有定理支持的，在满足如下两个条件时，Sarsa算法将收敛至最优行为价值函数。 

* 条件1：任何时候的策略$\pi(a |s)$符合GLIE特性；
* 条件二：步长系数$\alpha_t$满足：$\sum_{t=1}^{\infty} \alpha_{t}=\infty$以及$\sum_{t=1}^{\infty} \alpha_{t}^{2}<\infty$

### n-Step Sarsa

考虑从$n=1,2, ...,\infty$得到的n-step回报，
$$
\begin{array}{rl}n=1 & (\text { Sarsa }) \quad q_{t}^{(1)}=R_{t+1}+\gamma Q\left(S_{t+1}\right) \\ n=2 & q_{t}^{(2)}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} Q\left(S_{t+2}\right) \\ \vdots & \vdots \\ n=\infty & (M C) \quad q_{t}^{(\infty)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T}\end{array}
$$
我们可以得到n-step的Q-return：
$$
q_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} Q\left(S_{t+n}\right)
$$
所以接下来可以根据n-step的Q-return去更新Q函数：
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(q_t^{(n)}-Q(S, A)\right)
$$

### 前向观点的Sarsa($\lambda$)

![image.png](https://upload-images.jianshu.io/upload_images/15463866-1acfa572062ae79e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$TD(\lambda)$是考虑了所有的n-step回报，同样我们对$q^{\lambda}$，我们也可以考虑所有的n-step的Q-return $q_t^{(n)}$。
$$
q_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q_{t}^{(n)}
$$
接下来可以根据 $q_t^{(\lambda)}$去更新Q函数：
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(q_t^{(\lambda)}-Q(S, A)\right)
$$


### 后向观点的Sarsa($\lambda$)

 和$TD(\lambda)$一样，我们在算法中会使用到资格迹（Eligibility trace），但是Sarsa($\lambda$)算法是对环境中的的state-action对维护了一个资格迹。
$$
E_{0}(s, a)=0 \\
E_{t}(s, a)=\gamma \lambda E_{t-1}(s, a)+\mathbf{1}\left(S_{t}=s, A_{t}=a\right)
$$


它体现的是一个结果与某一个状态行为对的因果关系，与得到结果最近的状态行为对，以及那些在此之前频繁发生的状态行为对对得到这个结果的影响最大。

在引入资格迹之后，Q函数的更新规则可以进行如下更新：
$$
\delta_{t}=R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right) \\
Q(s, a) \leftarrow Q(s, a)+\alpha \delta_{t} E_{t}(s, a)
$$

### Sarsa($\lambda$)算法

除了状态价值函数$Q(s,a)$的更新方式、超参数$\lambda$，以及资格迹$E(s,a)$以为，Sarsa($\lambda$)算法的思想和Sarsa是类似的，这里总结下算法流程：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-4992671596fe716b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/15463866-4ef6a91aa649cb6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 5、异轨策略学习（Off-policy Learning）

异轨策略学习的目标是通过计算$v_{\pi}(s)$或者$q_{\pi}(s,a)$去评估一个目标策略$\pi(a|s)$，但是会遵循另外一个行为策略$\mu(s,a)$来进行。其采样可以是：$\left\{S_{1}, A_{1}, R_{2}, \ldots, S_{T}\right\} \sim \mu$

异轨策略这种方式之所以有效，主要有几个考虑的因素：

* 智能体可以不从自身的行为学习，而是从观测人类专家的行为或者观察其他智能体的行为中进行学习。
* 我们可以服用在运行中那些以前产生的旧策略进行学习，比如在更新过程中产生的各种策略 $\pi_1,\pi_2...\pi_{t-1}$。
* 智能体可以在遵循一些探索策略的时候去学习最优策略。
* 智能体可以在只遵循一个策略的时候去学习多种策略。

### 重要性采样

这里要先介绍下载Off-policy中重要的概念重要性采样（Importance Sampling），重要性采样就是我们要计算函数$f(X)$在分布$P$下的期望时候，不好计算。那么我们可以转换下思路，去转化到计算函数在一个比较容易计算的分布下$Q$下的期望：
$$
\begin{aligned} \mathbb{E}_{X \sim P}[f(X)] &=\sum P(X) f(X) \\ &=\sum Q(X) \frac{P(X)}{Q(X)} f(X) \\ &=\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right] \end{aligned}
$$
考虑$t$时刻之后的动作状态轨迹 $\rho_{t}=A_{t}, S_{t+1}, A_{t+1}, \cdots, S_{T}$，可以得到该轨迹出现的概率为：
$$
\mathbb{P}\left(\rho_{t}\right)=\prod_{k=t}^{T-1} \pi\left(A_{k} \mid S_{k}\right) \mathbb{P}\left(S_{k+1} \mid S_{k}, A_{k}\right)
$$
因此可以得到相应的重要性权重为：
$$
\eta_{t}^{T}=\frac{\prod_{k=t}^{T-1} \pi\left(A_{k} \mid S_{k}\right) \mathbb{P}\left(S_{k+1} \mid S_{k}, A_{k}\right)}{\prod_{k=t}^{T-1} \mu\left(A_{k} \mid S_{k}\right) \mathbb{P}\left(S_{k+1} \mid S_{k}, A_{k}\right)}=\prod_{k=t}^{T-1} \frac{\pi\left(A_{k} \mid S_{k}\right)}{\mu\left(A_{k} \mid S_{k}\right)}
$$
即便是未知环境模型，也能得到重要性权重。

### IS下的异轨MC

对于off-policy Monte-Carlo使用importance sampling：

* a、使用从MC的行为策略$\mu$中得到的回报去评估策略$\pi$,
* b、得到的加权回报为：

$$
G_{t}^{\pi / \mu}=\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)} \frac{\pi\left(A_{t+1} \mid S_{t+1}\right)}{\mu\left(A_{t+1} \mid S_{t+1}\right)} \cdots \frac{\pi\left(A_{T} \mid S_{T}\right)}{\mu\left(A_{T} \mid S_{T}\right)} G_{t}
$$

* c、根据加权回报更新值函数：
  $$
  V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\pi / \mu}-V\left(S_{t}\right)\right)
  $$

这里要注意的是如果行为策略$\mu$的概率为0，但是目标策略$\pi$的概率非0，就不能用了。

同时我们知道，MC方法的方差本来就很大，而重要性采样将会使得方差急剧增大，因此结合重要性采样的MC方法更不适用。

### IS下的异轨TD

对于off-policy TD使用importance sampling，使用从TD方法在遵循行为策略$\mu$的同时去去评估策略$\pi$，其更新方式为：
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)}\left(R_{t+1}+\gamma V\left(S_{t+1}\right)\right)-V\left(S_{t}\right)\right)
$$
采用TD的方式比MC的方式大大降低了方差。

### $Q-Learning$算法

$Q-Learning$算法是不需要使用重要性采样的，其过程是这样的：

* 在$t$时刻与环境进行实际交互的行为$A_t$由一个$\epsilon-greedy$策略$\mu$生成：$A_{t} \sim \mu\left(\cdot \mid S_{t}\right)$
* 在$t+1$时刻用来更新$Q$值的行为$A_{t+1}'$，通过一个完全greedy的策略$\pi$产生：$A_{t+1}' \sim \pi\left(\cdot \mid S_{t+1}\right)$

其动作状态值函数的更新方式为：
$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
$$
其中的TD target是：$R_{t+1}+\gamma Q(S_{t+1}, A^{\prime})$，$A’ \sim \pi\left(\cdot \mid S_{t}\right)$

这里和之前的TD target还是有区别的：$R_{t+1}+\gamma Q(S_{t+1}, A_{t+1})$，$A_{t+1} \sim \mu\left(\cdot \mid S_{t}\right)$

### $Q-Learning$进行异轨控制

我们已经知道$Q-Learning$中目标策略$\pi$是一个关于$Q(s,a)$的贪婪策略：
$$
\pi\left(S_{t+1}\right)=\underset{a^{\prime}}{\operatorname{argmax}} Q\left(S_{t+1}, a^{\prime}\right)
$$
行为策略$\mu$是一个关于$Q(s,a)$的$\epsilon-greedy$策略，所以$Q-Learning$的目标可以简化为：
$$
\begin{aligned} & R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right) \\=& R_{t+1}+\gamma Q\left(S_{t+1}, \underset{a^{\prime}}{\operatorname{argmax}} Q\left(S_{t+1}, a^{\prime}\right)\right) \\=& R_{t+1}+\max _{a^{\prime}} \gamma Q\left(S_{t+1}, a^{\prime}\right) \end{aligned}
$$
其Q函数的更新方式为：
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma \max _{a^{\prime}} Q\left(S^{\prime}, a^{\prime}\right)-Q(S, A)\right)
$$
算法描述为：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-46665ffb8a59c563.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/15463866-d6f2be70b7fde830.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 6、总结

下面两张图概括了各种DP算法和各种TD算法，同时也揭示了各种不同算法之间的区别和联系。总的来说TD是采样+有数据引导（bootstrap），DP是全宽度+实际数据。如果从Bellman期望方程角度看：聚焦于状态本身价值的是迭代法策略评估（DP）和TD学习，聚焦于状态行为对价值函数的则是Q-策略迭代（DP）和SARSA；如果从针对状态行为价值函数的Bellman优化方程角度看，则是Q-价值迭代（DP）和Q学习。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-d0c65c96e43aa6c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/15463866-5adf397c8891d3fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)