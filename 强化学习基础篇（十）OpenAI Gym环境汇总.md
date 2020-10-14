# 强化学习基础篇（十）OpenAI Gym环境汇总

 *Gym*中从简单到复杂，包含了许多经典的仿真环境和各种数据，主要包含了经典控制、算法、2D机器人，3D机器人，文字游戏，Atari视频游戏等等。接下来我们会简单看看主要的常用的环境。在Gym注册表中有着大量的其他环境，就没办法介绍了。

## 1、经典控制环境（Classic control）

经典的强化学习示例，方便入门，包括了：

* Acrobot-v1

  Acrobot机器人系统包括两个关节和两个连杆，其中两个连杆之间的关节可以被致动。 最初，连杆是向下悬挂的，目标是将下部连杆的末端摆动到给定的高度。

  ![Acrobot-v1.gif](https://upload-images.jianshu.io/upload_images/15463866-02a0f2c68bf2f329.gif?imageMogr2/auto-orient/strip)

* CartPole-v1

  CartPole-v1环境中，手推车上面有一个杆，手推车沿着无摩擦的轨道移动。 通过对推车施加+1或-1的力来控制系统。 钟摆最开始为直立状态，训练的目的是防止其跌落。 杆保持直立的每个时间步长都提供+1的奖励。 当杆与垂直线的夹角超过15度时，或者推车从中心移出2.4个单位以上时，训练结束。

  ![cartpole.gif](https://upload-images.jianshu.io/upload_images/15463866-96e5868037544267.gif?imageMogr2/auto-orient/strip)

* MountaiCar-v0

  MountaiCar-v0环境中汽车位于一维轨道上，轨道位于两个“山”之间。 目标是让汽车驶向右侧的山峰； 由于汽车的引擎强度不足以直接到达右侧山峰，所以，成功的唯一方法是来回驱动以产生动力。

  ![MountainCar-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-064e7a86517ee361.gif?imageMogr2/auto-orient/strip)

* MountainCarContinuous-v0

  MountainCarContinuous-v0环境与MountaiCar-v0环类似，动力不足的汽车必须爬上一维小山才能到达目标。 与MountainCar-v0不同，动作（应用的引擎力）允许是连续值。目标位于汽车右侧的山顶上。 如果汽车到达或超出，则剧集终止。

  ![MountainCarContinuous-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-646d8d59e8975ab8.gif?imageMogr2/auto-orient/strip)

* Pendulum-v0

  倒立摆摆动问题是控制文献中的经典问题。 在此问题的版本中，摆锤开始于随机位置，目标是将其摆动以使其保持直立。
  
  ![Pendulum-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-0885fc86ff916945.gif?imageMogr2/auto-orient/strip)

## 2 算法学习环境(Algorithms)

从例子中学习强化学习的相关算法，在*Gym*的仿真算法中，由易到难方便新手入坑，包括了：

* Copy-v0

  此任务就是让智能体学习对序列的拷贝操作。

  ![Copy-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-8456c2faf950af4e.gif?imageMogr2/auto-orient/strip)

* DuplicatedInput-v0

  此任务让智能体学习从输入序列$[x_1x_2...x_k]$，输出对每个元素的三次复制$[x_1x_1x_1x_2x_2x_2...x_kx_kx_k]$
  
  ![DuplicatedInput-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-336ceafa771b2711.gif?imageMogr2/auto-orient/strip)
  
* RepeatCopy-v0

  此任务让智能体学习对输入序列$[mx_1x_2...x_k]$，输出对序列的m次复制$[x_1x_2...x_kx_1x_2...x_kx_1x_2...x_k]$。

  ![RepeatCopy-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-4595ac3aa8eadae0.gif?imageMogr2/auto-orient/strip)

* Reverse-v0

  目的是反转输入序列

  ![Reverse-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-885a8d7350c447cd.gif?imageMogr2/auto-orient/strip)

* ReversedAddition-v0

  此任务让智能体学习在提供的网格数字序列之上增加数字序列。在这个任务中需要智能体学会（i）记住加法表； （ii）学习如何在输入网格上移动，以及（iii）发现进位的概念。

  ![ReversedAddition-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-703d9905cb6d4733.gif?imageMogr2/auto-orient/strip)

* ReversedAddition3-v0

  ReversedAddition3-v0与ReversedAddition-v0基本任务相同，但是现在要添加三个数字。 由于奖励信号的更具有稀疏性，因此难度加大（因为必须先完成更多正确的操作，然后才能产生正确的输出结果）。
  
  ![ReversedAddition3-v0.gif](https://upload-images.jianshu.io/upload_images/15463866-221bc5ca12358497.gif?imageMogr2/auto-orient/strip)

## 3、2D仿真环境（Box2D）

* BipedalWalker-v2

  Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

* BipedalWalkerHardcore-v2

  Hardcore version with ladders, stumps, pitfalls. Time limit is increased due to obstacles. Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

* CarRacing-v0

  Easiest continuous control task to learn from pixels, a top-down racing environment. Discreet control is reasonable in this environment as well, on/off discretisation is fine. State consists of 96x96 pixels. Reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles in track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points. Episode finishes when all tiles are visited. Some indicators shown at the bottom of the window and the state RGB buffer. From left to right: true speed, four ABS sensors, steering wheel position, gyroscope.

* LunarLander-v2

  Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

* LunarLanderContinuous-v2

* Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.

## 4、 MuJoCo环境

* # Ant-v2

* Make a four-legged creature walk forward as fast as possible.

* # HalfCheetah-v2

* # Hopper-v2

* Make a two-dimensional one-legged robot hop forward as fast as possible.

* # Humanoid-v2

* Make a three-dimensional bipedal robot walk forward as fast as possible, without falling over.

* # HumanoidStandup-v2

* Make a three-dimensional bipedal robot standup as fast as possible.

* # InvertedDoublePendulum-v2

* # InvertedPendulum-v2

* # Reacher-v2

* # Swimmer-v2

* This task involves a 3-link swimming robot in a viscous fluid, where the goal is to make it swim forward as fast as possible, by actuating the two joints. The origins of task can be traced back to Remi Coulom's thesis [[1\]](http://gym.openai.com/envs/Swimmer-v2/#id2).

* # Walker2d-v2

  Make a two-dimensional bipedal robot walk forward as fast as possible.

## 5、Atari视频游戏环境

利用强化学习来玩雅达利的游戏。*Gym*中集成了对强化学习有着重要影响的*Arcade Learning Environment*，并且方便用户安装；一共118个

# Boxing-ram-v0

Maximize your score in the Atari 2600 game Boxing. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}

# Breakout-ram-v0

Maximize your score in the Atari 2600 game Breakout. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}{2,3,4}.

# Enduro-ram-v0

Maximize your score in the Atari 2600 game Enduro. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}{2,3,4}.

# Qbert-ram-v0

Maximize your score in the Atari 2600 game Qbert. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}{2,3,4}.

# Seaquest-ram-v0

Maximize your score in the Atari 2600 game Seaquest. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}{2,3,4}.



# SpaceInvaders-ram-v0

Maximize your score in the Atari 2600 game SpaceInvaders. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k*k* frames, where k*k* is uniformly sampled from \{2, 3, 4\}{2,3,4}.







## 6、机器人环境（Robotics）

* # FetchPickAndPlace-v1

  A goal is randomly chosen in 3D space. Control Fetch's end effector to grasp and lift the block up to reach that goal.

* # FetchPush-v1

  A goal position is randomly chosen on the table surface. Control Fetch's end effector to push the block towards that position..

* # FetchReach-v1

  A goal position is randomly chosen in 3D space. Control Fetch's end effector to reach that goal as quickly as possible.

* # FetchSlide-v1

  A goal position is chosen on the table in front of Fetch, out of reach for the robot. Control Fetch's end effector to slide the given puck to this goal.

* # HandManipulateBlock-v0

  A goal orientation is randomly chosen for a block which is placed in the ShadowHand's grip. Control the ShadowHand actuators to reach the given target orientation for the block.

* # HandManipulateEgg-v0

  A goal orientation is randomly chosen for an egg which is placed in the ShadowHand's grip. Control the ShadowHand actuators to reach the given target orientation for the egg.

* # HandManipulatePen-v0

  A goal orientation is randomly chosen for a pen which is placed in the ShadowHand's grip. Control the ShadowHand actuators to reach the given target orientation for the pen.

* # HandReach-v0

  A goal hand pose is randomly chosen in 3D space. Control the ShadowHand actuators to reach this position for all the five fingers.

## 7、文字游戏环境（Toy text）

Simple text environments to get you started.

# Blackjack-v0

# GuessingGame-v0

> The goal of the game is to guess within 1% of the randomly chosen number within 200 time steps
>
> After each step the agent is provided with one of four possible observations which indicate where the guess is in relation to the randomly chosen number
>
> 0 - No guess yet submitted (only after reset) 1 - Guess is lower than the target 2 - Guess is equal to the target 3 - Guess is higher than the target
>
> The rewards are: 0 if the agent's guess is outside of 1% of the target 1 if the agent's guess is inside 1% of the target
>
> The episode terminates after the agent guesses within 1% of the target or 200 steps have been taken
>
> The agent will need to use a memory of previously submitted actions and observations in order to efficiently explore the available actions.

# HotterColder-v0

> The goal of the game is to effective use the reward provided in order to understand the best action to take.
>
> After each step the agent receives an observation of: 0 - No guess yet submitted (only after reset) 1 - Guess is lower than the target 2 - Guess is equal to the target 3 - Guess is higher than the target
>
> The rewards is calculated as: ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2 This is essentially the squared percentage of the way the agent has guessed toward the target.
>
> Ideally an agent will be able to recognise the 'scent' of a higher reward and increase the rate in which is guesses in that direction until the reward reaches its maximum.

# NChain-v0

> n-Chain environment
>
> - This game presents moves along a linear chain of states, with two actions:
>
>   forward, which moves along the chain but returns no rewardbackward, which returns to the beginning and has a small reward
>
> The end of the chain, however, presents a large reward, and by moving 'forward' at the end of the chain this large reward can be repeated.
>
> At each action, there is a small probability that the agent 'slips' and the opposite transition is instead taken.
>
> The observed state is the current state in the chain (0 to n-1).

# Roulette-v0

The agent plays 0-to-36 Roulette in a modified casino setting. For each spin, the agent bets on a number. The agent receives a positive reward iff the rolled number is not zero and its parity matches the agent's bet. Additionally, the agent can choose to walk away from the table, ending the episode.

# FrozenLake-v0

The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

# FrozenLake8x8-v0

The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

# Taxi-v3

This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.





## 8、其他注册的环境

上面只能介绍到主要的环境，还有大量的注册环境，可以通过查询注册表找到：

```python
from gym import envs
print(envs.registry.all())
```

这里列出来在0.17.3版本以查询到的所有环境：

 dict_values([EnvSpec(Copy-v0)
 EnvSpec(RepeatCopy-v0)
 EnvSpec(ReversedAddition-v0)
 EnvSpec(ReversedAddition3-v0)
 EnvSpec(DuplicatedInput-v0)
 EnvSpec(Reverse-v0)
 EnvSpec(CartPole-v0)
 EnvSpec(CartPole-v1)
 EnvSpec(MountainCar-v0)
 EnvSpec(MountainCarContinuous-v0)
 EnvSpec(Pendulum-v0)
 EnvSpec(Acrobot-v1)
 EnvSpec(LunarLander-v2)
 EnvSpec(LunarLanderContinuous-v2)
 EnvSpec(BipedalWalker-v3)
 EnvSpec(BipedalWalkerHardcore-v3)
 EnvSpec(CarRacing-v0)
 EnvSpec(Blackjack-v0)
 EnvSpec(KellyCoinflip-v0)
 EnvSpec(KellyCoinflipGeneralized-v0)
 EnvSpec(FrozenLake-v0)
 EnvSpec(FrozenLake8x8-v0)
 EnvSpec(CliffWalking-v0)
 EnvSpec(NChain-v0)
 EnvSpec(Roulette-v0)
 EnvSpec(Taxi-v3)
 EnvSpec(GuessingGame-v0)
 EnvSpec(HotterColder-v0)
 EnvSpec(Reacher-v2)
 EnvSpec(Pusher-v2)
 EnvSpec(Thrower-v2)
 EnvSpec(Striker-v2)
 EnvSpec(InvertedPendulum-v2)
 EnvSpec(InvertedDoublePendulum-v2)
 EnvSpec(HalfCheetah-v2)
 EnvSpec(HalfCheetah-v3)
 EnvSpec(Hopper-v2)
 EnvSpec(Hopper-v3)
 EnvSpec(Swimmer-v2)
 EnvSpec(Swimmer-v3)
 EnvSpec(Walker2d-v2)
 EnvSpec(Walker2d-v3)
 EnvSpec(Ant-v2)
 EnvSpec(Ant-v3)
 EnvSpec(Humanoid-v2)
 EnvSpec(Humanoid-v3)
 EnvSpec(HumanoidStandup-v2)
 EnvSpec(FetchSlide-v1)
 EnvSpec(FetchPickAndPlace-v1)
 EnvSpec(FetchReach-v1)
 EnvSpec(FetchPush-v1)
 EnvSpec(HandReach-v0)
 EnvSpec(HandManipulateBlockRotateZ-v0)
 EnvSpec(HandManipulateBlockRotateZTouchSensors-v0)
 EnvSpec(HandManipulateBlockRotateZTouchSensors-v1)
 EnvSpec(HandManipulateBlockRotateParallel-v0)
 EnvSpec(HandManipulateBlockRotateParallelTouchSensors-v0)
 EnvSpec(HandManipulateBlockRotateParallelTouchSensors-v1)
 EnvSpec(HandManipulateBlockRotateXYZ-v0)
 EnvSpec(HandManipulateBlockRotateXYZTouchSensors-v0)
 EnvSpec(HandManipulateBlockRotateXYZTouchSensors-v1)
 EnvSpec(HandManipulateBlockFull-v0)
 EnvSpec(HandManipulateBlock-v0)
 EnvSpec(HandManipulateBlockTouchSensors-v0)
 EnvSpec(HandManipulateBlockTouchSensors-v1)
 EnvSpec(HandManipulateEggRotate-v0)
 EnvSpec(HandManipulateEggRotateTouchSensors-v0)
 EnvSpec(HandManipulateEggRotateTouchSensors-v1)
 EnvSpec(HandManipulateEggFull-v0)
 EnvSpec(HandManipulateEgg-v0)
 EnvSpec(HandManipulateEggTouchSensors-v0)
 EnvSpec(HandManipulateEggTouchSensors-v1)
 EnvSpec(HandManipulatePenRotate-v0)
 EnvSpec(HandManipulatePenRotateTouchSensors-v0)
 EnvSpec(HandManipulatePenRotateTouchSensors-v1)
 EnvSpec(HandManipulatePenFull-v0)
 EnvSpec(HandManipulatePen-v0)
 EnvSpec(HandManipulatePenTouchSensors-v0)
 EnvSpec(HandManipulatePenTouchSensors-v1)
 EnvSpec(FetchSlideDense-v1)
 EnvSpec(FetchPickAndPlaceDense-v1)
 EnvSpec(FetchReachDense-v1)
 EnvSpec(FetchPushDense-v1)
 EnvSpec(HandReachDense-v0)
 EnvSpec(HandManipulateBlockRotateZDense-v0)
 EnvSpec(HandManipulateBlockRotateZTouchSensorsDense-v0)
 EnvSpec(HandManipulateBlockRotateZTouchSensorsDense-v1)
 EnvSpec(HandManipulateBlockRotateParallelDense-v0)
 EnvSpec(HandManipulateBlockRotateParallelTouchSensorsDense-v0)
 EnvSpec(HandManipulateBlockRotateParallelTouchSensorsDense-v1)
 EnvSpec(HandManipulateBlockRotateXYZDense-v0)
 EnvSpec(HandManipulateBlockRotateXYZTouchSensorsDense-v0)
 EnvSpec(HandManipulateBlockRotateXYZTouchSensorsDense-v1)
 EnvSpec(HandManipulateBlockFullDense-v0)
 EnvSpec(HandManipulateBlockDense-v0)
 EnvSpec(HandManipulateBlockTouchSensorsDense-v0)
 EnvSpec(HandManipulateBlockTouchSensorsDense-v1)
 EnvSpec(HandManipulateEggRotateDense-v0)
 EnvSpec(HandManipulateEggRotateTouchSensorsDense-v0)
 EnvSpec(HandManipulateEggRotateTouchSensorsDense-v1)
 EnvSpec(HandManipulateEggFullDense-v0)
 EnvSpec(HandManipulateEggDense-v0)
 EnvSpec(HandManipulateEggTouchSensorsDense-v0)
 EnvSpec(HandManipulateEggTouchSensorsDense-v1)
 EnvSpec(HandManipulatePenRotateDense-v0)
 EnvSpec(HandManipulatePenRotateTouchSensorsDense-v0)
 EnvSpec(HandManipulatePenRotateTouchSensorsDense-v1)
 EnvSpec(HandManipulatePenFullDense-v0)
 EnvSpec(HandManipulatePenDense-v0)
 EnvSpec(HandManipulatePenTouchSensorsDense-v0)
 EnvSpec(HandManipulatePenTouchSensorsDense-v1)
 EnvSpec(Adventure-v0)
 EnvSpec(Adventure-v4)
 EnvSpec(AdventureDeterministic-v0)
 EnvSpec(AdventureDeterministic-v4)
 EnvSpec(AdventureNoFrameskip-v0)
 EnvSpec(AdventureNoFrameskip-v4)
 EnvSpec(Adventure-ram-v0)
 EnvSpec(Adventure-ram-v4)
 EnvSpec(Adventure-ramDeterministic-v0)
 EnvSpec(Adventure-ramDeterministic-v4)
 EnvSpec(Adventure-ramNoFrameskip-v0)
 EnvSpec(Adventure-ramNoFrameskip-v4)
 EnvSpec(AirRaid-v0)
 EnvSpec(AirRaid-v4)
 EnvSpec(AirRaidDeterministic-v0)
 EnvSpec(AirRaidDeterministic-v4)
 EnvSpec(AirRaidNoFrameskip-v0)
 EnvSpec(AirRaidNoFrameskip-v4)
 EnvSpec(AirRaid-ram-v0)
 EnvSpec(AirRaid-ram-v4)
 EnvSpec(AirRaid-ramDeterministic-v0)
 EnvSpec(AirRaid-ramDeterministic-v4)
 EnvSpec(AirRaid-ramNoFrameskip-v0)
 EnvSpec(AirRaid-ramNoFrameskip-v4)
 EnvSpec(Alien-v0)
 EnvSpec(Alien-v4)
 EnvSpec(AlienDeterministic-v0)
 EnvSpec(AlienDeterministic-v4)
 EnvSpec(AlienNoFrameskip-v0)
 EnvSpec(AlienNoFrameskip-v4)
 EnvSpec(Alien-ram-v0)
 EnvSpec(Alien-ram-v4)
 EnvSpec(Alien-ramDeterministic-v0)
 EnvSpec(Alien-ramDeterministic-v4)
 EnvSpec(Alien-ramNoFrameskip-v0)
 EnvSpec(Alien-ramNoFrameskip-v4)
 EnvSpec(Amidar-v0)
 EnvSpec(Amidar-v4)
 EnvSpec(AmidarDeterministic-v0)
 EnvSpec(AmidarDeterministic-v4)
 EnvSpec(AmidarNoFrameskip-v0)
 EnvSpec(AmidarNoFrameskip-v4)
 EnvSpec(Amidar-ram-v0)
 EnvSpec(Amidar-ram-v4)
 EnvSpec(Amidar-ramDeterministic-v0)
 EnvSpec(Amidar-ramDeterministic-v4)
 EnvSpec(Amidar-ramNoFrameskip-v0)
 EnvSpec(Amidar-ramNoFrameskip-v4)
 EnvSpec(Assault-v0)
 EnvSpec(Assault-v4)
 EnvSpec(AssaultDeterministic-v0)
 EnvSpec(AssaultDeterministic-v4)
 EnvSpec(AssaultNoFrameskip-v0)
 EnvSpec(AssaultNoFrameskip-v4)
 EnvSpec(Assault-ram-v0)
 EnvSpec(Assault-ram-v4)
 EnvSpec(Assault-ramDeterministic-v0)
 EnvSpec(Assault-ramDeterministic-v4)
 EnvSpec(Assault-ramNoFrameskip-v0)
 EnvSpec(Assault-ramNoFrameskip-v4)
 EnvSpec(Asterix-v0)
 EnvSpec(Asterix-v4)
 EnvSpec(AsterixDeterministic-v0)
 EnvSpec(AsterixDeterministic-v4)
 EnvSpec(AsterixNoFrameskip-v0)
 EnvSpec(AsterixNoFrameskip-v4)
 EnvSpec(Asterix-ram-v0)
 EnvSpec(Asterix-ram-v4)
 EnvSpec(Asterix-ramDeterministic-v0)
 EnvSpec(Asterix-ramDeterministic-v4)
 EnvSpec(Asterix-ramNoFrameskip-v0)
 EnvSpec(Asterix-ramNoFrameskip-v4)
 EnvSpec(Asteroids-v0)
 EnvSpec(Asteroids-v4)
 EnvSpec(AsteroidsDeterministic-v0)
 EnvSpec(AsteroidsDeterministic-v4)
 EnvSpec(AsteroidsNoFrameskip-v0)
 EnvSpec(AsteroidsNoFrameskip-v4)
 EnvSpec(Asteroids-ram-v0)
 EnvSpec(Asteroids-ram-v4)
 EnvSpec(Asteroids-ramDeterministic-v0)
 EnvSpec(Asteroids-ramDeterministic-v4)
 EnvSpec(Asteroids-ramNoFrameskip-v0)
 EnvSpec(Asteroids-ramNoFrameskip-v4)
 EnvSpec(Atlantis-v0)
 EnvSpec(Atlantis-v4)
 EnvSpec(AtlantisDeterministic-v0)
 EnvSpec(AtlantisDeterministic-v4)
 EnvSpec(AtlantisNoFrameskip-v0)
 EnvSpec(AtlantisNoFrameskip-v4)
 EnvSpec(Atlantis-ram-v0)
 EnvSpec(Atlantis-ram-v4)
 EnvSpec(Atlantis-ramDeterministic-v0)
 EnvSpec(Atlantis-ramDeterministic-v4)
 EnvSpec(Atlantis-ramNoFrameskip-v0)
 EnvSpec(Atlantis-ramNoFrameskip-v4)
 EnvSpec(BankHeist-v0)
 EnvSpec(BankHeist-v4)
 EnvSpec(BankHeistDeterministic-v0)
 EnvSpec(BankHeistDeterministic-v4)
 EnvSpec(BankHeistNoFrameskip-v0)
 EnvSpec(BankHeistNoFrameskip-v4)
 EnvSpec(BankHeist-ram-v0)
 EnvSpec(BankHeist-ram-v4)
 EnvSpec(BankHeist-ramDeterministic-v0)
 EnvSpec(BankHeist-ramDeterministic-v4)
 EnvSpec(BankHeist-ramNoFrameskip-v0)
 EnvSpec(BankHeist-ramNoFrameskip-v4)
 EnvSpec(BattleZone-v0)
 EnvSpec(BattleZone-v4)
 EnvSpec(BattleZoneDeterministic-v0)
 EnvSpec(BattleZoneDeterministic-v4)
 EnvSpec(BattleZoneNoFrameskip-v0)
 EnvSpec(BattleZoneNoFrameskip-v4)
 EnvSpec(BattleZone-ram-v0)
 EnvSpec(BattleZone-ram-v4)
 EnvSpec(BattleZone-ramDeterministic-v0)
 EnvSpec(BattleZone-ramDeterministic-v4)
 EnvSpec(BattleZone-ramNoFrameskip-v0)
 EnvSpec(BattleZone-ramNoFrameskip-v4)
 EnvSpec(BeamRider-v0)
 EnvSpec(BeamRider-v4)
 EnvSpec(BeamRiderDeterministic-v0)
 EnvSpec(BeamRiderDeterministic-v4)
 EnvSpec(BeamRiderNoFrameskip-v0)
 EnvSpec(BeamRiderNoFrameskip-v4)
 EnvSpec(BeamRider-ram-v0)
 EnvSpec(BeamRider-ram-v4)
 EnvSpec(BeamRider-ramDeterministic-v0)
 EnvSpec(BeamRider-ramDeterministic-v4)
 EnvSpec(BeamRider-ramNoFrameskip-v0)
 EnvSpec(BeamRider-ramNoFrameskip-v4)
 EnvSpec(Berzerk-v0)
 EnvSpec(Berzerk-v4)
 EnvSpec(BerzerkDeterministic-v0)
 EnvSpec(BerzerkDeterministic-v4)
 EnvSpec(BerzerkNoFrameskip-v0)
 EnvSpec(BerzerkNoFrameskip-v4)
 EnvSpec(Berzerk-ram-v0)
 EnvSpec(Berzerk-ram-v4)
 EnvSpec(Berzerk-ramDeterministic-v0)
 EnvSpec(Berzerk-ramDeterministic-v4)
 EnvSpec(Berzerk-ramNoFrameskip-v0)
 EnvSpec(Berzerk-ramNoFrameskip-v4)
 EnvSpec(Bowling-v0)
 EnvSpec(Bowling-v4)
 EnvSpec(BowlingDeterministic-v0)
 EnvSpec(BowlingDeterministic-v4)
 EnvSpec(BowlingNoFrameskip-v0)
 EnvSpec(BowlingNoFrameskip-v4)
 EnvSpec(Bowling-ram-v0)
 EnvSpec(Bowling-ram-v4)
 EnvSpec(Bowling-ramDeterministic-v0)
 EnvSpec(Bowling-ramDeterministic-v4)
 EnvSpec(Bowling-ramNoFrameskip-v0)
 EnvSpec(Bowling-ramNoFrameskip-v4)
 EnvSpec(Boxing-v0)
 EnvSpec(Boxing-v4)
 EnvSpec(BoxingDeterministic-v0)
 EnvSpec(BoxingDeterministic-v4)
 EnvSpec(BoxingNoFrameskip-v0)
 EnvSpec(BoxingNoFrameskip-v4)
 EnvSpec(Boxing-ram-v0)
 EnvSpec(Boxing-ram-v4)
 EnvSpec(Boxing-ramDeterministic-v0)
 EnvSpec(Boxing-ramDeterministic-v4)
 EnvSpec(Boxing-ramNoFrameskip-v0)
 EnvSpec(Boxing-ramNoFrameskip-v4)
 EnvSpec(Breakout-v0)
 EnvSpec(Breakout-v4)
 EnvSpec(BreakoutDeterministic-v0)
 EnvSpec(BreakoutDeterministic-v4)
 EnvSpec(BreakoutNoFrameskip-v0)
 EnvSpec(BreakoutNoFrameskip-v4)
 EnvSpec(Breakout-ram-v0)
 EnvSpec(Breakout-ram-v4)
 EnvSpec(Breakout-ramDeterministic-v0)
 EnvSpec(Breakout-ramDeterministic-v4)
 EnvSpec(Breakout-ramNoFrameskip-v0)
 EnvSpec(Breakout-ramNoFrameskip-v4)
 EnvSpec(Carnival-v0)
 EnvSpec(Carnival-v4)
 EnvSpec(CarnivalDeterministic-v0)
 EnvSpec(CarnivalDeterministic-v4)
 EnvSpec(CarnivalNoFrameskip-v0)
 EnvSpec(CarnivalNoFrameskip-v4)
 EnvSpec(Carnival-ram-v0)
 EnvSpec(Carnival-ram-v4)
 EnvSpec(Carnival-ramDeterministic-v0)
 EnvSpec(Carnival-ramDeterministic-v4)
 EnvSpec(Carnival-ramNoFrameskip-v0)
 EnvSpec(Carnival-ramNoFrameskip-v4)
 EnvSpec(Centipede-v0)
 EnvSpec(Centipede-v4)
 EnvSpec(CentipedeDeterministic-v0)
 EnvSpec(CentipedeDeterministic-v4)
 EnvSpec(CentipedeNoFrameskip-v0)
 EnvSpec(CentipedeNoFrameskip-v4)
 EnvSpec(Centipede-ram-v0)
 EnvSpec(Centipede-ram-v4)
 EnvSpec(Centipede-ramDeterministic-v0)
 EnvSpec(Centipede-ramDeterministic-v4)
 EnvSpec(Centipede-ramNoFrameskip-v0)
 EnvSpec(Centipede-ramNoFrameskip-v4)
 EnvSpec(ChopperCommand-v0)
 EnvSpec(ChopperCommand-v4)
 EnvSpec(ChopperCommandDeterministic-v0)
 EnvSpec(ChopperCommandDeterministic-v4)
 EnvSpec(ChopperCommandNoFrameskip-v0)
 EnvSpec(ChopperCommandNoFrameskip-v4)
 EnvSpec(ChopperCommand-ram-v0)
 EnvSpec(ChopperCommand-ram-v4)
 EnvSpec(ChopperCommand-ramDeterministic-v0)
 EnvSpec(ChopperCommand-ramDeterministic-v4)
 EnvSpec(ChopperCommand-ramNoFrameskip-v0)
 EnvSpec(ChopperCommand-ramNoFrameskip-v4)
 EnvSpec(CrazyClimber-v0)
 EnvSpec(CrazyClimber-v4)
 EnvSpec(CrazyClimberDeterministic-v0)
 EnvSpec(CrazyClimberDeterministic-v4)
 EnvSpec(CrazyClimberNoFrameskip-v0)
 EnvSpec(CrazyClimberNoFrameskip-v4)
 EnvSpec(CrazyClimber-ram-v0)
 EnvSpec(CrazyClimber-ram-v4)
 EnvSpec(CrazyClimber-ramDeterministic-v0)
 EnvSpec(CrazyClimber-ramDeterministic-v4)
 EnvSpec(CrazyClimber-ramNoFrameskip-v0)
 EnvSpec(CrazyClimber-ramNoFrameskip-v4)
 EnvSpec(Defender-v0)
 EnvSpec(Defender-v4)
 EnvSpec(DefenderDeterministic-v0)
 EnvSpec(DefenderDeterministic-v4)
 EnvSpec(DefenderNoFrameskip-v0)
 EnvSpec(DefenderNoFrameskip-v4)
 EnvSpec(Defender-ram-v0)
 EnvSpec(Defender-ram-v4)
 EnvSpec(Defender-ramDeterministic-v0)
 EnvSpec(Defender-ramDeterministic-v4)
 EnvSpec(Defender-ramNoFrameskip-v0)
 EnvSpec(Defender-ramNoFrameskip-v4)
 EnvSpec(DemonAttack-v0)
 EnvSpec(DemonAttack-v4)
 EnvSpec(DemonAttackDeterministic-v0)
 EnvSpec(DemonAttackDeterministic-v4)
 EnvSpec(DemonAttackNoFrameskip-v0)
 EnvSpec(DemonAttackNoFrameskip-v4)
 EnvSpec(DemonAttack-ram-v0)
 EnvSpec(DemonAttack-ram-v4)
 EnvSpec(DemonAttack-ramDeterministic-v0)
 EnvSpec(DemonAttack-ramDeterministic-v4)
 EnvSpec(DemonAttack-ramNoFrameskip-v0)
 EnvSpec(DemonAttack-ramNoFrameskip-v4)
 EnvSpec(DoubleDunk-v0)
 EnvSpec(DoubleDunk-v4)
 EnvSpec(DoubleDunkDeterministic-v0)
 EnvSpec(DoubleDunkDeterministic-v4)
 EnvSpec(DoubleDunkNoFrameskip-v0)
 EnvSpec(DoubleDunkNoFrameskip-v4)
 EnvSpec(DoubleDunk-ram-v0)
 EnvSpec(DoubleDunk-ram-v4)
 EnvSpec(DoubleDunk-ramDeterministic-v0)
 EnvSpec(DoubleDunk-ramDeterministic-v4)
 EnvSpec(DoubleDunk-ramNoFrameskip-v0)
 EnvSpec(DoubleDunk-ramNoFrameskip-v4)
 EnvSpec(ElevatorAction-v0)
 EnvSpec(ElevatorAction-v4)
 EnvSpec(ElevatorActionDeterministic-v0)
 EnvSpec(ElevatorActionDeterministic-v4)
 EnvSpec(ElevatorActionNoFrameskip-v0)
 EnvSpec(ElevatorActionNoFrameskip-v4)
 EnvSpec(ElevatorAction-ram-v0)
 EnvSpec(ElevatorAction-ram-v4)
 EnvSpec(ElevatorAction-ramDeterministic-v0)
 EnvSpec(ElevatorAction-ramDeterministic-v4)
 EnvSpec(ElevatorAction-ramNoFrameskip-v0)
 EnvSpec(ElevatorAction-ramNoFrameskip-v4)
 EnvSpec(Enduro-v0)
 EnvSpec(Enduro-v4)
 EnvSpec(EnduroDeterministic-v0)
 EnvSpec(EnduroDeterministic-v4)
 EnvSpec(EnduroNoFrameskip-v0)
 EnvSpec(EnduroNoFrameskip-v4)
 EnvSpec(Enduro-ram-v0)
 EnvSpec(Enduro-ram-v4)
 EnvSpec(Enduro-ramDeterministic-v0)
 EnvSpec(Enduro-ramDeterministic-v4)
 EnvSpec(Enduro-ramNoFrameskip-v0)
 EnvSpec(Enduro-ramNoFrameskip-v4)
 EnvSpec(FishingDerby-v0)
 EnvSpec(FishingDerby-v4)
 EnvSpec(FishingDerbyDeterministic-v0)
 EnvSpec(FishingDerbyDeterministic-v4)
 EnvSpec(FishingDerbyNoFrameskip-v0)
 EnvSpec(FishingDerbyNoFrameskip-v4)
 EnvSpec(FishingDerby-ram-v0)
 EnvSpec(FishingDerby-ram-v4)
 EnvSpec(FishingDerby-ramDeterministic-v0)
 EnvSpec(FishingDerby-ramDeterministic-v4)
 EnvSpec(FishingDerby-ramNoFrameskip-v0)
 EnvSpec(FishingDerby-ramNoFrameskip-v4)
 EnvSpec(Freeway-v0)
 EnvSpec(Freeway-v4)
 EnvSpec(FreewayDeterministic-v0)
 EnvSpec(FreewayDeterministic-v4)
 EnvSpec(FreewayNoFrameskip-v0)
 EnvSpec(FreewayNoFrameskip-v4)
 EnvSpec(Freeway-ram-v0)
 EnvSpec(Freeway-ram-v4)
 EnvSpec(Freeway-ramDeterministic-v0)
 EnvSpec(Freeway-ramDeterministic-v4)
 EnvSpec(Freeway-ramNoFrameskip-v0)
 EnvSpec(Freeway-ramNoFrameskip-v4)
 EnvSpec(Frostbite-v0)
 EnvSpec(Frostbite-v4)
 EnvSpec(FrostbiteDeterministic-v0)
 EnvSpec(FrostbiteDeterministic-v4)
 EnvSpec(FrostbiteNoFrameskip-v0)
 EnvSpec(FrostbiteNoFrameskip-v4)
 EnvSpec(Frostbite-ram-v0)
 EnvSpec(Frostbite-ram-v4)
 EnvSpec(Frostbite-ramDeterministic-v0)
 EnvSpec(Frostbite-ramDeterministic-v4)
 EnvSpec(Frostbite-ramNoFrameskip-v0)
 EnvSpec(Frostbite-ramNoFrameskip-v4)
 EnvSpec(Gopher-v0)
 EnvSpec(Gopher-v4)
 EnvSpec(GopherDeterministic-v0)
 EnvSpec(GopherDeterministic-v4)
 EnvSpec(GopherNoFrameskip-v0)
 EnvSpec(GopherNoFrameskip-v4)
 EnvSpec(Gopher-ram-v0)
 EnvSpec(Gopher-ram-v4)
 EnvSpec(Gopher-ramDeterministic-v0)
 EnvSpec(Gopher-ramDeterministic-v4)
 EnvSpec(Gopher-ramNoFrameskip-v0)
 EnvSpec(Gopher-ramNoFrameskip-v4)
 EnvSpec(Gravitar-v0)
 EnvSpec(Gravitar-v4)
 EnvSpec(GravitarDeterministic-v0)
 EnvSpec(GravitarDeterministic-v4)
 EnvSpec(GravitarNoFrameskip-v0)
 EnvSpec(GravitarNoFrameskip-v4)
 EnvSpec(Gravitar-ram-v0)
 EnvSpec(Gravitar-ram-v4)
 EnvSpec(Gravitar-ramDeterministic-v0)
 EnvSpec(Gravitar-ramDeterministic-v4)
 EnvSpec(Gravitar-ramNoFrameskip-v0)
 EnvSpec(Gravitar-ramNoFrameskip-v4)
 EnvSpec(Hero-v0)
 EnvSpec(Hero-v4)
 EnvSpec(HeroDeterministic-v0)
 EnvSpec(HeroDeterministic-v4)
 EnvSpec(HeroNoFrameskip-v0)
 EnvSpec(HeroNoFrameskip-v4)
 EnvSpec(Hero-ram-v0)
 EnvSpec(Hero-ram-v4)
 EnvSpec(Hero-ramDeterministic-v0)
 EnvSpec(Hero-ramDeterministic-v4)
 EnvSpec(Hero-ramNoFrameskip-v0)
 EnvSpec(Hero-ramNoFrameskip-v4)
 EnvSpec(IceHockey-v0)
 EnvSpec(IceHockey-v4)
 EnvSpec(IceHockeyDeterministic-v0)
 EnvSpec(IceHockeyDeterministic-v4)
 EnvSpec(IceHockeyNoFrameskip-v0)
 EnvSpec(IceHockeyNoFrameskip-v4)
 EnvSpec(IceHockey-ram-v0)
 EnvSpec(IceHockey-ram-v4)
 EnvSpec(IceHockey-ramDeterministic-v0)
 EnvSpec(IceHockey-ramDeterministic-v4)
 EnvSpec(IceHockey-ramNoFrameskip-v0)
 EnvSpec(IceHockey-ramNoFrameskip-v4)
 EnvSpec(Jamesbond-v0)
 EnvSpec(Jamesbond-v4)
 EnvSpec(JamesbondDeterministic-v0)
 EnvSpec(JamesbondDeterministic-v4)
 EnvSpec(JamesbondNoFrameskip-v0)
 EnvSpec(JamesbondNoFrameskip-v4)
 EnvSpec(Jamesbond-ram-v0)
 EnvSpec(Jamesbond-ram-v4)
 EnvSpec(Jamesbond-ramDeterministic-v0)
 EnvSpec(Jamesbond-ramDeterministic-v4)
 EnvSpec(Jamesbond-ramNoFrameskip-v0)
 EnvSpec(Jamesbond-ramNoFrameskip-v4)
 EnvSpec(JourneyEscape-v0)
 EnvSpec(JourneyEscape-v4)
 EnvSpec(JourneyEscapeDeterministic-v0)
 EnvSpec(JourneyEscapeDeterministic-v4)
 EnvSpec(JourneyEscapeNoFrameskip-v0)
 EnvSpec(JourneyEscapeNoFrameskip-v4)
 EnvSpec(JourneyEscape-ram-v0)
 EnvSpec(JourneyEscape-ram-v4)
 EnvSpec(JourneyEscape-ramDeterministic-v0)
 EnvSpec(JourneyEscape-ramDeterministic-v4)
 EnvSpec(JourneyEscape-ramNoFrameskip-v0)
 EnvSpec(JourneyEscape-ramNoFrameskip-v4)
 EnvSpec(Kangaroo-v0)
 EnvSpec(Kangaroo-v4)
 EnvSpec(KangarooDeterministic-v0)
 EnvSpec(KangarooDeterministic-v4)
 EnvSpec(KangarooNoFrameskip-v0)
 EnvSpec(KangarooNoFrameskip-v4)
 EnvSpec(Kangaroo-ram-v0)
 EnvSpec(Kangaroo-ram-v4)
 EnvSpec(Kangaroo-ramDeterministic-v0)
 EnvSpec(Kangaroo-ramDeterministic-v4)
 EnvSpec(Kangaroo-ramNoFrameskip-v0)
 EnvSpec(Kangaroo-ramNoFrameskip-v4)
 EnvSpec(Krull-v0)
 EnvSpec(Krull-v4)
 EnvSpec(KrullDeterministic-v0)
 EnvSpec(KrullDeterministic-v4)
 EnvSpec(KrullNoFrameskip-v0)
 EnvSpec(KrullNoFrameskip-v4)
 EnvSpec(Krull-ram-v0)
 EnvSpec(Krull-ram-v4)
 EnvSpec(Krull-ramDeterministic-v0)
 EnvSpec(Krull-ramDeterministic-v4)
 EnvSpec(Krull-ramNoFrameskip-v0)
 EnvSpec(Krull-ramNoFrameskip-v4)
 EnvSpec(KungFuMaster-v0)
 EnvSpec(KungFuMaster-v4)
 EnvSpec(KungFuMasterDeterministic-v0)
 EnvSpec(KungFuMasterDeterministic-v4)
 EnvSpec(KungFuMasterNoFrameskip-v0)
 EnvSpec(KungFuMasterNoFrameskip-v4)
 EnvSpec(KungFuMaster-ram-v0)
 EnvSpec(KungFuMaster-ram-v4)
 EnvSpec(KungFuMaster-ramDeterministic-v0)
 EnvSpec(KungFuMaster-ramDeterministic-v4)
 EnvSpec(KungFuMaster-ramNoFrameskip-v0)
 EnvSpec(KungFuMaster-ramNoFrameskip-v4)
 EnvSpec(MontezumaRevenge-v0)
 EnvSpec(MontezumaRevenge-v4)
 EnvSpec(MontezumaRevengeDeterministic-v0)
 EnvSpec(MontezumaRevengeDeterministic-v4)
 EnvSpec(MontezumaRevengeNoFrameskip-v0)
 EnvSpec(MontezumaRevengeNoFrameskip-v4)
 EnvSpec(MontezumaRevenge-ram-v0)
 EnvSpec(MontezumaRevenge-ram-v4)
 EnvSpec(MontezumaRevenge-ramDeterministic-v0)
 EnvSpec(MontezumaRevenge-ramDeterministic-v4)
 EnvSpec(MontezumaRevenge-ramNoFrameskip-v0)
 EnvSpec(MontezumaRevenge-ramNoFrameskip-v4)
 EnvSpec(MsPacman-v0)
 EnvSpec(MsPacman-v4)
 EnvSpec(MsPacmanDeterministic-v0)
 EnvSpec(MsPacmanDeterministic-v4)
 EnvSpec(MsPacmanNoFrameskip-v0)
 EnvSpec(MsPacmanNoFrameskip-v4)
 EnvSpec(MsPacman-ram-v0)
 EnvSpec(MsPacman-ram-v4)
 EnvSpec(MsPacman-ramDeterministic-v0)
 EnvSpec(MsPacman-ramDeterministic-v4)
 EnvSpec(MsPacman-ramNoFrameskip-v0)
 EnvSpec(MsPacman-ramNoFrameskip-v4)
 EnvSpec(NameThisGame-v0)
 EnvSpec(NameThisGame-v4)
 EnvSpec(NameThisGameDeterministic-v0)
 EnvSpec(NameThisGameDeterministic-v4)
 EnvSpec(NameThisGameNoFrameskip-v0)
 EnvSpec(NameThisGameNoFrameskip-v4)
 EnvSpec(NameThisGame-ram-v0)
 EnvSpec(NameThisGame-ram-v4)
 EnvSpec(NameThisGame-ramDeterministic-v0)
 EnvSpec(NameThisGame-ramDeterministic-v4)
 EnvSpec(NameThisGame-ramNoFrameskip-v0)
 EnvSpec(NameThisGame-ramNoFrameskip-v4)
 EnvSpec(Phoenix-v0)
 EnvSpec(Phoenix-v4)
 EnvSpec(PhoenixDeterministic-v0)
 EnvSpec(PhoenixDeterministic-v4)
 EnvSpec(PhoenixNoFrameskip-v0)
 EnvSpec(PhoenixNoFrameskip-v4)
 EnvSpec(Phoenix-ram-v0)
 EnvSpec(Phoenix-ram-v4)
 EnvSpec(Phoenix-ramDeterministic-v0)
 EnvSpec(Phoenix-ramDeterministic-v4)
 EnvSpec(Phoenix-ramNoFrameskip-v0)
 EnvSpec(Phoenix-ramNoFrameskip-v4)
 EnvSpec(Pitfall-v0)
 EnvSpec(Pitfall-v4)
 EnvSpec(PitfallDeterministic-v0)
 EnvSpec(PitfallDeterministic-v4)
 EnvSpec(PitfallNoFrameskip-v0)
 EnvSpec(PitfallNoFrameskip-v4)
 EnvSpec(Pitfall-ram-v0)
 EnvSpec(Pitfall-ram-v4)
 EnvSpec(Pitfall-ramDeterministic-v0)
 EnvSpec(Pitfall-ramDeterministic-v4)
 EnvSpec(Pitfall-ramNoFrameskip-v0)
 EnvSpec(Pitfall-ramNoFrameskip-v4)
 EnvSpec(Pong-v0)
 EnvSpec(Pong-v4)
 EnvSpec(PongDeterministic-v0)
 EnvSpec(PongDeterministic-v4)
 EnvSpec(PongNoFrameskip-v0)
 EnvSpec(PongNoFrameskip-v4)
 EnvSpec(Pong-ram-v0)
 EnvSpec(Pong-ram-v4)
 EnvSpec(Pong-ramDeterministic-v0)
 EnvSpec(Pong-ramDeterministic-v4)
 EnvSpec(Pong-ramNoFrameskip-v0)
 EnvSpec(Pong-ramNoFrameskip-v4)
 EnvSpec(Pooyan-v0)
 EnvSpec(Pooyan-v4)
 EnvSpec(PooyanDeterministic-v0)
 EnvSpec(PooyanDeterministic-v4)
 EnvSpec(PooyanNoFrameskip-v0)
 EnvSpec(PooyanNoFrameskip-v4)
 EnvSpec(Pooyan-ram-v0)
 EnvSpec(Pooyan-ram-v4)
 EnvSpec(Pooyan-ramDeterministic-v0)
 EnvSpec(Pooyan-ramDeterministic-v4)
 EnvSpec(Pooyan-ramNoFrameskip-v0)
 EnvSpec(Pooyan-ramNoFrameskip-v4)
 EnvSpec(PrivateEye-v0)
 EnvSpec(PrivateEye-v4)
 EnvSpec(PrivateEyeDeterministic-v0)
 EnvSpec(PrivateEyeDeterministic-v4)
 EnvSpec(PrivateEyeNoFrameskip-v0)
 EnvSpec(PrivateEyeNoFrameskip-v4)
 EnvSpec(PrivateEye-ram-v0)
 EnvSpec(PrivateEye-ram-v4)
 EnvSpec(PrivateEye-ramDeterministic-v0)
 EnvSpec(PrivateEye-ramDeterministic-v4)
 EnvSpec(PrivateEye-ramNoFrameskip-v0)
 EnvSpec(PrivateEye-ramNoFrameskip-v4)
 EnvSpec(Qbert-v0)
 EnvSpec(Qbert-v4)
 EnvSpec(QbertDeterministic-v0)
 EnvSpec(QbertDeterministic-v4)
 EnvSpec(QbertNoFrameskip-v0)
 EnvSpec(QbertNoFrameskip-v4)
 EnvSpec(Qbert-ram-v0)
 EnvSpec(Qbert-ram-v4)
 EnvSpec(Qbert-ramDeterministic-v0)
 EnvSpec(Qbert-ramDeterministic-v4)
 EnvSpec(Qbert-ramNoFrameskip-v0)
 EnvSpec(Qbert-ramNoFrameskip-v4)
 EnvSpec(Riverraid-v0)
 EnvSpec(Riverraid-v4)
 EnvSpec(RiverraidDeterministic-v0)
 EnvSpec(RiverraidDeterministic-v4)
 EnvSpec(RiverraidNoFrameskip-v0)
 EnvSpec(RiverraidNoFrameskip-v4)
 EnvSpec(Riverraid-ram-v0)
 EnvSpec(Riverraid-ram-v4)
 EnvSpec(Riverraid-ramDeterministic-v0)
 EnvSpec(Riverraid-ramDeterministic-v4)
 EnvSpec(Riverraid-ramNoFrameskip-v0)
 EnvSpec(Riverraid-ramNoFrameskip-v4)
 EnvSpec(RoadRunner-v0)
 EnvSpec(RoadRunner-v4)
 EnvSpec(RoadRunnerDeterministic-v0)
 EnvSpec(RoadRunnerDeterministic-v4)
 EnvSpec(RoadRunnerNoFrameskip-v0)
 EnvSpec(RoadRunnerNoFrameskip-v4)
 EnvSpec(RoadRunner-ram-v0)
 EnvSpec(RoadRunner-ram-v4)
 EnvSpec(RoadRunner-ramDeterministic-v0)
 EnvSpec(RoadRunner-ramDeterministic-v4)
 EnvSpec(RoadRunner-ramNoFrameskip-v0)
 EnvSpec(RoadRunner-ramNoFrameskip-v4)
 EnvSpec(Robotank-v0)
 EnvSpec(Robotank-v4)
 EnvSpec(RobotankDeterministic-v0)
 EnvSpec(RobotankDeterministic-v4)
 EnvSpec(RobotankNoFrameskip-v0)
 EnvSpec(RobotankNoFrameskip-v4)
 EnvSpec(Robotank-ram-v0)
 EnvSpec(Robotank-ram-v4)
 EnvSpec(Robotank-ramDeterministic-v0)
 EnvSpec(Robotank-ramDeterministic-v4)
 EnvSpec(Robotank-ramNoFrameskip-v0)
 EnvSpec(Robotank-ramNoFrameskip-v4)
 EnvSpec(Seaquest-v0)
 EnvSpec(Seaquest-v4)
 EnvSpec(SeaquestDeterministic-v0)
 EnvSpec(SeaquestDeterministic-v4)
 EnvSpec(SeaquestNoFrameskip-v0)
 EnvSpec(SeaquestNoFrameskip-v4)
 EnvSpec(Seaquest-ram-v0)
 EnvSpec(Seaquest-ram-v4)
 EnvSpec(Seaquest-ramDeterministic-v0)
 EnvSpec(Seaquest-ramDeterministic-v4)
 EnvSpec(Seaquest-ramNoFrameskip-v0)
 EnvSpec(Seaquest-ramNoFrameskip-v4)
 EnvSpec(Skiing-v0)
 EnvSpec(Skiing-v4)
 EnvSpec(SkiingDeterministic-v0)
 EnvSpec(SkiingDeterministic-v4)
 EnvSpec(SkiingNoFrameskip-v0)
 EnvSpec(SkiingNoFrameskip-v4)
 EnvSpec(Skiing-ram-v0)
 EnvSpec(Skiing-ram-v4)
 EnvSpec(Skiing-ramDeterministic-v0)
 EnvSpec(Skiing-ramDeterministic-v4)
 EnvSpec(Skiing-ramNoFrameskip-v0)
 EnvSpec(Skiing-ramNoFrameskip-v4)
 EnvSpec(Solaris-v0)
 EnvSpec(Solaris-v4)
 EnvSpec(SolarisDeterministic-v0)
 EnvSpec(SolarisDeterministic-v4)
 EnvSpec(SolarisNoFrameskip-v0)
 EnvSpec(SolarisNoFrameskip-v4)
 EnvSpec(Solaris-ram-v0)
 EnvSpec(Solaris-ram-v4)
 EnvSpec(Solaris-ramDeterministic-v0)
 EnvSpec(Solaris-ramDeterministic-v4)
 EnvSpec(Solaris-ramNoFrameskip-v0)
 EnvSpec(Solaris-ramNoFrameskip-v4)
 EnvSpec(SpaceInvaders-v0)
 EnvSpec(SpaceInvaders-v4)
 EnvSpec(SpaceInvadersDeterministic-v0)
 EnvSpec(SpaceInvadersDeterministic-v4)
 EnvSpec(SpaceInvadersNoFrameskip-v0)
 EnvSpec(SpaceInvadersNoFrameskip-v4)
 EnvSpec(SpaceInvaders-ram-v0)
 EnvSpec(SpaceInvaders-ram-v4)
 EnvSpec(SpaceInvaders-ramDeterministic-v0)
 EnvSpec(SpaceInvaders-ramDeterministic-v4)
 EnvSpec(SpaceInvaders-ramNoFrameskip-v0)
 EnvSpec(SpaceInvaders-ramNoFrameskip-v4)
 EnvSpec(StarGunner-v0)
 EnvSpec(StarGunner-v4)
 EnvSpec(StarGunnerDeterministic-v0)
 EnvSpec(StarGunnerDeterministic-v4)
 EnvSpec(StarGunnerNoFrameskip-v0)
 EnvSpec(StarGunnerNoFrameskip-v4)
 EnvSpec(StarGunner-ram-v0)
 EnvSpec(StarGunner-ram-v4)
 EnvSpec(StarGunner-ramDeterministic-v0)
 EnvSpec(StarGunner-ramDeterministic-v4)
 EnvSpec(StarGunner-ramNoFrameskip-v0)
 EnvSpec(StarGunner-ramNoFrameskip-v4)
 EnvSpec(Tennis-v0)
 EnvSpec(Tennis-v4)
 EnvSpec(TennisDeterministic-v0)
 EnvSpec(TennisDeterministic-v4)
 EnvSpec(TennisNoFrameskip-v0)
 EnvSpec(TennisNoFrameskip-v4)
 EnvSpec(Tennis-ram-v0)
 EnvSpec(Tennis-ram-v4)
 EnvSpec(Tennis-ramDeterministic-v0)
 EnvSpec(Tennis-ramDeterministic-v4)
 EnvSpec(Tennis-ramNoFrameskip-v0)
 EnvSpec(Tennis-ramNoFrameskip-v4)
 EnvSpec(TimePilot-v0)
 EnvSpec(TimePilot-v4)
 EnvSpec(TimePilotDeterministic-v0)
 EnvSpec(TimePilotDeterministic-v4)
 EnvSpec(TimePilotNoFrameskip-v0)
 EnvSpec(TimePilotNoFrameskip-v4)
 EnvSpec(TimePilot-ram-v0)
 EnvSpec(TimePilot-ram-v4)
 EnvSpec(TimePilot-ramDeterministic-v0)
 EnvSpec(TimePilot-ramDeterministic-v4)
 EnvSpec(TimePilot-ramNoFrameskip-v0)
 EnvSpec(TimePilot-ramNoFrameskip-v4)
 EnvSpec(Tutankham-v0)
 EnvSpec(Tutankham-v4)
 EnvSpec(TutankhamDeterministic-v0)
 EnvSpec(TutankhamDeterministic-v4)
 EnvSpec(TutankhamNoFrameskip-v0)
 EnvSpec(TutankhamNoFrameskip-v4)
 EnvSpec(Tutankham-ram-v0)
 EnvSpec(Tutankham-ram-v4)
 EnvSpec(Tutankham-ramDeterministic-v0)
 EnvSpec(Tutankham-ramDeterministic-v4)
 EnvSpec(Tutankham-ramNoFrameskip-v0)
 EnvSpec(Tutankham-ramNoFrameskip-v4)
 EnvSpec(UpNDown-v0)
 EnvSpec(UpNDown-v4)
 EnvSpec(UpNDownDeterministic-v0)
 EnvSpec(UpNDownDeterministic-v4)
 EnvSpec(UpNDownNoFrameskip-v0)
 EnvSpec(UpNDownNoFrameskip-v4)
 EnvSpec(UpNDown-ram-v0)
 EnvSpec(UpNDown-ram-v4)
 EnvSpec(UpNDown-ramDeterministic-v0)
 EnvSpec(UpNDown-ramDeterministic-v4)
 EnvSpec(UpNDown-ramNoFrameskip-v0)
 EnvSpec(UpNDown-ramNoFrameskip-v4)
 EnvSpec(Venture-v0)
 EnvSpec(Venture-v4)
 EnvSpec(VentureDeterministic-v0)
 EnvSpec(VentureDeterministic-v4)
 EnvSpec(VentureNoFrameskip-v0)
 EnvSpec(VentureNoFrameskip-v4)
 EnvSpec(Venture-ram-v0)
 EnvSpec(Venture-ram-v4)
 EnvSpec(Venture-ramDeterministic-v0)
 EnvSpec(Venture-ramDeterministic-v4)
 EnvSpec(Venture-ramNoFrameskip-v0)
 EnvSpec(Venture-ramNoFrameskip-v4)
 EnvSpec(VideoPinball-v0)
 EnvSpec(VideoPinball-v4)
 EnvSpec(VideoPinballDeterministic-v0)
 EnvSpec(VideoPinballDeterministic-v4)
 EnvSpec(VideoPinballNoFrameskip-v0)
 EnvSpec(VideoPinballNoFrameskip-v4)
 EnvSpec(VideoPinball-ram-v0)
 EnvSpec(VideoPinball-ram-v4)
 EnvSpec(VideoPinball-ramDeterministic-v0)
 EnvSpec(VideoPinball-ramDeterministic-v4)
 EnvSpec(VideoPinball-ramNoFrameskip-v0)
 EnvSpec(VideoPinball-ramNoFrameskip-v4)
 EnvSpec(WizardOfWor-v0)
 EnvSpec(WizardOfWor-v4)
 EnvSpec(WizardOfWorDeterministic-v0)
 EnvSpec(WizardOfWorDeterministic-v4)
 EnvSpec(WizardOfWorNoFrameskip-v0)
 EnvSpec(WizardOfWorNoFrameskip-v4)
 EnvSpec(WizardOfWor-ram-v0)
 EnvSpec(WizardOfWor-ram-v4)
 EnvSpec(WizardOfWor-ramDeterministic-v0)
 EnvSpec(WizardOfWor-ramDeterministic-v4)
 EnvSpec(WizardOfWor-ramNoFrameskip-v0)
 EnvSpec(WizardOfWor-ramNoFrameskip-v4)
 EnvSpec(YarsRevenge-v0)
 EnvSpec(YarsRevenge-v4)
 EnvSpec(YarsRevengeDeterministic-v0)
 EnvSpec(YarsRevengeDeterministic-v4)
 EnvSpec(YarsRevengeNoFrameskip-v0)
 EnvSpec(YarsRevengeNoFrameskip-v4)
 EnvSpec(YarsRevenge-ram-v0)
 EnvSpec(YarsRevenge-ram-v4)
 EnvSpec(YarsRevenge-ramDeterministic-v0)
 EnvSpec(YarsRevenge-ramDeterministic-v4)
 EnvSpec(YarsRevenge-ramNoFrameskip-v0)
 EnvSpec(YarsRevenge-ramNoFrameskip-v4)
 EnvSpec(Zaxxon-v0)
 EnvSpec(Zaxxon-v4)
 EnvSpec(ZaxxonDeterministic-v0)
 EnvSpec(ZaxxonDeterministic-v4)
 EnvSpec(ZaxxonNoFrameskip-v0)
 EnvSpec(ZaxxonNoFrameskip-v4)
 EnvSpec(Zaxxon-ram-v0)
 EnvSpec(Zaxxon-ram-v4)
 EnvSpec(Zaxxon-ramDeterministic-v0)
 EnvSpec(Zaxxon-ramDeterministic-v4)
 EnvSpec(Zaxxon-ramNoFrameskip-v0)
 EnvSpec(Zaxxon-ramNoFrameskip-v4)
 EnvSpec(CubeCrash-v0)
 EnvSpec(CubeCrashSparse-v0)
 EnvSpec(CubeCrashScreenBecomesBlack-v0)
 EnvSpec(MemorizeDigits-v0)])