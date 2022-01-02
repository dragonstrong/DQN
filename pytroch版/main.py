
import maze
import DQN
import torch
import time
import numpy as np  # 导入numpy
import matplotlib.pyplot as plt




MEMORY_CAPACITY=2000  #经验池大小

env = maze.Maze ()   #创建环境对象

dqn = DQN.DQN (MEMORY_SIZE=MEMORY_CAPACITY,e_greedy_increment=0.0005)  # DQN  贪婪系数动态增长

reward = np.zeros (1000)      #存每次迭代的总奖励
epo = np.arange (1, 1001)
count = 0
for i in range (1000):  # 1000个episode循环
    print ('<<<<<<<<<Episode: %s' % i)
    s = env.reset ()  # 重置环境
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    path = []
    path.append (s)

    while True:  # 开始一个episode (每一个循环代表一步)
        env.render ()  # 显示实验动画
        a = dqn.choose_action (s)  # 输入该步对应的状态s，选择动作
        time.sleep (0.001)
        s_, r, done, info = env.step (a)  # 执行动作，获得反馈,自定义环境只需实现该函数 s_ r
        env.render ()  # 显示实验动画

        dqn.store_transition (s, a, r, s_)  # 存储样本
        episode_reward_sum += r  # 逐步加上一个episode内每个step的reward

        s = s_  # 更新状态
        path.append (s_)

        if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
            dqn.learn ()  # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔200次将评估网络的参数赋给目标网络)

        if done:  # 如果done为True
            break  # 该episode结束

    print (
        'episode%s---reward_sum: %s' % (i, round (episode_reward_sum, 2)))  # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
    # print ('path:',path)

    reward[i] = episode_reward_sum
    if episode_reward_sum > 0:
        count = count + 1


print ("reward>0的共：", count, " epoch")
ave = np.zeros (50)
for i in range (50):
    sum = 0
    for j in range (20 * i, 20 * (i + 1)):
        sum += reward[j]
        ave[i] = sum / 20.0    #每20次算平均奖励

episode = np.arange (1, 51)
plt.plot (episode, ave)
plt.xlabel ("episode")

plt.ylabel ("ave")
plt.show ()

torch.save (dqn, './CartPole.pth')  # 保存模型
# model_dict=torch.load('./CartPole.pth')
env.close ()

