#DQN类

import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F  # 导入torch.nn.functional
import numpy as np  # 导入numpy


# 定义Net类 (定义网络)
class Net (nn.Module):
    def __init__(self,N_STATES,N_ACTIONS):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super (Net, self).__init__ ()  # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear (N_STATES, 20)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到20个神经元
        self.fc1.weight.data.normal_ (0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

        self.fc2 = nn.Linear (20, 20)  # 设置第二个全连接层(隐藏层到隐藏层): 状态数个神经元到20个神经元
        self.fc2.weight.data.normal_ (0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

        self.out = nn.Linear (20, N_ACTIONS)  # 设置第三个全连接层(隐藏层到输出层): 20个神经元到动作数个神经元
        self.out.weight.data.normal_ (0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数   x为状态
        #x = x.view (-1, 1)
        x = F.relu (self.fc1 (x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = F.relu (self.fc2 (x))  # 连接隐藏层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out (x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value  # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN (object):
    def __init__(self, N_STATES=2,N_ACTIONS=4,learning_rate=0.005,GAMMA=0.6,MEMORY_SIZE=2000,BATCH_SIZE=32,TARGET_REPLACE_ITER = 200,e_greedy=0.9, e_greedy_increment=None ):  # 定义DQN的一系列属性  无默认值的参数放前 有默认值的参数放后
        self.N_STATES=N_STATES  #状态空间维数
        self.N_ACTIONS=N_ACTIONS   #动作个数  ,动作永远编号为1维，所以注意后面存入经验池和取出的数据格式
        self.lr=learning_rate   #学习率
        self.GAMMA = GAMMA  # 折扣因子
        self.MEMORY_SIZE = MEMORY_SIZE  # 经验池大小
        self.BATCH_SIZE=BATCH_SIZE   #每次从经验池抽取的数据条数
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER  #目标网络参数更新频率
        self.epsilon_max = e_greedy    #贪婪算法系数上限
        self.epsilon_increment = e_greedy_increment   #贪婪系数增幅
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  #贪婪算法系数， 如果e_greedy_increment不为None则随着学习次数缓慢增长

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # 统计存入经验池的数据条数

        self.eval_net, self.target_net = Net (N_STATES,N_ACTIONS), Net (N_STATES,N_ACTIONS)  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.memory = np.zeros ((MEMORY_SIZE, N_STATES * 2 +2))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam (self.eval_net.parameters (), lr=learning_rate)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss ()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        # self.loss_func = F.smooth_l1_loss()


    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform () < self.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            x = torch.unsqueeze (torch.FloatTensor (x), 0)
            actions_value = self.eval_net.forward (x)  # 通过对评估网络输入状态x，前向传播获得动作值，即Q值
            action = torch.max (actions_value, -1)[1].data.numpy ()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]  # 输出action的第一个数

        else:  # 随机选择动作
            action = np.random.randint (0, self.N_ACTIONS)  # 这里action随机等于0 1 2 3 (N_ACTIONS = 4)

        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack ((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.MEMORY_SIZE  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict (self.eval_net.state_dict ())  # 将评估网络的参数赋给目标网络
            print ('\ntarget_params_replaced\n')
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice (self.MEMORY_SIZE, self.BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor (b_memory[:, :self.N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor (b_memory[:, self.N_STATES:self.N_STATES + 1].astype (int))    #不是多维数据，不管多少个动作，都编号为1维
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor (b_memory[:, self.N_STATES +1:self.N_STATES +2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor (b_memory[:, -self.N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net (b_s).gather (1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net (b_s_).detach ()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + self.GAMMA * q_next.max (1)[0].view (self.BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func (q_eval, q_target)

        # loss = F.smooth_l1_loss (q_eval, q_target.unsqueeze (1))

        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad ()  # 清空上一步的残余更新参数值
        loss.backward ()  # 误差反向传播, 计算参数更新值
        self.optimizer.step ()  # 更新评估网络的所有参数

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        #print("贪婪系数：",self.epsilon)
