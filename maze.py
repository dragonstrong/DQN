#自定义环境maze


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk     #GUI包
else:
    import tkinter as tk

#tkinter 坐标系和javafx坐标系一样  都是向下

UNIT = 40   # pixels      #像素，也是每个格子的长度
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']    #动作空间
        self.n_actions = len(self.action_space)      #动作个数
        self.n_features = 2                          #状态维数   ，状态为机器人的中心点坐标
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT)) #格式化设置分辨率
        self._build_maze()                          #画图形界面

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)           #画布

        # Tkinter坐标系和javafx一样  都是向下

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):     #[0 40 80 120]   竖线
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):       #横线
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])    #机器人起始位置

        # hell
        hell1_center = origin + np.array([UNIT, UNIT])                  #障碍物1   ，画矩形传入左上角  右下角坐标
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([3*UNIT, UNIT * 2])         #障碍物2
        self.hell2 = self.canvas.create_rectangle(
             hell2_center[0] - 15, hell2_center[1] - 15,
             hell2_center[0] + 15, hell2_center[1] + 15,
             fill='black')

        hell3_center = origin + np.array ([0, UNIT * 3])            # 障碍物3
        self.hell3 = self.canvas.create_rectangle (
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')




        # create oval                                                 #宝藏
        oval_center = origin + UNIT * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect                                           #机器人
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all                                           #将所有元素加入画布
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])                       #机器人复位
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        return np.array(self.canvas.coords(self.rect)[:2])   #返回位置坐标 coords是一个4维数据：左上角和右下角坐标  [:2]取左上角坐标

    def step(self, action):
        s = self.canvas.coords(self.rect)   #获取当前机器人坐标   包含左左上角 和右下角坐标
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] >= UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] <= (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] <= (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] >= UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动机器人

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        elif next_coords in [self.canvas.coords(self.hell2)]:
            reward = -1
            done = True

        elif next_coords in [self.canvas.coords(self.hell3)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        s_ = np.array(next_coords[:2])
        #print('next_coords:',next_coords,'s_:',s_)    #例：next_coords：[125.0, 45.0, 155.0, 75.0] s_: [125.  45.]  next_coords为4维，因为是左上角 和右下角  不是中心点坐标  此处注意，否则存入经验池的数据维数不匹配
        return s_, reward, done,{}

    def render(self):     #渲染
        # time.sleep(0.01)
        self.update()
