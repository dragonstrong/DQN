"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

'''
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk   #GUI库
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super (Maze, self).__init__ ()
        self.action_space = ['u', 'd', 'l', 'r']  # 动作空间
        self.n_actions = len (self.action_space)  # 动作个数
        self.n_features = 2  # 状态维数
        self.title ('maze')
        self.geometry ('{0}x{1}'.format (MAZE_H * UNIT, MAZE_H * UNIT))   #格式化设置分辨率
        self._build_maze ()    #创建GUI

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)       #画布

        # create grids                                #线
        for c in range(0, MAZE_W * UNIT, UNIT):   #[0 40 80 120]   #竖线
            print(c)
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)


        for r in range(0, MAZE_H * UNIT, UNIT):   #横线
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell    #障碍物
        hell1_center = origin + np.array([UNIT * 2, UNIT])  #(100,60)   坐标系和javafx的相同
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')                                          #左上角 右下角  建矩形
         #hell

        hell2_center = origin + np.array([UNIT, UNIT * 2])   #（60，100）
        self.hell2 = self.canvas.create_rectangle(
             hell2_center[0] - 15, hell2_center[1] - 15,
             hell2_center[0] + 15, hell2_center[1] + 15,
             fill='black')


        # create oval  #宝藏
        oval_center = origin + UNIT * 2        #圆心坐标   (100,100)
        self.oval = self.canvas.create_oval(
            oval_center[0] - 20, oval_center[1] - 20,         #20半径
            oval_center[0] + 20, oval_center[1] + 20,
            fill='yellow')                           #圆外矩形左上角与右下角坐标


        # create red rect     #机器人
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()    #打包所有元素

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)     #复位机器人
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)  #返回两个图形 左上角坐标之差  与总长之比

    def step(self, action):
        s = self.canvas.coords(self.rect)  #获取机器人中心点坐标
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] >=UNIT:    #判断上边界
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] <=(MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] <= (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] >= UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent   第2、3个参数为横纵坐标改变量

        next_coords = self.canvas.coords(self.rect)  # next state   #获取机器人下一位置中心点坐标

        # reward function
        if next_coords == self.canvas.coords(self.oval):    #机器人中心点和宝石重合
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:   #在障碍物1
            reward = -1
            done = True

        #elif next_coords in [self.canvas.coords(self.hell2)]:   #在障碍物2
           # reward = -1
            #done = True

        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()



'''

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  #动作空间
        self.n_actions = len(self.action_space)  #动作个数
        self.n_features = 2     #状态维数
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT)) #格式化设置分辨率
        self._build_maze()

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
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT, UNIT])                  #障碍物1
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

        hell3_center = origin + np.array ([0, UNIT * 3])  # 障碍物2
        self.hell3 = self.canvas.create_rectangle (
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')




        # create oval
        oval_center = origin + UNIT * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
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

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

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
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


