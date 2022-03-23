import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math


class TwoRoom(object):
    def __init__(self, maxsteps=50):
        # self.map_size = 13
        self.maxsteps = maxsteps
        self.height = 5
        self.weight = 15
        self.num_agent = 1
        self.state = []
        self.goal  = []
        self.subgoal = []
        self.subgoal_nature = [] # [1, 0]表示先加分点，后减分点
        self.occupancy = np.zeros((self.height, self.weight))
        self.count = 0
        self.c = None  # subgoal是加分或减分顺序
        self.reach1 = 0 # 是否到达加分点
        self.reach2 = 0 # 是否到达减分点

    def reset(self):
        self.state = []
        self.goal = []
        self.subgoal = []
        self.subgoal_nature = []
        self.occupancy = np.zeros((self.height, self.weight))
        self.count = 0
        self.reach1 = 0
        self.reach2 = 0
        self.c = random.randint(1, 2)
        if self.c == 1:
            # 红绿
            subgoal_nature = [1, 0]
            self.subgoal_nature.append(subgoal_nature)
        else:
            # 绿红
            subgoal_nature = [0, 1]
            self.subgoal_nature.append(subgoal_nature)

        # 产生初始状态
        for i in range(self.num_agent):
            # 随机生成初始状态和目标状态，且二者不能相等
            state = [random.randint(1, 3), 1]
            self.state.append(state)
            # subgoal1表示红色球的位置，subgoal2表示蓝色球的位置
            if self.c == 1:
                # 红绿
                subgoal1 = [random.randint(1, 3), 5]
                self.subgoal.append(subgoal1)
                subgoal2 = [random.randint(1, 3), 9]
                self.subgoal.append(subgoal2)
            elif self.c == 2:
                # 绿红
                subgoal1 = [random.randint(1, 3), 9]
                self.subgoal.append(subgoal1)
                subgoal2 = [random.randint(1, 3), 5]
                self.subgoal.append(subgoal2)

            goal = [random.randint(1, 3), 13]
            self.goal.append(goal)

        # 设置墙壁
        for i in range(self.weight):
            self.occupancy[0][i] = 1
            self.occupancy[self.height - 1][i] = 1
        for j in range(self.height):
            self.occupancy[j][0] = 1
            self.occupancy[j][self.weight - 1] = 1
            if j in [1, 3]:
                # self.occupancy[j][3] = 1
                self.occupancy[j][7] = 1
                # self.occupancy[j][11] = 1

        return self.get_state()

    def get_env_info(self):
        return 0

    def get_state(self):
        # [[agent坐标],[subgoal1坐标],[subgoal2坐标],[goal坐标]],  nature可以不要
        state = np.zeros((1, 10))
        state[0, 0] = self.state[0][0] / (self.height - 2)
        state[0, 1] = self.state[0][1] / (self.weight - 2)
        state[0, 2] = self.subgoal[0][0] / (self.height - 2)
        state[0, 3] = self.subgoal[0][1] / (self.weight - 2)
        state[0, 4] = self.subgoal[1][0] / (self.height - 2)
        state[0, 5] = self.subgoal[1][1] / (self.weight - 2)
        state[0, 6] = self.goal[0][0] / (self.height - 2)
        state[0, 7] = self.goal[0][1] / (self.weight - 2)
        state[0, 8] = self.subgoal_nature[0][0]
        state[0, 9] = self.subgoal_nature[0][1]

        return state

    def get_obs(self):
        obs = np.zeros((self.num_agent, 4+self.num_agent))
        for i in range(self.num_agent):
            obs[i, 0] = self.state[i][0] / (self.height - 2)
            obs[i, 1] = self.state[i][1] / (self.weight - 2)
            obs[i, 2] = self.goal[i][0] / (self.height - 2)
            obs[i, 3] = self.goal[i][1] / (self.weight - 2)
            for j in range(self.num_agent):
                obs[i, 4+j] = self.order[i][j]
        return obs

    def get_reward(self, state):
        sparse_reward = True
        now = state[0:2].tolist()
        subgoal1 = state[2:4].tolist()
        subgoal2 = state[4:6].tolist()
        goal = state[6:8].tolist()
        if sparse_reward:
            reward = -1
            if now == goal:
                reward = 20
                print('Reach goal')
            if now == subgoal1 and self.reach1 == 0:
                reward = 20
                self.reach1 = 1
                print('++')
            if now == subgoal2 and self.reach2 == 0:
                reward = -20
                self.reach2 = 1
                print('--')
        else:
            if self.reach == 0 and state[0:2].tolist() != state[2:4].tolist():
                reward = -self.sqr_dist(state[0:2].tolist(), state[2:4].tolist())
            if self.reach == 0 and state[0:2].tolist() == state[2:4].tolist():
                reward = 1
                self.reach = 1
                print('Reach subgoal: ', self.c)
            if self.reach == 1 and state[0:2].tolist() != state[4:6].tolist():
                reward = -self.sqr_dist(state[0:2].tolist(), state[4:6].tolist())
            if self.reach == 1 and state[0:2].tolist() == state[4:6].tolist():
                reward = 10
                self.reach = 2
                print('Reach goal')
            if self.reach == 2:
                reward = -self.sqr_dist(state[0:2].tolist(), state[4:6].tolist())
        return reward

    def step(self, action_list):
        for i in range(self.num_agent):
            # agent_i move
            if action_list[i] == 0:  # move up
                if self.occupancy[self.state[i][0] - 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] - 1
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.state[i][0] + 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] + 1
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.state[i][0]][self.state[i][1] - 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] - 1
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.state[i][0]][self.state[i][1] + 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] + 1
            elif action_list[i] == 4:  # stay
                pass

        reward = self.get_reward(self.get_state()[0])
        done = False
        self.count += 1
        if (self.state[0][0] == self.goal[0][0] and self.state[0][1] == self.goal[0][1]) or self.count >= self.maxsteps:
            done = True
        return self.get_state(), reward, done, {}

    def sqr_dist(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1]))

    def get_global_obs(self):
        obs = np.zeros((self.height, self.weight, 4))
        for i in range(self.height):
            for j in range(self.weight):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i%6 == 0:
                # state
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                # goal
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
                # subgoal1
                obs[self.subgoal[0][0], self.subgoal[0][1], 0] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 1] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 2] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 3] = 0.0
                # subgoal2
                obs[self.subgoal[1][0], self.subgoal[1][1], 0] = 1.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 1] = 0.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 2] = 1.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 3] = 1.0




        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        obs = self.get_global_obs()
        enlarge = 40
        henlarge = int(enlarge/2)
        qenlarge = int(enlarge/8)
        new_obs = np.ones((self.height*enlarge, self.weight*enlarge, 3))
        for i in range(self.height):
            for j in range(self.weight):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                # 方形agent及其圆形目标
                # 蓝色start
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                # 蓝色goal
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i+ 1) * henlarge), henlarge, (255, 0, 0), -1)
                # 红色++
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 0, 255), -1)
                    # order = str(self.order[0].index(1) + 1)
                    # cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 绿色--
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 0), -1)
                    # order = str(self.order[1].index(1) + 1)
                    # cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('image', new_obs)
        cv2.waitKey(100)


class TwoRoomTest(object):
    def __init__(self, maxsteps=50):
        # self.map_size = 13
        self.maxsteps = maxsteps
        self.height = 5
        self.weight = 15
        self.num_agent = 1
        self.state = []
        self.goal  = []
        self.subgoal = []
        self.subgoal_nature = [] # [1, 0]表示先加分点，后减分点
        self.occupancy = np.zeros((self.height, self.weight))
        self.count = 0
        self.c = None  # subgoal是加分或减分顺序
        self.reach1 = 0 # 是否到达加分点
        self.reach2 = 0 # 是否到达减分点

    def reset(self):
        self.state = []
        self.goal = []
        self.subgoal = []
        self.subgoal_nature = []
        self.occupancy = np.zeros((self.height, self.weight))
        self.count = 0
        self.reach1 = 0
        self.reach2 = 0
        self.c = random.randint(3, 4)
        if self.c == 3:
            # 红绿
            subgoal_nature = [0, 0]
            self.subgoal_nature.append(subgoal_nature)
        elif self.c == 4:
            # 绿红
            subgoal_nature = [1, 1]
            self.subgoal_nature.append(subgoal_nature)

        # 产生初始状态
        for i in range(self.num_agent):
            # 随机生成初始状态和目标状态，且二者不能相等
            state = [random.randint(1, 3), 1]
            self.state.append(state)
            # subgoal1表示红色球的位置，subgoal2表示蓝色球的位置
            if self.c == 3:
                # 红绿(红点在room的四个角落）
                subgoal1_c = random.randint(1, 4)
                if subgoal1_c == 1:
                    subgoal1 = [1, 4]
                elif subgoal1_c == 2:
                    subgoal1 = [1, 6]
                elif subgoal1_c == 3:
                    subgoal1 = [3, 4]
                elif subgoal1_c == 4:
                    subgoal1 = [3, 6]
                self.subgoal.append(subgoal1)
                subgoal2 = [random.randint(1, 3), 9]
                self.subgoal.append(subgoal2)
            elif self.c == 4:
                # 绿红(红点在room的四个角落）
                subgoal1_c = random.randint(1, 4)
                if subgoal1_c == 1:
                    subgoal1 = [1, 8]
                elif subgoal1_c == 2:
                    subgoal1 = [1, 10]
                elif subgoal1_c == 3:
                    subgoal1 = [3, 8]
                elif subgoal1_c == 4:
                    subgoal1 = [3, 10]
                self.subgoal.append(subgoal1)
                subgoal2 = [random.randint(1, 3), 5]
                self.subgoal.append(subgoal2)
            # elif self.c == 4:
            #     # 红绿(红点在room的四个角落）
            #     subgoal1_c = random.randint(1, 4)
            #     if subgoal1_c == 1:
            #         subgoal1 = [1, 4]
            #     elif subgoal1_c == 2:
            #         subgoal1 = [1, 6]
            #     elif subgoal1_c == 3:
            #         subgoal1 = [3, 4]
            #     elif subgoal1_c == 4:
            #         subgoal1 = [3, 6]
            #     self.subgoal.append(subgoal1)
            #     subgoal2 = [random.randint(1, 3), 9]
            #     self.subgoal.append(subgoal2)

            goal = [random.randint(1, 3), 13]
            self.goal.append(goal)

        # 设置墙壁
        for i in range(self.weight):
            self.occupancy[0][i] = 1
            self.occupancy[self.height - 1][i] = 1
        for j in range(self.height):
            self.occupancy[j][0] = 1
            self.occupancy[j][self.weight - 1] = 1
            if j in [1, 3]:
                # self.occupancy[j][3] = 1
                self.occupancy[j][7] = 1
                # self.occupancy[j][11] = 1

        return self.get_state()

    def get_env_info(self):
        return 0

    def get_state(self):
        # [[agent坐标],[subgoal1坐标],[subgoal2坐标],[goal坐标]],  nature可以不要
        state = np.zeros((1, 10))
        state[0, 0] = self.state[0][0] / (self.height - 2)
        state[0, 1] = self.state[0][1] / (self.weight - 2)
        state[0, 2] = self.subgoal[0][0] / (self.height - 2)
        state[0, 3] = self.subgoal[0][1] / (self.weight - 2)
        state[0, 4] = self.subgoal[1][0] / (self.height - 2)
        state[0, 5] = self.subgoal[1][1] / (self.weight - 2)
        state[0, 6] = self.goal[0][0] / (self.height - 2)
        state[0, 7] = self.goal[0][1] / (self.weight - 2)
        state[0, 8] = self.subgoal_nature[0][0]
        state[0, 9] = self.subgoal_nature[0][1]

        return state

    def get_obs(self):
        obs = np.zeros((self.num_agent, 4+self.num_agent))
        for i in range(self.num_agent):
            obs[i, 0] = self.state[i][0] / (self.height - 2)
            obs[i, 1] = self.state[i][1] / (self.weight - 2)
            obs[i, 2] = self.goal[i][0] / (self.height - 2)
            obs[i, 3] = self.goal[i][1] / (self.weight - 2)
            for j in range(self.num_agent):
                obs[i, 4+j] = self.order[i][j]
        return obs

    def get_reward(self, state):
        sparse_reward = True
        now = state[0:2].tolist()
        subgoal1 = state[2:4].tolist()
        subgoal2 = state[4:6].tolist()
        goal = state[6:8].tolist()
        if sparse_reward:
            reward = -1
            if now == goal:
                reward = 20
                print('Reach goal')
            if now == subgoal1 and self.reach1 == 0:
                reward = 20
                self.reach1 = 1
                print('++')
            if now == subgoal2 and self.reach2 == 0:
                reward = -20
                self.reach2 = 1
                print('--')
        else:
            if self.reach == 0 and state[0:2].tolist() != state[2:4].tolist():
                reward = -self.sqr_dist(state[0:2].tolist(), state[2:4].tolist())
            if self.reach == 0 and state[0:2].tolist() == state[2:4].tolist():
                reward = 1
                self.reach = 1
                print('Reach subgoal: ', self.c)
            if self.reach == 1 and state[0:2].tolist() != state[4:6].tolist():
                reward = -self.sqr_dist(state[0:2].tolist(), state[4:6].tolist())
            if self.reach == 1 and state[0:2].tolist() == state[4:6].tolist():
                reward = 10
                self.reach = 2
                print('Reach goal')
            if self.reach == 2:
                reward = -self.sqr_dist(state[0:2].tolist(), state[4:6].tolist())
        return reward

    def step(self, action_list):
        for i in range(self.num_agent):
            # agent_i move
            if action_list[i] == 0:  # move up
                if self.occupancy[self.state[i][0] - 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] - 1
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.state[i][0] + 1][self.state[i][1]] != 1:  # if can move
                    self.state[i][0] = self.state[i][0] + 1
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.state[i][0]][self.state[i][1] - 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] - 1
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.state[i][0]][self.state[i][1] + 1] != 1:  # if can move
                    self.state[i][1] = self.state[i][1] + 1
            elif action_list[i] == 4:  # stay
                pass

        reward = self.get_reward(self.get_state()[0])
        done = False
        self.count += 1
        if (self.state[0][0] == self.goal[0][0] and self.state[0][1] == self.goal[0][1]) or self.count >= self.maxsteps:
            done = True
        return self.get_state(), reward, done, {}

    def sqr_dist(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1]))

    def get_global_obs(self):
        obs = np.zeros((self.height, self.weight, 4))
        for i in range(self.height):
            for j in range(self.weight):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i%6 == 0:
                # state
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                # goal
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
                # subgoal1
                obs[self.subgoal[0][0], self.subgoal[0][1], 0] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 1] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 2] = 1.0
                obs[self.subgoal[0][0], self.subgoal[0][1], 3] = 0.0
                # subgoal2
                obs[self.subgoal[1][0], self.subgoal[1][1], 0] = 1.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 1] = 0.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 2] = 1.0
                obs[self.subgoal[1][0], self.subgoal[1][1], 3] = 1.0

        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        obs = self.get_global_obs()
        enlarge = 40
        henlarge = int(enlarge/2)
        qenlarge = int(enlarge/8)
        new_obs = np.ones((self.height*enlarge, self.weight*enlarge, 3))
        for i in range(self.height):
            for j in range(self.weight):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                # 方形agent及其圆形目标
                # 蓝色start
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                # 蓝色goal
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i+ 1) * henlarge), henlarge, (255, 0, 0), -1)
                # 红色++
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 0, 255), -1)
                    # order = str(self.order[0].index(1) + 1)
                    # cv2.putText(new_obs, order, ((8*j+3) * qenlarge, (8*i+5)*qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 绿色--
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 0), -1)
                    # order = str(self.order[1].index(1) + 1)
                    # cv2.putText(new_obs, order, ((8 * j + 3) * qenlarge, (8 * i + 5) * qenlarge), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('image', new_obs)
        cv2.waitKey(100)
