import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import tensorflow as tf

class env(gym.Env):
    def __init__(self,data,history_t = 90):
        self.data = data
        self.history_t = history_t
        self.reset()


    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.netPnL = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.netPnL] + self.history # observation

    # action = 0: stay, 1: buy, 2: sell
    def step(self, action=0):
        cur_price = self.data.iloc[self.t, :]['Close']
        reward = 0 #stay
        if action== 1: #buy
            self.positions.append(cur_price)
        elif action == 2: # sell
            if len(self.positions) == 0:
                reward = -1 #punished when no position on sell
            else:
                profits = 0
                for p in self.positions:
                    profits += (cur_price - p)
                reward += profits
                self.profits += profits
                self.positions = []
        # set next time
        self.t += 1
        next_price = self.data.iloc[self.t, :]['Close']
        self.netPnL = 0
        for p in self.positions:
            self.netPnL += (next_price - p)
        self.history.pop(0)
        self.history.append(next_price - cur_price)
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        # 每个S 是当前的利润和之前历史差价的数组
        return [self.netPnL] + self.history, reward, self.done # obs, reward, done

def get_env(params,mode = tf.estimator.ModeKeys.TRAIN):
    data = pd.read_csv(params.data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    print(data.index.min(), data.index.max())
    data.head()
    date_split = '2016-01-01'
    train = data[:date_split]
    test = data[date_split:]
    if(mode == tf.estimator.ModeKeys.TRAIN):
        return env(train,params.history_size)
    else:
        return env(test,params.history_size)




# 历程重现
# 这个类赋予了网络存储、重采样来进行训练的能力
class Experience():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])  # 5 是 s，a，r，s_1, done