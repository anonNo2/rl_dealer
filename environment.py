import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import tensorflow as tf

#from plotly import tools
#from plotly.graph_objs import *
#from plotly.offline import init_notebook_mode, iplot, iplot_mpl
#init_notebook_mode()
#
# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     return s



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
        self.netPnL = [0,4,0]
        self.history = [0 for _ in range(self.history_t)]
        return self.netPnL + self.history # observation

    def data_len(self):
        return len(self.data)-1

    # action = 0: stay, 1: buy, 2: sell
    def step(self, action=0):
        cur_price = self.data.iloc[self.t, :]['Close']
        reward = 0 #stay
        if action== 1: #buy
            if len(self.positions) >=4:
                reward = -0.1
            else:
                self.positions.append(cur_price)
                avg = np.average(self.positions)
                for i in range(len(self.positions)):
                    self.positions[i] = avg
                reward = -0.05
        elif action == 2: # sell
            if len(self.positions) == 0:
                reward = -0.1 #punished when no position on sell
            else:
                profit = (cur_price/(self.positions.pop(0))-1)*100
                reward = np.tanh(profit/2)
                self.profits += profit
        # set next time
        self.t += 1
        next_price = self.data.iloc[self.t, :]['Close']
        self.history.pop(0)
        self.history.append((next_price /cur_price-1)*100)

        self.netPnL = 0
        if len(self.positions)>0:
            self.netPnL = (next_price/np.average(self.positions)-1)*100

        # 每个S 是当前的利润和之前历史差价的数组
        return [len(self.positions),4-len(self.positions),self.netPnL] + self.history, reward, self.done # obs, reward, done

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


import matplotlib.pyplot as plt

def main(_):
   # print ("hello,world")
   # data = pd.read_csv("data/Stocks/goog.us.txt")
   # data['Date'] = pd.to_datetime(data['Date'])
   # data = data.set_index('Date')
   # print(data.index.min(), data.index.max())
   # data.head()
   # date_split = '2016-01-01'
   # train = data[:date_split]
   # test = data[date_split:]
   #
   # cur_price = test.iloc[range(0,100), :]['Close']
   # plt.plot(cur_price)
   #plt.plot([1,2,3,4],[1,4,9,16],'ro')
   #plt.plot([1,2,3,4],[1,4,9,16],'b-')
   t=np.arange(0.,5.,0.2)
   #ro 红点， b- 蓝色连续线， r-- 缸线， bs 蓝色的方块， g^ 绿色的三角
   line1,line2 = plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')

   plt.show()


if __name__=="__main__":
   main(None)





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