import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import tensorflow as tf


#
# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     return s

class LstmEnv(gym.Env):
    def __init__(self,data,history_t = 90):
        self.data = data
        self.history_t = history_t
        self.reset()
        self.curent_price =0
        self.lastSell = 0
        self.lastbuy = 0


    def reset(self):
        self.t = 0
        self.done = False

        self.positions = []
        self.netPnL = [0,4,0]

        self.profits = 0

        self.cur_start = 0
        self.cur_high = 0
        self.cur_low = 0
        self.cur_end = 0
        self.volume = 0
        # 5 features
        zero_day = [self.cur_start,self.cur_high,self.cur_low,self.cur_end,self.volume]
        self.history = [zero_day for _ in range(self.history_t)]
        return self.netPnL, self.history # observation

    def data_len(self):
        return len(self.data)-1

    def select(self,a):
        self.curent_price = self.data.iloc[self.t, :]['close']
        if(self.lastSell>0):
            self.lastSell-=1
        if(self.lastbuy>0):
            self.lastbuy-=1
        if(len(self.positions)==0) or self.lastbuy>0:
            a[2] = -1000
        if(len(self.positions)>=4) or self.lastSell>0 :
            a[1] = -1000
        sel = np.argmax(a)
        if(sel==2):#sell
            self.lastSell = 15 # can not buy with 15day
        if(sel==1):#buy
            self.lastbuy = 15 # can not sell with 15day
        return sel

    # action = 0: stay, 1: buy, 2: sell
    def step(self, action=0):
        cur_price = self.data.iloc[self.t, :]['close']
        cur_volume = self.data.iloc[self.t,:] ['vol']
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
                reward = np.tanh(profit/12)
                #reward = profit
                self.profits += profit
        # set next time
        self.t += 1
        next_close = self.data.iloc[self.t, :]['close']
        next_open = self.data.iloc[self.t, :]['open']
        next_High = self.data.iloc[self.t, :]['high']
        next_Low = self.data.iloc[self.t, :]['low']
        next_Volume = self.data.iloc[self.t, :]['vol']
        self.history.pop(0)
        cur_day = [
            (next_open /cur_price-1)*100,
            (next_High /cur_price-1)*100,
            (next_Low /cur_price-1)*100,
            (next_close /cur_price-1)*100,
            (next_Volume/cur_volume-1)
            ]
        self.history.append(cur_day)
        self.netPnL = 0
        if len(self.positions)>0:
            self.netPnL = (next_close/np.average(self.positions)-1)*100

        # 每个S 是当前的利润和之前历史差价的数组
        return [len(self.positions),4-len(self.positions),self.netPnL] ,self.history, reward, self.done # obs, reward, done

def get_env(params,mode = tf.estimator.ModeKeys.TRAIN):
    data = pd.read_csv(params.data_path)
    data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')
    data = data.set_index('trade_date')
    data = data.sort_index()
    print(data.index.min(), data.index.max())
    data.head()
    date_split = '2015-12-18'
    train = data[:date_split]
    test = data[date_split:]
    if(mode == tf.estimator.ModeKeys.TRAIN):
        return LstmEnv(data,params.history_size)
    else:
        return LstmEnv(data,params.history_size)


#import matplotlib.pyplot as plt
def main(_):
    data = pd.read_csv("data/Stocks/goog.us.txt")
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    print(data.index.min(), data.index.max())
    data.head()
    date_split = '2016-01-01'
    train = data[:date_split]
    test = data[date_split:]

    cur_price = test.iloc[range(0,100), :]['Close']
    #plt.plot(cur_price)


if __name__=="__main__":
    tf.app.run(main)




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