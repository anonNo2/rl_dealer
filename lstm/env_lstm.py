import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import tensorflow as tf

#
#用lstm，成功的用profit当reword，profit就是当天涨跌的百分比，傻agent倾向跌了不卖，
#必须逼迫它！
def get_env(params,mode = tf.estimator.ModeKeys.TRAIN):
    data = pd.read_csv(params.data_path)
    data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')

    data = data.set_index('trade_date')
    data = data.sort_index()
    data['10_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=10,min_periods=0).mean()
    data['30_ema'] =data["close"].ewm(adjust=True,ignore_na=False,span=30,min_periods=0).mean()
    data['60_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=60,min_periods=0).mean()
    #data["close"].data["close"].ewm(adjust=True,ignore_na=False,span=60,min_periods=0).mean()
    print(data.index.min(), data.index.max())
    data.head()
    date_split = '2016-12-18'
    train = data[:date_split]
    test = data[date_split:]
    if(mode == tf.estimator.ModeKeys.TRAIN):
        return LstmEnv(train,params.history_size)
    else:
        return LstmEnv(test,params.history_size,tf.estimator.ModeKeys.EVAL)

class LstmEnv(gym.Env):
    def __init__(self,data,history_t = 60,mode = tf.estimator.ModeKeys.TRAIN):
        self.data = data
        self.history_t = history_t
        self.reset()
        self.mode = mode
    def reset(self):
        self.t = 0
        self.profits = 0
        self.curent_price = 0
        self.done = False
        self.positions = 0
        self.in_position_step = 0
        self.buy_price =0.0
        self.head = [0,0,0] #[ispostion,netPnL,buysteps]
        zero_day = self.getFirstDay()
        self.history = [zero_day for _ in range(self.history_t)]
        return self.head, self.history # observation

    def data_len(self):
        return len(self.data)-1

    def isOutRange(self):
        if(len(self.positions)>0):
            cur_price = self.data.iloc[self.t, :]['close']
            bj = float(cur_price/self.positions[0])
            if(bj>1.1) or bj<0.9:
                return True
        else:
            return False

    # action = 0: stay, 1: buy or sell
    def step(self, action=0):
        self.curent_price = self.data.iloc[self.t, :]['close']
        last_price = self.data.iloc[0 if self.t==0 else self.t-1, :]['close']
        reward = 0
        self.done = False

        if action == 0:  #stay
            reward+=-0.001
            if self.positions==1:
                self.in_position_step+=1


        elif action== 1: #buy or sell
            if self.positions==0: #buy
                self.buy_price = self.curent_price
                self.positions = 1
                reward += -0.1
            else :#sell
                profit = (self.curent_price/self.buy_price-1)*100
                if self.mode == tf.estimator.ModeKeys.EVAL:
                    print ("buy at {}, sell at {}, profit {}".format(self.buy_price,self.curent_price,profit))
                reward += np.tanh(profit/10)+(-0.01)*self.in_position_step
                self.profits += profit
                self.done = True
                self.buy_price = 0
                self.positions = 0
                self.in_position_step = 0

        # set next time
        ema10_p = self.data.iloc[self.t, :]['10_ema']
        ema30_p = self.data.iloc[self.t, :]['30_ema']
        ema60_p = self.data.iloc[self.t, :]['60_ema']
        cur_volume = self.data.iloc[self.t,:] ['vol']

        self.t += 1
        next_close = self.data.iloc[self.t, :]['close']
        next_open = self.data.iloc[self.t, :]['open']
        next_High = self.data.iloc[self.t, :]['high']
        next_Low = self.data.iloc[self.t, :]['low']
        next_Volume = self.data.iloc[self.t, :]['vol']
        ema10 = self.data.iloc[self.t, :]['10_ema']
        ema30 = self.data.iloc[self.t, :]['30_ema']
        ema60 = self.data.iloc[self.t, :]['60_ema']

        self.history.pop(0)
        cur_day = [
            (next_open /self.curent_price-1)*100, #上涨1个点算0.1
            (next_High /self.curent_price-1)*100,
            (next_Low /self.curent_price-1)*100,
            (next_close /self.curent_price-1)*100,
            (ema10 /ema10_p-1)*100,
            (ema30 /ema30_p-1)*100,
            (ema60 /ema60_p-1)*100,
            (next_Volume/cur_volume-1)
            ]

        self.history.append(cur_day)
        netPnL = 0
        if self.positions==1:
            netPnL = ((next_close/self.buy_price)-1)*100

        return [self.positions,netPnL,float(self.in_position_step)/20.0] ,self.history, reward, self.done # obs, reward, done

    def getFirstDay(self):
        self.cur_start = 0
        self.cur_high = 0
        self.cur_low = 0
        self.cur_end = 0
        self.volume = 0
        self.ema10 = 0
        self.ema30 = 0
        self.ema60 = 0
        # 8 features
        zero_day = [self.cur_start,self.cur_high,self.cur_low,self.cur_end,self.volume,self.ema10,self.ema30,self.ema60]
        return zero_day





#import matplotlib.pyplot as plt
def main(_):
    data = pd.read_csv("../china_stock/000002.SZ.csv")
    data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')
    data = data.set_index('trade_date')
    data = data.sort_index()
    data['10_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=10,min_periods=0).mean()
    data['30_ema'] =data["close"].ewm(adjust=True,ignore_na=False,span=30,min_periods=0).mean()
    data['60_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=60,min_periods=0).mean()
    print(data.index.min(), data.index.max())
    data.head()
    date_split = '2016-01-01'
    train = data[ '2014-01-01': '2015-12-01']
    print (train)
    test = data[date_split:]

    #cur_price = test.iloc[range(0,100), :]['Close']
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