import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import tensorflow as tf
import random
def get_env(data_path,mode = tf.estimator.ModeKeys.TRAIN):
    if(mode ==tf.estimator.ModeKeys.EVAL):
        return LstmEnv("../eval_data",history_t = 60,mode = tf.estimator.ModeKeys.EVAL)
    return LstmEnv(data_path,history_t = 60,mode = mode)
import os

class LstmEnv(gym.Env):
    def __init__(self,data_path,history_t = 60,mode = tf.estimator.ModeKeys.TRAIN):
        list = os.listdir(data_path) #列出文件夹下所有的目录与文件
        self.file_list = []
        for i in range(0,len(list)):
            path = os.path.join(data_path,list[i])
            if os.path.isfile(path) and path.endswith("csv"):
                self.file_list.append(path)
        self.history_t = history_t
        self.rewards = []
        #self.reset(mode = 0)
        self.mode = mode

    def get_data(self):
        y=list(range(len(self.file_list)))
        index = random.sample(y, 1)
        file =self.file_list[index[0]]
        print ("use file {} as input".format(file))
        #data = pd.read_csv(file)
        #data = pd.read_csv("../china_stock/000858_1.SZ.csv")
        data = pd.read_csv("../eval_data/000002.SZ.csv")
        data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')
        data = data.set_index('trade_date')
        data = data.sort_index()
        vol = data['vol']
        dividee = vol.max() - vol.min()
        data['vol_norm'] = (vol - vol.min()) /dividee
        data['5_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=5,min_periods=0).mean()
        data['10_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=10,min_periods=0).mean()
        data['30_ema'] =data["close"].ewm(adjust=True,ignore_na=False,span=30,min_periods=0).mean()
        data['60_ema'] = data["close"].ewm(adjust=True,ignore_na=False,span=60,min_periods=0).mean()
        return data
    def reset(self,mode=1):
        # if mode==0:
        #     self.data = None
        # else:
        self.data = self.get_data()
        self.t = 0
        self.profits = 0
        self.curent_price = 0
        self.done = False
        self.positions = 0
        self.in_position_step = 0
        self.buy_price =0.0

        if(len(self.rewards)>0) and self.mode == tf.estimator.ModeKeys.EVAL:
            #print("avg reword per epoch is {}".format(np.average(self.rewards)))
            self.rewards = []


        self.head = [0,0,0] #[ispostion,netPnL,buysteps]
        zero_day = self.getFirstDay()
        self.history = [zero_day for _ in range(self.history_t)]
        return self.head, self.history # observation

    def data_len(self):
        return len(self.data)-1

    def isOutRange(self):
        #return False
        if(self.positions==1):
            self.curent_price = self.data.iloc[self.t, :]['close']
            netPnL = ((self.curent_price/self.buy_price)-1)*100
            if(netPnL<-7):
                return True
        return False

    def inblock(self):
        if(self.positions==1):
            # self.curent_price = self.data.iloc[self.t, :]['close']
            # netPnL = ((self.curent_price/self.buy_price)-1)*100
            # if(netPnL<2)and netPnL>-1:
            #     return True
            if self.in_position_step>0 and self.in_position_step<=3:
                return True
            else:
                return False
    # action = 0: stay, 1: buy or sell
    def step(self, action=0,confidence = 0):
        ema5_current = self.data.iloc[self.t, :]['5_ema']
        ema5_last = self.data.iloc[0 if self.t==0 else self.t-1, :]['5_ema']
        self.curent_price = self.data.iloc[self.t, :]['close']
        reward = 0
        self.done = False

        if action == 0:  #stay
            # reward+=-0.001
            if self.positions==1:
                self.in_position_step+=1
                delta = (ema5_current-ema5_last)/ema5_current
                reward +=  delta * 3#0 if np.abs(delta)<0.01 else np.tanh(delta*0.2)  #0.2-0.23???

            else:
                delta = (ema5_last-ema5_current)/ema5_current
                reward += delta * 3# if np.abs(delta)<0.01 else np.tanh(delta*0.2)


        elif action== 1: #buy or sell
            if self.positions==0: #buy
                self.buy_price = self.curent_price
                self.positions = 1
                if self.mode == tf.estimator.ModeKeys.EVAL:
                    print ("buy at {}, price {},confident = {}".format(self.t,self.curent_price,confidence))
                #self.done = True
                #reward+=-0.9
            else :#sell
                profit = (self.curent_price/self.buy_price-1)*100
                if self.mode == tf.estimator.ModeKeys.EVAL:
                    print ("sell at {}, price {}, profit {},confident = {}".format(self.t,self.curent_price,profit,confidence))
                reward = 0#2.14*np.tanh(profit/10)

                self.profits += profit
                #self.done = True
                self.buy_price = 0
                self.positions = 0
                self.in_position_step = 0

        self.rewards.append(reward)

        # set next time
        ema5_p = self.data.iloc[self.t, :]['5_ema']
        ema10_p = self.data.iloc[self.t, :]['10_ema']
        ema30_p = self.data.iloc[self.t, :]['30_ema']
        ema60_p = self.data.iloc[self.t, :]['60_ema']
        cur_volume = self.data.iloc[self.t,:] ['vol_norm']

        self.t += 1
        next_close = self.data.iloc[self.t, :]['close']
        # next_open = self.data.iloc[self.t, :]['open']
        # next_High = self.data.iloc[self.t, :]['high']
        # next_Low = self.data.iloc[self.t, :]['low']
        next_Volume = self.data.iloc[self.t, :]['vol_norm']
        ema5 = self.data.iloc[self.t, :]['5_ema']
        ema10 = self.data.iloc[self.t, :]['10_ema']
        ema30 = self.data.iloc[self.t, :]['30_ema']
        ema60 = self.data.iloc[self.t, :]['60_ema']

        self.history.pop(0)
        cur_day = [
            # (next_open /self.curent_price-1)*100, #上涨1个点算0.1
            # (next_High /self.curent_price-1)*100,
            # (next_Low /self.curent_price-1)*100,
            (next_close /self.curent_price-1)*100,
            (ema5 /ema5_p-1)*100,
            (ema10 /ema10_p-1)*100,
            (ema30 /ema30_p-1)*100,
            (ema60 /ema60_p-1)*100,
            #(next_Volume/cur_volume-1)
            next_Volume
        ]

        self.history.append(cur_day)
        netPnL = 0
        if self.positions==1:
            netPnL = ((next_close/self.buy_price)-1)*100

        return [self.positions,netPnL,float(self.in_position_step)/20.0] ,self.history, reward, self.done # obs, reward, done

    def getFirstDay(self):
        # self.cur_start = 0
        # self.cur_high = 0
        # self.cur_low = 0
        self.cur_end = 0
        self.volume = 0
        self.ema5 = 0
        self.ema10 = 0
        self.ema30 = 0
        self.ema60 = 0
        # 8 features
        zero_day = [self.cur_end,self.volume,self.ema5,self.ema10,self.ema30,self.ema60]
        return zero_day





#import matplotlib.pyplot as plt
def main(_):
    data = pd.read_csv("../china_stock/000002.SZ.csv")
    data['trade_date'] = pd.to_datetime(data['trade_date'],format='%Y%m%d')
    data = data.set_index('trade_date')
    data = data.sort_index()
    vol = data['vol']
    dividee = vol.max() - vol.min()
    data['vol_norm'] = (vol - vol.min()) /dividee
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