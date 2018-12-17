import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.base import Layer
from tensorflow.python.layers.core import Dense
import numpy as np
from tensorflow.python.training import training_util


class QLstmNetwork(Layer):
    def __init__(self, history_size,hidden_size,action_space_size=2,is_trainning=True,dropout_rate=0.0
                 , name="Qnet"
                 , dtype=tf.float32):
        super(QLstmNetwork,self).__init__(is_trainning, name, dtype)
        self.history_size = history_size
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape = [None],dtype=tf.int32)
        self.step = tf.placeholder(shape=[],dtype=tf.int32)
        self.scalar_step = 0

    def build(self, _):
        half_hidden_size = self.hidden_size//2

        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size
                                            ,use_peepholes=True
                                            ,initializer=xavier_initializer())
        self.fc3 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc3",trainable=self.trainable)
        self.fc4 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc4",trainable=self.trainable)

        self.buy_value = Dense(1,activation=None,kernel_initializer=xavier_initializer(),name = "value",trainable=self.trainable)
        self.sell_value = Dense(1,activation=None,kernel_initializer=xavier_initializer(),name = "value",trainable=self.trainable)

        self.buy_hold = Dense(self.action_space_size,activation=None,kernel_initializer=xavier_initializer(),name = "Q",trainable=self.trainable)
        self.sell_hold = Dense(self.action_space_size,activation=None,kernel_initializer=xavier_initializer(),name = "Q",trainable=self.trainable)

    def call(self, head, **kwargs):
        history = kwargs.get("history")
        batch_size= tf.shape(history)[0]
        init_state = self.cell.zero_state(batch_size,dtype=tf.float32)
        result = tf.nn.dynamic_rnn(self.cell,history,initial_state=init_state)
        state = result[1][1]

        #head [是否有仓位，netPnL,从买了后走了多远了 float(steps)/20.0] , [batch_size,2]
        isPositionThere, netPnL= tf.split(head,[1,2],axis = 1)
        feature = tf.concat([netPnL,state],-1)

        value_logits = self.fc3(feature)
        Q_logits = self.fc4(feature)


        # buy and sell caculate seperately ,can not mix together, for only one can be choosen at a single time
        position_value = self.sell_value(value_logits)
        position_advantage = self.sell_hold(Q_logits)
        position_advantage_mean = tf.reduce_mean(position_advantage,axis=-1,keep_dims=True)
        position_Q = (position_advantage-position_advantage_mean) + position_value


        empty_value = self.buy_value(value_logits)
        empty_advantage = self.buy_hold(Q_logits)
        empty_advantage_mean = tf.reduce_mean(empty_advantage,axis=-1,keep_dims=True)
        empty_Q = (empty_advantage-empty_advantage_mean) + empty_value

        mirror_position = 1-isPositionThere
        Q = tf.where(tf.cast(tf.concat([isPositionThere,mirror_position],axis=1),tf.bool),position_Q,empty_Q)

        # 0 is hold, both sell and buy is 1
        greedy_action = tf.argmax(Q,axis=-1)

        inference_value = Q*tf.one_hot(self.action,depth= self.action_space_size,axis=-1)
        inference_value = tf.reduce_sum(inference_value,axis = -1)

        td_error = tf.square(self.targetQ - inference_value)
        loss = tf.reduce_mean(td_error)


        lr = tf.minimum(0.001, 0.001 / tf.log(999.) * tf.log(tf.cast(self.step, tf.float32) + 1))
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)

        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        updateModel = optimizer.minimize(loss)

        return updateModel,loss,Q ,greedy_action












