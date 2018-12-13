import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.base import Layer
from tensorflow.python.layers.core import Dense
import numpy as np
from tensorflow.python.training import training_util


class QLstmNetwork(Layer):
    def __init__(self, history_size,hidden_size,action_space_size,is_trainning=True,dropout_rate=0.0
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

        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size,initializer=xavier_initializer())
        self.fc3 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc3",trainable=self.trainable)
        self.fc4 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc4",trainable=self.trainable)

        self.value = Dense(1,activation=None,kernel_initializer=xavier_initializer(),name = "value",trainable=self.trainable)
        self.Q = Dense(self.action_space_size,activation=None,kernel_initializer=xavier_initializer(),name = "Q",trainable=self.trainable)

    def call(self, head, **kwargs):
        history = kwargs.get("history")
        batch_size= tf.shape(history)[0]
        seq_len = tf.shape(history)[1]
        init_state = self.cell.zero_state(batch_size,dtype=tf.float32)
        #his_list = tf.split(history,self.history_size,axis=1)
        result = tf.nn.dynamic_rnn(self.cell,history,initial_state=init_state)
 #       result = tf.nn.static_rnn(self.cell,his_list,init_state,dtype=tf.float32,sequence_length=seq_len)
        state = result[1][1]

        r_1 = tf.concat([head,state],-1)
        value_logits = self.fc3(r_1)
        Q_logits = self.fc4(r_1)
        state_value = self.value(value_logits)
        advantage_value = self.Q(Q_logits)
        advantage_mean = tf.reduce_mean(advantage_value,axis=-1,keep_dims=True)
        Q_value = (advantage_value-advantage_mean) + state_value
        greedy_action = tf.argmax(Q_value,axis=-1)

        inference_value = Q_value*tf.one_hot(self.action,depth= self.action_space_size,axis=-1)
        inference_value = tf.reduce_sum(inference_value,axis = -1)

        td_error = tf.square(self.targetQ - inference_value)
        loss = tf.reduce_mean(td_error)


        lr = tf.minimum(0.0005, 0.0005 / tf.log(999.) * tf.log(tf.cast(self.step, tf.float32) + 1))
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)

        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        updateModel = optimizer.minimize(loss)

        return updateModel,loss,Q_value,greedy_action

class action():
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decrease = 3e-4
        self.epsilon_min = 0.1
        self.start_reduce_epsilon = 200


    def get_action(self,global_steps,greedy_action,env,mode = tf.estimator.ModeKeys.TRAIN):
        if mode != tf.estimator.ModeKeys.TRAIN:
            return env.select(greedy_action)
        # epsilon
        if self.epsilon > self.epsilon_min and global_steps >self.start_reduce_epsilon:
            self.epsilon -= self.epsilon_decrease
        if np.random.rand() > self.epsilon:
            return env.select(greedy_action)
        else:
            return np.random.randint(3)










