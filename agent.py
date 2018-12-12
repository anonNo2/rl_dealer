import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.base import Layer
from tensorflow.python.layers.core import Dense
import numpy as np


class QNetwork(Layer):
    def __init__(self, feature_size,hidden_size,action_space_size,is_trainning=True,dropout_rate=0.0
                 , name="Qnet"
                 , dtype=tf.float32):
        super(QNetwork,self).__init__(is_trainning, name, dtype)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape = [None],dtype=tf.int32)

    def build(self, _):
        half_hidden_size = self.hidden_size//2
        self.fc1 = Dense(self.hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name="fc1",trainable=self.trainable)
        self.fc2 = Dense(self.hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc2",trainable=self.trainable)
        self.fc3 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc3",trainable=self.trainable)
        self.fc4 = Dense(half_hidden_size,activation=tf.nn.relu,kernel_initializer=xavier_initializer(),name = "fc4",trainable=self.trainable)

        self.value = Dense(1,activation=None,kernel_initializer=xavier_initializer(),name = "value",trainable=self.trainable)
        self.Q = Dense(self.action_space_size,activation=None,kernel_initializer=xavier_initializer(),name = "Q",trainable=self.trainable)

    def call(self, inputs, **kwargs):
        r_1 = self.fc2(self.fc1(inputs))
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
        optimizer = tf.train.AdamOptimizer()
        updateModel = optimizer.minimize(loss)

        return updateModel,loss,Q_value,greedy_action

class action():
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decrease = 1e-3
        self.epsilon_min = 0.1
        self.start_reduce_epsilon = 200


    def get_action(self,global_steps,greedy_action,mode = tf.estimator.ModeKeys.TRAIN):
        if mode != tf.estimator.ModeKeys.TRAIN:
            return greedy_action
        # epsilon
        if self.epsilon > self.epsilon_min and global_steps >self.start_reduce_epsilon:
            self.epsilon -= self.epsilon_decrease
        if np.random.rand() > self.epsilon:
            return greedy_action
        else:
            return np.random.randint(3)










