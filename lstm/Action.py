import tensorflow as tf
import numpy as np

class action():
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decrease = 1e-4
        self.epsilon_min = 0.1
        self.start_reduce_epsilon = 200

    def get_action(self,global_steps,greedy_action,env,mode = tf.estimator.ModeKeys.TRAIN):
        # if env.isOutRange():
        #     return 2
        if mode != tf.estimator.ModeKeys.TRAIN:
            return greedy_action
        # epsilon
        if self.epsilon > self.epsilon_min and global_steps >self.start_reduce_epsilon:
            self.epsilon -= self.epsilon_decrease
        if np.random.rand() > self.epsilon:
            return greedy_action
        else:
            return np.random.randint(2)