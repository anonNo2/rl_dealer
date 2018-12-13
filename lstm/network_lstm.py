import tensorflow as tf

from agent import QNetwork, QLstmNetwork


class model():
    def __init__(self,params,mode = tf.estimator.ModeKeys.TRAIN):
        self.main_network = QLstmNetwork(params.history_size,50,3,True,name = "main")
        self.main_head = tf.placeholder(tf.float32,[None,3])
        self.main_history = tf.placeholder(tf.float32,[None,params.history_size,5])
        self.updateModel,self.loss, self.Q_main, self.A_main = self.main_network( self.main_head,history =self.main_history)

        self.object_network =  QLstmNetwork(params.history_size,50,3,True,name = "object")
        self.object_head = tf.placeholder(tf.float32,[None,3])
        self.object_history = tf.placeholder(tf.float32,[None,params.history_size,5])
        _, _,self.Q_object,_ = self.object_network( self.object_head,history = self.object_history)