import tensorflow as tf

from lstm.agent import QLstmNetwork


class model():
    def __init__(self,params,mode = tf.estimator.ModeKeys.TRAIN):
        self.main_network = QLstmNetwork(params.history_size,50,2,True,name = "main")
        self.main_head = tf.placeholder(tf.float32,[None,params.head_size])
        self.main_history = tf.placeholder(tf.float32,[None,params.history_size,8])
        self.updateModel,self.loss, self.Q_main, self.A_main = self.main_network( self.main_head,history =self.main_history)

        self.object_network =  QLstmNetwork(params.history_size,50,2,True,name = "object")
        self.object_head = tf.placeholder(tf.float32,[None,params.head_size])
        self.object_history = tf.placeholder(tf.float32,[None,params.history_size,8])
        _, _,self.Q_object,_ = self.object_network( self.object_head,history = self.object_history)