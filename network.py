import tensorflow as tf

from agent import QNetwork


class model():
    def __init__(self,params,mode = tf.estimator.ModeKeys.TRAIN):
        self.main_network = QNetwork(params.history_size,100,3,True,name = "main")
        self.main_input = tf.placeholder(tf.float32,[None,params.history_size+3])
        self.updateModel,self.loss, self.Q_main, self.A_main = self.main_network( self.main_input)

        self.object_network =  QNetwork(params.history_size,100,3,True,name = "object")
        self.object_input = tf.placeholder(tf.float32,[None,params.history_size+3])
        _, _,self.Q_object,_ = self.object_network( self.object_input)